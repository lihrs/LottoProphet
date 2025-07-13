# -*- coding: utf-8 -*-
"""
LSTM-CRF model implementation for lottery prediction
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any
from .base import BaseLotteryModel


class CRF(nn.Module):
    """
    Conditional Random Field layer
    """
    
    def __init__(self, num_tags: int, batch_first: bool = False):
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize the transition parameters"""
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'
        
    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'sum') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores"""
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
            
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        # compute the log sum exp of all possible paths
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        
        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        elif reduction == 'mean':
            return llh.mean()
        else:  # reduction == 'token_mean'
            return llh.sum() / mask.float().sum()
            
    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm"""
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
            
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        return self._viterbi_decode(emissions, mask)
        
    def _validate(self, emissions: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')
                
        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
                    
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')
                
    def _compute_score(self, emissions: torch.Tensor, tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()
        
        seq_length, batch_size = tags.shape
        mask = mask.float()
        
        # Start transition score and first emission
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        
        # Transition scores
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
            
        # End transition score
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        
        return score
        
    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        
        seq_length = emissions.size(0)
        
        # Start transition score and first emission; score has size (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            broadcast_score = score.unsqueeze(2)
            
            # Broadcast emission score for every possible current tag
            broadcast_emissions = emissions[i].unsqueeze(1)
            
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # next_score[i, j, k] = score[i, j] + transition_score[j, k] + emission_score[i, k]
            next_score = broadcast_score + self.transitions + broadcast_emissions
            
            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: next_score has size (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            
        # End transition score
        score += self.end_transitions
        
        return torch.logsumexp(score, dim=1)
        
    def _viterbi_decode(self, emissions: torch.Tensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()
        
        seq_length, batch_size = mask.shape
        
        # Start transition and first emission
        score = self.start_transitions + emissions[0]
        history = []
        
        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence
        
        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible last tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            broadcast_score = score.unsqueeze(2)
            
            # Broadcast emission score for every possible current tag
            broadcast_emission = emissions[i].unsqueeze(1)
            
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # next_score[i, j, k] = score[i, j] + transition_score[j, k] + emission_score[i, k]
            next_score = broadcast_score + self.transitions + broadcast_emission
            
            # Find the maximum score over all possible current tag
            next_score, indices = next_score.max(dim=1)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
            
        # End transition score
        score += self.end_transitions
        
        # Now, compute the best path for each sample
        
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        
        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            
            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
                
            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)
            
        return best_tags_list


class LstmCRFModel(BaseLotteryModel, nn.Module):
    """
    LSTM-CRF model for lottery number prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, lottery_type: str, dropout: float = 0.5,
                 bidirectional: bool = True, use_attention: bool = False,
                 use_residual: bool = False):
        BaseLotteryModel.__init__(self, lottery_type)
        nn.Module.__init__(self)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
        # Residual connection
        if use_residual and input_size == lstm_output_size:
            self.residual_projection = None
        elif use_residual:
            self.residual_projection = nn.Linear(input_size, lstm_output_size)
        else:
            self.residual_projection = None
            
        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
        
        # CRF layer
        self.crf = CRF(num_classes, batch_first=True)
        
    def forward(self, x: torch.Tensor, tags: Optional[torch.LongTensor] = None,
                mask: Optional[torch.ByteTensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            tags: Target tags for training (batch_size, seq_len)
            mask: Mask for variable length sequences (batch_size, seq_len)
            
        Returns:
            If tags is provided (training), returns negative log likelihood
            If tags is None (inference), returns predicted tag sequences
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
        # Apply residual connection if enabled
        if self.use_residual and self.residual_projection is not None:
            residual = self.residual_projection(x)
            lstm_out = lstm_out + residual
        elif self.use_residual and self.residual_projection is None:
            lstm_out = lstm_out + x
            
        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # Fully connected layer
        emissions = self.fc(lstm_out)
        
        if tags is not None:
            # Training mode: return negative log likelihood
            return -self.crf(emissions, tags, mask=mask)
        else:
            # Inference mode: return predicted sequences
            return self.crf.decode(emissions, mask=mask)
            
    def train(self, data: Any, **kwargs) -> None:
        """
        Train the LSTM-CRF model
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        # This method should be implemented based on specific training requirements
        self.is_trained = True
        
    def predict(self, **kwargs) -> Tuple[List[int], List[int]]:
        """
        Make predictions using the trained model
        
        Returns:
            Tuple of (red_numbers, blue_numbers)
        """
        # This method should be implemented based on specific prediction requirements
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return [], []
        
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_classes': self.num_classes,
                'lottery_type': self.lottery_type,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'use_attention': self.use_attention,
                'use_residual': self.use_residual
            },
            'is_trained': self.is_trained
        }, filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', False)