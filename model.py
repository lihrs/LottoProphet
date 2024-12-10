# model.py

import torch.nn as nn
from torchcrf import CRF

class LstmCRFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_length, num_layers=1, dropout=0.5):
        super(LstmCRFModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim * output_seq_length)
        self.crf = CRF(output_dim, batch_first=True)
        self.output_seq_length = output_seq_length
        self.output_dim = output_dim

    def forward(self, x, labels=None, mask=None):
        lstm_out, _ = self.lstm(x)  # (batch_size, window_size, hidden_dim)
        final_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        logits = self.fc(final_hidden)  # (batch_size, output_seq_length * output_dim)
        logits = logits.view(-1, self.output_seq_length, self.output_dim)  # (batch_size, output_seq_length, output_dim)

        if labels is not None:
            if mask is not None:
                mask = mask.bool()
            loss = -self.crf(logits, labels, mask=mask)
            return loss
        else:
            predictions = self.crf.decode(logits, mask=mask)
            return predictions
