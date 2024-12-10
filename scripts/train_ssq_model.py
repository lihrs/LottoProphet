import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchcrf import CRF

# 配置
DATA_FILE = "./ssq/ssq_history.csv"
MODEL_PATH = "./ssq/ssq_model.pth"

# 数据集定义
class LotteryDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features, self.labels = self.preprocess(self.data)

    def preprocess(self, data):
        # 创建特征
        data['红球和值'] = data[['红球_1', '红球_2', '红球_3', '红球_4', '红球_5', '红球_6']].sum(axis=1)
        data['红球奇数个数'] = data[['红球_1', '红球_2', '红球_3', '红球_4', '红球_5', '红球_6']].apply(lambda row: sum(num % 2 != 0 for num in row), axis=1)

        # 创建目标
        for i in range(6):
            data[f"目标红球_{i+1}"] = data[f"红球_{i+1}"].shift(-1)
        data["目标蓝球"] = data["蓝球"].shift(-1)

        data = data.dropna()

        # 提取特征和标签
        features = data[['红球和值', '红球奇数个数']].values
        labels = data[[f"目标红球_{i+1}" for i in range(6)] + ["目标蓝球"]].values

        # 重塑标签为 (batch_size, sequence_length)
        labels = labels.reshape(-1, 7)  # 6 红球 + 1 蓝球作为时间步

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 模型定义
class LotteryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LotteryModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, x):
        # 调整输入以支持序列（添加时间步维度）
        x = x.unsqueeze(1).repeat(1, 7, 1)  # 重复时间步为 7
        lstm_out, _ = self.lstm(x)  # 输出形状 (batch_size, sequence_length, hidden_dim)
        emissions = self.fc(lstm_out)  # 输出形状 (batch_size, sequence_length, num_classes)
        return emissions

    def compute_loss(self, emissions, labels):
        mask = labels.ne(0)  # 标签不为 0 的位置有效
        return -self.crf(emissions, labels, mask)

    def predict(self, emissions):
        mask = torch.ones(emissions.size()[:2], dtype=torch.bool)  # 全有效
        return self.crf.decode(emissions, mask)

# 训练函数
def train_model():
    # 加载数据
    dataset = LotteryDataset(DATA_FILE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型初始化
    input_dim = 2  # 特征数量
    hidden_dim = 128
    output_dim = 34  # 红球和蓝球的最大值 (1-33 红球 + 1-16 蓝球)

    model = LotteryModel(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for epoch in range(20):
        total_loss = 0
        for features, labels in dataloader:
            emissions = model(features)  # 输出形状为 (batch_size, sequence_length, num_classes)
            loss = model.compute_loss(emissions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至 {MODEL_PATH}")

# 预测函数
def predict():
    # 加载模型
    input_dim = 2
    hidden_dim = 128
    output_dim = 34

    model = LotteryModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 加载数据
    dataset = LotteryDataset(DATA_FILE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for features, _ in dataloader:
            emissions = model(features)  # 输出形状为 (batch_size, sequence_length, num_classes)
            prediction = model.predict(emissions)
            predictions.append(prediction)

    # 格式化预测结果
    for i, pred in enumerate(predictions):
        red_balls = pred[0][:6]
        blue_ball = pred[0][6]
        print(f"预测 {i + 1}: 红球 = {red_balls}, 蓝球 = {blue_ball}")

# 主程序入口
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"数据文件不存在：{DATA_FILE}")
    else:
        print("开始训练模型...")
        train_model()
        print("模型训练完成，开始预测...")
        predict()
