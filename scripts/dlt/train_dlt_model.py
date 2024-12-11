# train_dlt_model.py

import os
import sys
import subprocess
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from loguru import logger
from sklearn.metrics import accuracy_score
import copy

# 配置 loguru
logger.remove()  # 移除默认的处理器
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}", level="INFO")

# ---------------- 配置 ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, "dlt_history.csv")
MODEL_PATH = os.path.join(current_dir, "dlt_model.pth")
SCALER_PATH = os.path.join(current_dir, "scaler_X.pkl")
BATCH_SIZE = 32
EPOCHS = 100  # 增加训练轮数
LEARNING_RATE = 0.001
WINDOW_SIZE = 10
PATIENCE = 10  # 早停耐心值

# 获取项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from model import LstmCRFModel  # 使用绝对导入
except ImportError as e:
    logger.error(f"导入模型类失败: {e}")
    sys.exit(1)

from sklearn.model_selection import train_test_split

class LotteryDataset(Dataset):
    def __init__(self, csv_file, window_size, red_balls=5, blue_balls=2):
        self.data = pd.read_csv(csv_file)

        # 映射列名到代码中预期的列名
        self.data.rename(columns={
            '红球_1': 'Red_1',
            '红球_2': 'Red_2',
            '红球_3': 'Red_3',
            '红球_4': 'Red_4',
            '红球_5': 'Red_5',
            '蓝球_1': 'Blue_1',
            '蓝球_2': 'Blue_2'
        }, inplace=True)

        self.scaler_X = MinMaxScaler()
        self.features, self.labels = self.preprocess(self.data, window_size, red_balls, blue_balls)

    def preprocess(self, data, window_size, red_balls, blue_balls):
        features, labels = [], []
        expected_columns = 1 + red_balls + blue_balls
        if len(data.columns) < expected_columns:
            raise ValueError(f"数据列数不足，当前列数: {len(data.columns)}，期望至少 {expected_columns} 列。")

        for i in range(len(data) - window_size):
            # 特征：选取窗口内的红球和蓝球数据
            feature_window = data.iloc[i:i + window_size, 1:1 + red_balls + blue_balls].values
            features.append(feature_window)

            # 标签：下一期的红球和蓝球
            red_labels_seq = data.iloc[i + window_size, 1:1 + red_balls].values - 1  # 减1使其从0开始
            blue_label = data.iloc[i + window_size, 1 + red_balls:1 + red_balls + blue_balls].values - 1
            combined_labels = np.concatenate((red_labels_seq, blue_label))
            labels.append(combined_labels)

        # 转换为 NumPy 数组并进行缩放
        features_np = np.array(features)  # 形状: (num_samples, window_size, feature_dim)
        features_scaled = self.scaler_X.fit_transform(features_np.reshape(-1, features_np.shape[-1])).reshape(features_np.shape)

        labels_np = np.array(labels)  # 形状: (num_samples, total_labels)

        return (
            torch.tensor(features_scaled, dtype=torch.float32),
            torch.tensor(labels_np, dtype=torch.long)
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
def fetch_data_if_not_exists():
    """
    检查 CSV 文件是否存在，如果不存在，则调用 fetch_dlt_data.py 获取数据
    """
    if not os.path.exists(DATA_FILE):
        logger.info(f"数据文件 {DATA_FILE} 不存在，开始获取数据...")
        fetch_script = os.path.join(current_dir, 'fetch_dlt_data.py')
        if not os.path.exists(fetch_script):
            logger.error(f"数据获取脚本不存在: {fetch_script}")
            sys.exit(1)
        try:
            # 使用当前运行的 Python 解释器
            python_executable = sys.executable
            logger.info(f"运行数据获取脚本: {fetch_script} 使用解释器: {python_executable}")
            subprocess.run([python_executable, fetch_script], check=True)
            logger.info("数据获取完成。")
        except subprocess.CalledProcessError as e:
            logger.error(f"运行数据获取脚本失败: {e}")
            sys.exit(1)
    else:
        logger.info(f"数据文件 {DATA_FILE} 已存在。")

def train_model():
    fetch_data_if_not_exists()

    if not os.path.exists(DATA_FILE):
        logger.error(f"数据文件不存在: {DATA_FILE}")
        sys.exit(1)

    # 数据加载
    logger.info("加载数据...")
    dataset = LotteryDataset(DATA_FILE, WINDOW_SIZE)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = dataset.features.shape[-1]

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 模型初始化并移动到设备
    red_model = LstmCRFModel(input_dim, hidden_dim=128, output_dim=35, output_seq_length=5, num_layers=1).to(device)
    blue_model = LstmCRFModel(input_dim, hidden_dim=128, output_dim=12, output_seq_length=2, num_layers=1).to(device)
    red_optimizer = torch.optim.Adam(red_model.parameters(), lr=LEARNING_RATE)
    blue_optimizer = torch.optim.Adam(blue_model.parameters(), lr=LEARNING_RATE)

    # 学习率调度器，每10轮将学习率降低为原来的一半
    scheduler_red = torch.optim.lr_scheduler.StepLR(red_optimizer, step_size=10, gamma=0.5)
    scheduler_blue = torch.optim.lr_scheduler.StepLR(blue_optimizer, step_size=10, gamma=0.5)

    # 训练过程
    logger.info("开始模型训练...")
    best_val_loss = float('inf')
    best_model_wts = {
        'red_model': copy.deepcopy(red_model.state_dict()),
        'blue_model': copy.deepcopy(blue_model.state_dict())
    }
    trigger_times = 0

    for epoch in range(EPOCHS):
        red_model.train()
        blue_model.train()
        total_red_loss, total_blue_loss = 0, 0

        for features, labels in train_loader:
            # 将数据移动到设备
            features = features.to(device)
            labels = labels.to(device)

            # 红球训练
            red_labels = labels[:, :5]  # (batch_size, 5)
            red_mask = (red_labels >= 0)
            red_loss = red_model(features, red_labels, red_mask)
            red_optimizer.zero_grad()
            red_loss.backward()
            red_optimizer.step()
            total_red_loss += red_loss.item()

            # 蓝球训练
            blue_labels = labels[:, 5:]  # (batch_size, 2)
            blue_mask = (blue_labels >= 0)
            blue_loss = blue_model(features, blue_labels, blue_mask)
            blue_optimizer.zero_grad()
            blue_loss.backward()
            blue_optimizer.step()
            total_blue_loss += blue_loss.item()

        # 计算训练集平均损失
        avg_red_loss = total_red_loss / len(train_loader)
        avg_blue_loss = total_blue_loss / len(train_loader)

        # 验证过程
        red_model.eval()
        blue_model.eval()
        val_red_loss, val_blue_loss = 0, 0
        all_red_preds, all_red_labels = [], []
        all_blue_preds, all_blue_labels = [], []

        with torch.no_grad():
            for features, labels in val_loader:
                # 将数据移动到设备
                features = features.to(device)
                labels = labels.to(device)

                # 红球验证
                red_labels = labels[:, :5]
                red_mask = (red_labels >= 0)
                red_loss = red_model(features, red_labels, red_mask)
                val_red_loss += red_loss.item()
                red_preds = red_model(features)  # 返回的是嵌套列表
                # 展平列表并添加到 all_red_preds
                all_red_preds.extend([pred for sequence in red_preds for pred in sequence])
                all_red_labels.extend(red_labels.cpu().numpy().flatten())

                # 蓝球验证
                blue_labels = labels[:, 5:]
                blue_mask = (blue_labels >= 0)
                blue_loss = blue_model(features, blue_labels, blue_mask)
                val_blue_loss += blue_loss.item()
                blue_preds = blue_model(features)  # 返回的是嵌套列表
                # 展平列表并添加到 all_blue_preds
                all_blue_preds.extend([pred for sequence in blue_preds for pred in sequence])
                all_blue_labels.extend(blue_labels.cpu().numpy().flatten())

        # 计算验证集平均损失
        avg_val_red_loss = val_red_loss / len(val_loader)
        avg_val_blue_loss = val_blue_loss / len(val_loader)

        # 计算准确率
        red_accuracy = accuracy_score(all_red_labels, all_red_preds)
        blue_accuracy = accuracy_score(all_blue_labels, all_blue_preds)

        logger.info(f"Epoch {epoch + 1}: "
                    f"Train 红球 Loss = {avg_red_loss:.4f}, Train 蓝球 Loss = {avg_blue_loss:.4f}, "
                    f"Val 红球 Loss = {avg_val_red_loss:.4f}, Val 蓝球 Loss = {avg_val_blue_loss:.4f}, "
                    f"Val 红球 Accuracy = {red_accuracy:.4f}, Val 蓝球 Accuracy = {blue_accuracy:.4f}")

        # 学习率更新
        scheduler_red.step()
        scheduler_blue.step()

        # 早停检查
        total_val_loss = avg_val_red_loss + avg_val_blue_loss
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_wts['red_model'] = copy.deepcopy(red_model.state_dict())
            best_model_wts['blue_model'] = copy.deepcopy(blue_model.state_dict())
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                logger.info(f"早停触发，训练停止在第 {epoch + 1} 轮")
                break

    # 加载最佳模型权重
    red_model.load_state_dict(best_model_wts['red_model'])
    blue_model.load_state_dict(best_model_wts['blue_model'])

    # 保存模型和缩放器
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "red_model": red_model.state_dict(),
        "blue_model": blue_model.state_dict()
    }, MODEL_PATH)
    joblib.dump(dataset.scaler_X, SCALER_PATH)
    logger.info(f"模型已保存到 {MODEL_PATH}")
    logger.info(f"缩放器已保存到 {SCALER_PATH}")

if __name__ == "__main__":
    logger.info("开始训练模型...")
    train_model()
    logger.info("模型训练完成。")
