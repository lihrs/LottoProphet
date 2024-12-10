# train_dlt_model.py

import os
import pandas as pd
import numpy as np
import sys
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
import joblib
from tensorflow.keras import backend as K
import subprocess
os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#sys.setdefaultencoding('utf-8')
# 配置
name_path = {
    "dlt": {
        "name": "大乐透",
        "path": "./dlt/"
    }
}
data_file_name = "dlt_history.csv"
model_file_name = "dlt_model.h5"
scaler_X_file = "dlt_scaler_X.pkl"
scaler_y_file = "dlt_scaler_y.pkl"

# 定义回调函数以打印训练日志
class TrainingLogCallback(Callback):
    def __init__(self, log_callback):
        super().__init__()
        self.log_callback = log_callback

    def on_epoch_end(self, epoch, logs=None):
        message = (f"Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, "
                   f"mae={logs.get('mae', 0):.4f}, mse={logs.get('mse', 0):.4f}, "
                   f"val_loss={logs.get('val_loss', 0):.4f}, val_mae={logs.get('val_mae', 0):.4f}, "
                   f"val_mse={logs.get('val_mse', 0):.4f}")
        self.log_callback(message)

def build_model(input_dim, output_dim):
    """
    构建神经网络模型，添加Dropout层
    :param input_dim: 输入特征维度
    :param output_dim: 输出维度
    :return: 编译后的模型
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def load_lottery_data(lottery_name):
    """
    加载指定彩票类型的历史数据
    :param lottery_name: 'ssq' 或 'dlt'
    :return: DataFrame
    """
    file_path = os.path.join(name_path[lottery_name]['path'], data_file_name)
    if not os.path.exists(file_path):
        print(f"数据文件不存在：{file_path}")
        print("请先运行 'fetch_dlt_data.py' 爬取历史数据。")
        exit(1)
    data = pd.read_csv(file_path, encoding="utf-8")
    return data

def preprocess_lottery_data(data, lottery_name):
    """
    根据彩票类型进行特征工程和预处理
    :param data: DataFrame
    :param lottery_name: 'ssq' 或 'dlt'
    :return: X, y
    """
    if lottery_name == "dlt":
        # 创建特征
        data['前区和值'] = data[['红球_1', '红球_2', '红球_3', '红球_4', '红球_5']].astype(int).sum(axis=1)
        data['前区奇数个数'] = data[['红球_1', '红球_2', '红球_3', '红球_4', '红球_5']].astype(int).apply(lambda x: sum(i % 2 != 0 for i in x), axis=1)
        data['后区和值'] = data[['蓝球_1', '蓝球_2']].astype(int).sum(axis=1)

        # 创建目标变量（下期开奖号码）
        for column in ['红球_1', '红球_2', '红球_3', '红球_4', '红球_5', '蓝球_1', '蓝球_2']:
            data[f'目标_{column}'] = data[column].shift(-1)

        data = data.dropna()

        X = data[['前区和值', '前区奇数个数', '后区和值']]
        y = data[[f'目标_{col}' for col in ['红球_1', '红球_2', '红球_3', '红球_4', '红球_5', '蓝球_1', '蓝球_2']]]
    else:
        raise ValueError("不支持的彩票类型！")

    return X, y

def train_lottery_model(lottery_name, log_callback):
    """
    训练模型并保存
    :param lottery_name: 'ssq' 或 'dlt'
    :param log_callback: 回调函数，用于输出日志
    """
    try:
        # 调用数据爬取脚本
        if lottery_name == "dlt":
            fetch_script = os.path.join(name_path["dlt"]["path"], "fetch_dlt_data.py")
        elif lottery_name == "ssq":
            fetch_script = os.path.join(name_path["ssq"]["path"], "fetch_ssq_data.py")
        else:
            raise ValueError("不支持的彩票类型！")

        if os.path.exists(fetch_script):
            log_callback(f"正在运行数据爬取脚本: {fetch_script}")
            process = subprocess.Popen(['python', fetch_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                log_callback(f"数据爬取失败:\n{stderr}")
                return
            log_callback(f"数据爬取完成:\n{stdout}")
        else:
            log_callback(f"数据爬取脚本不存在：{fetch_script}")

        # 加载数据
        data = load_lottery_data(lottery_name)
        log_callback(f"加载数据完成，共{len(data)}条记录。")

        # 特征工程
        X, y = preprocess_lottery_data(data, lottery_name)
        log_callback(f"特征工程完成。特征维度: {X.shape}, 目标维度: {y.shape}")

        # 特征缩放
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        log_callback(f"数据拆分完成。训练集: {X_train.shape}, 测试集: {X_test.shape}")

        # 构建模型
        model = build_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
        log_callback("模型构建完成。")

        # 训练模型
        log_callback("开始训练模型...")
        model.fit(
            X_train, y_train,
            epochs=30,            # 可根据需要调整epochs数量
            batch_size=16,
            validation_split=0.2,
            verbose=0,            # 设置为0以禁用进度条
            callbacks=[TrainingLogCallback(log_callback)]
        )
        log_callback("模型训练完成。")

        # 评估模型
        loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
        log_callback(f"训练完成 - 测试集上的均方误差: {loss:.4f}")
        log_callback(f"训练完成 - 测试集上的平均绝对误差: {mae:.4f}")
        log_callback(f"训练完成 - 测试集上的均方根误差: {np.sqrt(loss):.4f}")

        # 保存模型和缩放器
        model_path = os.path.join(name_path[lottery_name]['path'], model_file_name)
        scaler_X_path = os.path.join(name_path[lottery_name]['path'], scaler_X_file)
        scaler_y_path = os.path.join(name_path[lottery_name]['path'], scaler_y_file)

        model.save(model_path)
        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_y, scaler_y_path)

        log_callback(f"模型已保存至 {model_path}")
        log_callback(f"缩放器已保存至 {scaler_X_path} 和 {scaler_y_path}")

    except Exception as e:
        log_callback(f"模型训练失败: {e}")

    finally:
        # 清理Keras会话
        K.clear_session()

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs(name_path["dlt"]["path"], exist_ok=True)

    # 训练大乐透模型，使用简单的日志回调
    def print_log(message):
        print(message)

    train_lottery_model("dlt", print_log)
