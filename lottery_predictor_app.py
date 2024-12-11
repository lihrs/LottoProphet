# -*- coding:utf-8 -*-
"""
Author: Zhao Yang
"""

import sys
import os
import io
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton,
    QLabel, QComboBox, QWidget, QTextEdit, QSpinBox, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread
import torch
from torch import nn
from torchcrf import CRF
import joblib
import numpy as np
import subprocess
from model import LstmCRFModel  # 导入模型类

# 设置标准输出和错误输出编码为utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ---------------- 配置 ----------------
name_path = {
    "dlt": {
        "name": "大乐透",
        "path": "./scripts/dlt/",
        "model_file": "dlt_model.pth",
        "scaler_X_file": "scaler_X.pkl",
        "train_script": "train_dlt_model.py",
        "fetch_script": "fetch_dlt_data.py"
    },
    "ssq": {
        "name": "双色球",
        "path": "./scripts/ssq/",
        "model_file": "ssq_model.pth",
        "scaler_X_file": "scaler_X.pkl",
        "train_script": "train_ssq_model.py",
        "fetch_script": "fetch_ssq_data.py"
    }
}

# ---------------- 日志信号发射器 ----------------
class LogEmitter(QObject):
    new_log = pyqtSignal(str)

# ---------------- 模型加载函数 ----------------
def load_pytorch_model(model_path, input_dim, hidden_dim, output_dim, output_seq_length, lottery_type):
    """
    加载 PyTorch 模型及缩放器
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # 加载红球模型
    red_model = LstmCRFModel(input_dim, hidden_dim, output_dim['red'], output_seq_length['red'], num_layers=1)
    red_model.load_state_dict(checkpoint['red_model'])
    red_model.eval()

    # 加载蓝球模型
    blue_model = LstmCRFModel(input_dim, hidden_dim, output_dim['blue'], output_seq_length['blue'], num_layers=1)
    blue_model.load_state_dict(checkpoint['blue_model'])
    blue_model.eval()

    # 加载缩放器
    scaler_X_path = os.path.join(os.path.dirname(model_path), name_path[lottery_type]['scaler_X_file'])
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"特征缩放器文件不存在：{scaler_X_path}")
    scaler_X = joblib.load(scaler_X_path)

    return red_model, blue_model, scaler_X

def load_resources_pytorch(lottery_type):
    if lottery_type not in name_path:
        raise ValueError(f"不支持的彩票类型：{lottery_type}，请检查输入。")

    # 根据彩票类型设置模型的输入维度和其他参数
    if lottery_type == "dlt":
        input_dim = 7  # 修改为训练时的一致输入维度
        hidden_dim = 128
        output_dim = {
            'red': 35,
            'blue': 12
        }
        output_seq_length = {
            'red': 5,
            'blue': 2
        }
    elif lottery_type == "ssq":
        input_dim = 7  # 修改为训练时的一致输入维度
        hidden_dim = 128
        output_dim = {
            'red': 33,
            'blue': 16
        }
        output_seq_length = {
            'red': 6,
            'blue': 1
        }

    # 动态生成模型路径
    model_path = os.path.join(name_path[lottery_type]['path'], name_path[lottery_type]['model_file'])

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    # 加载模型和缩放器
    red_model, blue_model, scaler_X = load_pytorch_model(
        model_path, input_dim, hidden_dim, output_dim, output_seq_length, lottery_type
    )

    return red_model, blue_model, scaler_X

# ---------------- 训练线程类 ----------------
class TrainModelThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, lottery_type):
        super().__init__()
        self.lottery_type = lottery_type

    def run(self):
        try:
            # 根据彩票类型选择训练脚本
            if self.lottery_type == "dlt":
                train_script = os.path.join(name_path[self.lottery_type]['path'], name_path[self.lottery_type]['train_script'])
            elif self.lottery_type == "ssq":
                train_script = os.path.join(name_path[self.lottery_type]['path'], name_path[self.lottery_type]['train_script'])
            else:
                self.log_signal.emit("未知的彩票类型！")
                return

            if not os.path.exists(train_script):
                self.log_signal.emit(f"训练脚本不存在：{train_script}")
                return

            # 使用当前目录下的 Python 解释器
            python_executable = sys.executable  # 自动使用当前运行 GUI 的 Python 解释器

            # 使用 subprocess 调用训练脚本
            process = subprocess.Popen(
                [python_executable, train_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8'
            )

            for line in process.stdout:
                self.log_signal.emit(line.strip())

            process.wait()

            if process.returncode == 0:
                self.log_signal.emit("模型训练完成。")
            else:
                self.log_signal.emit(f"模型训练失败，返回码：{process.returncode}")

        except Exception as e:
            self.log_signal.emit(f"训练过程中发生异常：{str(e)}")

        finally:
            self.finished_signal.emit()

# ---------------- 预测结果处理函数 ----------------
def process_predictions(red_predictions, blue_predictions, lottery_type):
    """
    处理预测结果，确保号码在有效范围内且为整数
    :param red_predictions: list, 红球预测的类别索引
    :param blue_predictions: list, 蓝球预测的类别索引
    :param lottery_type: 'ssq' 或 'dlt'
    :return: list, 预测的开奖号码
    """
    if lottery_type == "dlt":
        # 大乐透前区：1-35，后区：1-12
        front_numbers = [min(max(int(num) + 1, 1), 35) for num in red_predictions[:5]]
        back_numbers = [min(max(int(num) + 1, 1), 12) for num in blue_predictions[:2]]

        # 确保前区号码唯一
        front_numbers = list(set(front_numbers))
        while len(front_numbers) < 5:
            additional_num = np.random.randint(1, 36)
            if additional_num not in front_numbers:
                front_numbers.append(additional_num)
        front_numbers = sorted(front_numbers)[:5]

    elif lottery_type == "ssq":
        # 双色球红球：1-33，蓝球：1-16
        front_numbers = [min(max(int(num) + 1, 1), 33) for num in red_predictions[:6]]
        back_number = min(max(int(blue_predictions[0]) + 1, 1), 16)

        # 确保红球号码唯一
        front_numbers = list(set(front_numbers))
        while len(front_numbers) < 6:
            additional_num = np.random.randint(1, 34)
            if additional_num not in front_numbers:
                front_numbers.append(additional_num)
        front_numbers = sorted(front_numbers)[:6]

    else:
        raise ValueError("不支持的彩票类型！请选择 'ssq' 或 'dlt'。")

    if lottery_type == "dlt":
        return front_numbers + back_numbers
    elif lottery_type == "ssq":
        return front_numbers + [back_number]

# ---------------- 主窗口类 ----------------
class LotteryPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.log_emitter = LogEmitter()
        self.initUI()

        # 连接日志信号
        self.log_emitter.new_log.connect(self.update_log)

    def initUI(self):
        self.setWindowTitle("彩票预测应用程序")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # 彩票类型选择
        self.lottery_combo = QComboBox()
        self.lottery_combo.addItems([name_path[key]['name'] for key in name_path.keys()])
        layout.addWidget(QLabel("请选择彩票类型:"))
        layout.addWidget(self.lottery_combo)

        # 添加预测数量选择
        prediction_layout = QHBoxLayout()
        prediction_layout.addWidget(QLabel("请选择生成的预测数量:"))
        self.prediction_spin = QSpinBox()
        self.prediction_spin.setRange(1, 20)  # 设置预测数量范围
        self.prediction_spin.setValue(1)      # 默认值
        prediction_layout.addWidget(self.prediction_spin)
        layout.addLayout(prediction_layout)

        # 生成预测按钮
        self.predict_button = QPushButton("生成预测")
        self.predict_button.clicked.connect(self.generate_prediction)
        layout.addWidget(self.predict_button)

        # 添加训练模型按钮
        self.train_button = QPushButton("训练模型")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # 预测结果显示
        self.result_label = QLabel("预测结果将在此显示")
        layout.addWidget(self.result_label)

        # 日志显示框
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(QLabel("训练和预测日志:"))
        layout.addWidget(self.log_box)

        # 设置中心窗口
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def train_model(self):
        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']

        # 禁用训练按钮和彩票选择，防止重复训练
        self.train_button.setEnabled(False)
        self.lottery_combo.setEnabled(False)

        # 清空日志框
        self.log_box.clear()

        # 启动训练线程
        self.train_thread = TrainModelThread(lottery_type)
        self.train_thread.log_signal.connect(self.update_log)
        self.train_thread.finished_signal.connect(self.on_train_finished)
        self.train_thread.start()

    def on_train_finished(self):
        # 启用训练按钮和彩票选择
        self.train_button.setEnabled(True)
        self.lottery_combo.setEnabled(True)
        self.update_log("训练线程已结束。")

    def generate_prediction(self):
        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']
        num_predictions = self.prediction_spin.value()

        try:
            # 加载模型和缩放器
            red_model, blue_model, scaler_X = load_resources_pytorch(lottery_type)
            self.log_emitter.new_log.emit(f"已加载 PyTorch 模型和缩放器 for {lottery_name}")

            # 生成预测
            result_text = f"预测的{num_predictions}个{lottery_name}号码：\n"

            for i in range(num_predictions):
                with torch.no_grad():
                    # 使用 LSTM 和 FC 层生成红球和蓝球的预测结果
                    red_logits = red_model.lstm(torch.tensor(np.zeros((1, 7)), dtype=torch.float32))  # 这里的输入可以根据需要修改
                    red_logits = red_model.fc(red_logits[0])
                    red_logits = red_logits.view(-1, red_model.output_seq_length * red_model.output_dim)
                    red_logits = red_logits.view(-1, red_model.output_seq_length, red_model.output_dim)
                    red_predictions = red_model.crf.decode(red_logits)

                    blue_logits = blue_model.lstm(torch.tensor(np.zeros((1, 7)), dtype=torch.float32))  # 同样是修改输入
                    blue_logits = blue_model.fc(blue_logits[0])
                    blue_logits = blue_logits.view(-1, blue_model.output_seq_length * blue_model.output_dim)
                    blue_logits = blue_logits.view(-1, blue_model.output_seq_length, blue_model.output_dim)
                    blue_predictions = blue_model.crf.decode(blue_logits)

                # 处理预测结果
                red_predicted = red_predictions[0]  # 假设 batch_size=1
                red_numbers = [num for num in red_predicted]  # 类别索引

                blue_predicted = blue_predictions[0]
                blue_number = blue_predicted[0]  # 类别索引

                # 转换为原始标签（加1）
                red_numbers = [num + 1 for num in red_numbers]
                blue_number = blue_number + 1

                # 处理预测结果
                predicted_numbers = process_predictions(
                    red_numbers, [blue_number],
                    lottery_type
                )
                self.log_emitter.new_log.emit(f"第 {i + 1} 个预测的最终号码: {predicted_numbers}")

                # 显示结果
                if lottery_type == "dlt":
                    front = predicted_numbers[:5]
                    back = predicted_numbers[5:]
                    result_text += f"预测 {i + 1}:\n前区：{front}\n后区：{back}\n"
                elif lottery_type == "ssq":
                    front = predicted_numbers[:6]
                    back = predicted_numbers[6]
                    result_text += f"预测 {i + 1}:\n红球：{front}\n蓝球：{back}\n"

            self.result_label.setText(result_text)

        except Exception as e:
            self.log_emitter.new_log.emit(f"预测失败: {str(e)}")
            self.result_label.setText("预测失败，请检查日志。")

    def update_log(self, message):
        self.log_box.append(message)
        self.log_box.ensureCursorVisible()

# ---------------- 主程序入口 ----------------
def main():
    app = QApplication(sys.argv)
    main_window = LotteryPredictorApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
