# lottery_predictor_app.py

# -*- coding:utf-8 -*-
"""
Author: BigCat
"""

import sys
import os
from idlelib.iomenu import encoding

import pandas as pd
import numpy as np
import io
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton,
    QLabel, QComboBox, QWidget, QTextEdit, QSpinBox, QDoubleSpinBox, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
import warnings
import subprocess
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ---------------- 配置 ----------------
name_path = {
    "dlt": {
        "name": "大乐透",
        "path": "./dlt/",
        "model_file": "dlt_model.h5",
        "scaler_X_file": "dlt_scaler_X.pkl",
        "scaler_y_file": "dlt_scaler_y.pkl",
        "train_script": "train_dlt_model.py",
        "fetch_script": "fetch_dlt_data.py"
    },
    "ssq": {
        "name": "双色球",
        "path": "./ssq/",
        "model_file": "ssq_model.h5",
        "scaler_X_file": "ssq_scaler_X.pkl",
        "scaler_y_file": "ssq_scaler_y.pkl",
        "train_script": "train_ssq_model.py",
        "fetch_script": "fetch_ssq_data.py"
    }
}

# ---------------- 日志信号发射器 ----------------
class LogEmitter(QObject):
    new_log = pyqtSignal(str)

# ---------------- 预测函数 ----------------

def load_resources(lottery_type):
    """
    加载指定彩票类型的模型和缩放器
    :param lottery_type: 'ssq' 或 'dlt'
    :return: model, scaler_X, scaler_y
    """
    if lottery_type not in name_path:
        raise ValueError("不支持的彩票类型！请选择 'ssq' 或 'dlt'。")

    path = name_path[lottery_type]['path']
    model_path = os.path.join(path, name_path[lottery_type]['model_file'])
    scaler_X_path = os.path.join(path, name_path[lottery_type]['scaler_X_file'])
    scaler_y_path = os.path.join(path, name_path[lottery_type]['scaler_y_file'])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"特征缩放器文件不存在：{scaler_X_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"目标缩放器文件不存在：{scaler_y_path}")

    # 加载模型时不编译
    model = load_model(model_path, compile=False)

    # 手动编译模型
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_X, scaler_y


def add_noise_to_features(features, noise_level=0.01):
    """
    在特征中添加随机噪声
    :param features: 输入特征 (numpy array)
    :param noise_level: 噪声强度 (0-1)
    :return: 添加噪声后的特征 (numpy array)
    """
    noise = np.random.normal(0, noise_level, features.shape)
    return features + noise

def inverse_scale_predictions(predictions, scaler_y):
    """
    将预测结果逆缩放回原始范围
    :param predictions: numpy array, 预测的缩放值
    :param scaler_y: 缩放器对象
    :return: numpy array, 逆缩放后的预测值
    """
    return scaler_y.inverse_transform(predictions)

def process_predictions(predictions, lottery_type):
    """
    处理逆缩放后的预测结果，确保号码在有效范围内且为整数
    :param predictions: numpy array, 逆缩放后的预测值
    :param lottery_type: 'ssq' 或 'dlt'
    :return: list, 预测的开奖号码
    """
    if lottery_type == "dlt":
        # 大乐透前区：1-35，后区：1-12
        front_numbers = [int(round(num)) for num in predictions[0][:5]]
        back_numbers = [int(round(num)) for num in predictions[0][5:]]

        # 确保号码在范围内
        front_numbers = [min(max(num, 1), 35) for num in front_numbers]
        back_numbers = [min(max(num, 1), 12) for num in back_numbers]

        # 确保前区号码唯一
        front_numbers = list(set(front_numbers))
        while len(front_numbers) < 5:
            additional_num = np.random.randint(1, 36)
            if additional_num not in front_numbers:
                front_numbers.append(additional_num)
        front_numbers = sorted(front_numbers)[:5]

    elif lottery_type == "ssq":
        # 双色球红球：1-33，蓝球：1-16
        front_numbers = [int(round(num)) for num in predictions[0][:6]]
        back_number = int(round(predictions[0][6]))

        # 确保号码在范围内
        front_numbers = [min(max(num, 1), 33) for num in front_numbers]
        back_number = min(max(back_number, 1), 16)

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

# ---------------- 训练线程类 ----------------
import os
import subprocess

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
                train_script = os.path.join(os.getcwd(), "train_dlt_model.py")
            elif self.lottery_type == "ssq":
                train_script = os.path.join(os.getcwd(), "train_ssq_model.py")
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
                encoding = 'utf-8'
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

        # 输入特征部分
        self.input_frame = QWidget()
        self.input_layout = QVBoxLayout()
        self.input_frame.setLayout(self.input_layout)
        layout.addWidget(QLabel("请输入特征值:"))
        layout.addWidget(self.input_frame)

        # 动态生成输入字段
        self.entries = {}
        self.update_input_fields()

        # 彩票类型改变时更新输入字段
        self.lottery_combo.currentIndexChanged.connect(self.update_input_fields)

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

    def update_input_fields(self):
        # 清空现有的输入字段
        while self.input_layout.count():
            child = self.input_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.entries = {}

        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key

        if lottery_type == "dlt":
            fields = [
                ("前区和值", "float"),
                ("前区奇数个数", "int"),
                ("后区和值", "float")
                # 如果训练时使用了更多特征，请在此添加
            ]
        elif lottery_type == "ssq":
            fields = [
                ("红球和值", "float"),
                ("红球奇数个数", "int")
                # 如果训练时使用了更多特征，请在此添加
            ]
        else:
            fields = []

        for field, ftype in fields:
            frame = QHBoxLayout()
            label = QLabel(field + ":")
            if ftype == "float":
                entry = QDoubleSpinBox()
                entry.setDecimals(2)
                entry.setSingleStep(0.1)
                entry.setRange(0, 1000)  # 根据实际情况调整范围
                entry.setValue(0.0)
            elif ftype == "int":
                entry = QSpinBox()
                entry.setRange(0, 1000)  # 根据实际情况调整范围
                entry.setValue(0)
            else:
                entry = QSpinBox()

            frame.addWidget(label)
            frame.addWidget(entry)
            self.input_layout.addLayout(frame)
            self.entries[field] = (entry, ftype)

    def get_input_features(self, lottery_type):
        """
        获取用户输入的特征
        :param lottery_type: 'ssq' 或 'dlt'
        :return: DataFrame 包含输入特征
        """
        inputs = {}
        for field, (entry, ftype) in self.entries.items():
            if ftype == "float":
                value = float(entry.value())
            elif ftype == "int":
                value = int(entry.value())
            else:
                value = int(entry.value())
            inputs[field] = value
        X = pd.DataFrame([inputs])
        return X

    def generate_prediction(self):
        selected_index = self.lottery_combo.currentIndex()
        selected_key = list(name_path.keys())[selected_index]
        lottery_type = selected_key
        lottery_name = name_path[selected_key]['name']
        num_predictions = self.prediction_spin.value()

        # 获取用户输入的特征
        X_input = self.get_input_features(lottery_type)

        try:
            # 加载模型和缩放器
            model, scaler_X, scaler_y = load_resources(lottery_type)
            self.log_emitter.new_log.emit(f"已加载模型和缩放器 for {lottery_name}")

            # 特征缩放
            X_scaled = scaler_X.transform(X_input)
            self.log_emitter.new_log.emit("已完成特征缩放。")

            result_text = f"预测的{num_predictions}个{lottery_name}号码：\n"

            for i in range(num_predictions):
                # 为输入特征添加随机噪声
                X_scaled_with_noise = add_noise_to_features(X_scaled, noise_level=0.05)

                # 预测
                predictions_scaled = model.predict(X_scaled_with_noise)
                self.log_emitter.new_log.emit(f"第 {i+1} 个预测已完成。")

                # 逆缩放
                predictions = inverse_scale_predictions(predictions_scaled, scaler_y)
                self.log_emitter.new_log.emit("已完成逆缩放预测结果。")

                # 处理预测结果
                predicted_numbers = process_predictions(predictions, lottery_type)
                self.log_emitter.new_log.emit("已完成预测结果处理。")

                # 显示结果
                if lottery_type == "dlt":
                    front = predicted_numbers[:5]
                    back = predicted_numbers[5:]
                    result_text += f"预测 {i+1}:\n前区：{front}\n后区：{back}\n"
                elif lottery_type == "ssq":
                    front = predicted_numbers[:6]
                    back = predicted_numbers[6]
                    result_text += f"预测 {i+1}:\n红球：{front}\n蓝球：{back}\n"

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
