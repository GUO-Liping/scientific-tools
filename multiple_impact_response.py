# -*- coding: UTF-8 -*-
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QComboBox, QFormLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMainWindow, QSplitter)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def multi_pulse_load(t, pulse_type, amplitude, duration, interval, n_pulses):
    load = 0
    for i in range(n_pulses):
        start = i * interval
        end = start + duration
        if start <= t <= end:
            t_rel = t - start
            if pulse_type == "短时间矩形脉冲":
                load += amplitude
            elif pulse_type == "三角脉冲":
                if t_rel < duration / 2:
                    load += 2 * amplitude * t_rel / duration
                else:
                    load += 2 * amplitude * (duration - t_rel) / duration
            elif pulse_type == "正弦脉冲":
                load += amplitude * np.sin(np.pi * t_rel / duration)
    return load

def simulate_response(m, c, k, pulse_type, amplitude, duration, interval, n_pulses, t_end=10.0, dt=0.001):
    N = int(t_end / dt)
    t = np.linspace(0, t_end, N)
    x = np.zeros(N)
    v = np.zeros(N)

    def deriv(xi, vi, time):
        F = multi_pulse_load(time, pulse_type, amplitude, duration, interval, n_pulses)
        return (F - c * vi - k * xi) / m

    for i in range(N - 1):
        ti = t[i]
        xi = x[i]
        vi = v[i]

        k1_v = deriv(xi, vi, ti)
        k1_x = vi

        k2_v = deriv(xi + dt * k1_x / 2, vi + dt * k1_v / 2, ti + dt / 2)
        k2_x = vi + dt * k1_v / 2

        k3_v = deriv(xi + dt * k2_x / 2, vi + dt * k2_v / 2, ti + dt / 2)
        k3_x = vi + dt * k2_v / 2

        k4_v = deriv(xi + dt * k3_x, vi + dt * k3_v, ti + dt)
        k4_x = vi + dt * k3_v

        x[i + 1] = xi + dt * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        v[i + 1] = vi + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return t, x

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多次冲击下多自由度系统动力分析程序")
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        form_layout = QFormLayout()
        self.mass_input = QLineEdit("1.0")
        self.damping_input = QLineEdit("0.2")
        self.stiffness_input = QLineEdit("10.0")
        self.amplitude_input = QLineEdit("5.0")
        self.duration_input = QLineEdit("0.1")
        self.n_pulses_input = QLineEdit("3")
        self.interval_input = QLineEdit("1.0")

        form_layout.addRow("质量 m (kg):", self.mass_input)
        form_layout.addRow("阻尼 c (N·s/m):", self.damping_input)
        form_layout.addRow("刚度 k (N/m):", self.stiffness_input)
        form_layout.addRow("脉冲幅值 (N):", self.amplitude_input)
        form_layout.addRow("持续时间 (s):", self.duration_input)
        form_layout.addRow("脉冲个数:", self.n_pulses_input)
        form_layout.addRow("脉冲间隔 (s):", self.interval_input)

        self.load_type_combo = QComboBox()
        self.load_type_combo.addItems(["短时间矩形脉冲", "三角脉冲", "正弦脉冲"])
        form_layout.addRow("脉冲荷载形式:", self.load_type_combo)

        self.run_button = QPushButton("运行分析")
        self.run_button.clicked.connect(self.run_analysis)

        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.run_button)

        splitter = QSplitter()

        self.response_fig = Figure(figsize=(5, 4), dpi=100)
        self.response_canvas = FigureCanvas(self.response_fig)

        self.load_fig = Figure(figsize=(5, 4), dpi=100)
        self.load_canvas = FigureCanvas(self.load_fig)

        splitter.addWidget(self.response_canvas)
        splitter.addWidget(self.load_canvas)
        splitter.setSizes([400, 400])

        main_layout.addWidget(splitter)

    def run_analysis(self):
        try:
            m = float(self.mass_input.text())
            c = float(self.damping_input.text())
            k = float(self.stiffness_input.text())
            amplitude = float(self.amplitude_input.text())
            duration = float(self.duration_input.text())
            n_pulses = int(self.n_pulses_input.text())
            interval = float(self.interval_input.text())
            pulse_type = self.load_type_combo.currentText()
        except ValueError:
            print("\u8f93\u5165\u53c2\u6570\u6709\u8bef")
            return

        t_end = n_pulses * interval + 5.0
        t, x = simulate_response(m, c, k, pulse_type, amplitude, duration, interval, n_pulses, t_end)
        F = np.array([multi_pulse_load(ti, pulse_type, amplitude, duration, interval, n_pulses) for ti in t])

        self.response_fig.clear()
        ax1 = self.response_fig.add_subplot(111)
        ax1.plot(t, x, label="位移响应")
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel("位移 (m)")
        ax1.set_title("动力响应")
        ax1.legend()
        self.response_canvas.draw()

        self.load_fig.clear()
        ax2 = self.load_fig.add_subplot(111)
        ax2.plot(t, F, label="冲击荷载")
        ax2.set_xlabel("时间 (s)")
        ax2.set_ylabel("荷载 (N)")
        ax2.set_title("冲击荷载曲线")
        ax2.legend()
        self.load_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
