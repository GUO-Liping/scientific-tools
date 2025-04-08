import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox,
    QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QFormLayout
)
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class MultiImpactLoad:
    def generate_impact_series(self, load_type, params, t_array, num_impacts, interval):
        force = np.zeros_like(t_array)
        for i in range(num_impacts):
            t_shift = i * interval
            for j, t in enumerate(t_array):
                t_local = t - t_shift
                if t_local < 0:
                    continue
                if load_type == "矩形冲击":
                    A, D = params
                    if t_local <= D:
                        force[j] += A
                elif load_type == "三角冲击":
                    A, D = params
                    if 0 <= t_local <= D:
                        force[j] += A * (1 - abs((2 * t_local / D) - 1))
                elif load_type == "正弦冲击":
                    A, f, D = params
                    if 0 <= t_local <= D:
                        force[j] += A * np.sin(2 * np.pi * f * t_local)
        return force


class MultiDOFResponse:
    def response(self, force, t, m=1.0, c=0.05, k=10.0):
        dt = t[1] - t[0]
        n = len(t)
        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)

        a[0] = (force[0] - c * v[0] - k * u[0]) / m
        for i in range(1, n):
            a[i] = (force[i] - c * v[i - 1] - k * u[i - 1]) / m
            v[i] = v[i - 1] + a[i] * dt
            u[i] = u[i - 1] + v[i] * dt

        return u, v, a


class ImpactResponseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多次连续冲击结构响应分析器")
        self.setGeometry(100, 100, 1000, 700)

        self.load_model = MultiImpactLoad()
        self.response_model = MultiDOFResponse()
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout()
        controls = QVBoxLayout()
        form = QFormLayout()

        self.load_type = QComboBox()
        self.load_type.addItems(["矩形冲击", "三角冲击", "正弦冲击"])
        form.addRow("冲击类型:", self.load_type)

        self.inputs = {
            "amplitude": QLineEdit("10"),
            "duration": QLineEdit("0.2"),
            "frequency": QLineEdit("5.0"),
            "interval": QLineEdit("1.0"),
            "num_impacts": QLineEdit("3"),
            "mass": QLineEdit("1.0"),
            "damping": QLineEdit("0.05"),
            "stiffness": QLineEdit("10.0"),
            "t_max": QLineEdit("5.0"),
            "dt": QLineEdit("0.001")
        }

        for key, widget in self.inputs.items():
            form.addRow(f"{key.replace('_', ' ').capitalize()}:", widget)

        controls.addLayout(form)

        self.calc_button = QPushButton("计算并绘图")
        self.calc_button.clicked.connect(self.calculate_and_plot)
        controls.addWidget(self.calc_button)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        controls.addWidget(QLabel("结果输出:"))
        controls.addWidget(self.result_box)

        layout.addLayout(controls, 3)

        self.fig, self.axs = plt.subplots(4, 1, figsize=(6, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 6)

        main_widget.setLayout(layout)

    def calculate_and_plot(self):
        try:
            load_type = self.load_type.currentText()
            A = float(self.inputs["amplitude"].text())
            D = float(self.inputs["duration"].text())
            f = float(self.inputs["frequency"].text())
            interval = float(self.inputs["interval"].text())
            num = int(self.inputs["num_impacts"].text())
            m = float(self.inputs["mass"].text())
            c = float(self.inputs["damping"].text())
            k = float(self.inputs["stiffness"].text())
            t_max = float(self.inputs["t_max"].text())
            dt = float(self.inputs["dt"].text())

            t = np.arange(0, t_max, dt)

            if load_type == "正弦冲击":
                params = (A, f, D)
            else:
                params = (A, D)

            force = self.load_model.generate_impact_series(load_type, params, t, num, interval)
            u, v, a = self.response_model.response(force, t, m, c, k)

            self.axs[0].clear()
            self.axs[0].plot(t, force, label="冲击荷载")
            self.axs[0].set_title("冲击荷载")
            self.axs[0].grid(True)

            self.axs[1].clear()
            self.axs[1].plot(t, u, label="位移", color='orange')
            self.axs[1].set_title("结构位移")
            self.axs[1].grid(True)

            self.axs[2].clear()
            self.axs[2].plot(t, v, label="速度", color='green')
            self.axs[2].set_title("结构速度")
            self.axs[2].grid(True)

            self.axs[3].clear()
            self.axs[3].plot(t, a, label="加速度", color='red')
            self.axs[3].set_title("结构加速度")
            self.axs[3].set_xlabel("时间 (s)")
            self.axs[3].grid(True)

            self.fig.tight_layout()
            self.canvas.draw()

            self.result_box.setText("计算完成，已显示各响应图像。")

        except Exception as e:
            self.result_box.setText(f"错误: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImpactResponseApp()
    window.show()
    sys.exit(app.exec_())
