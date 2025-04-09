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

class PulseLoadCalculator:
    def rectangular_pulse(self, amplitude, duration, time):
        return amplitude if 0 <= time <= duration else 0

    def triangular_pulse(self, amplitude, duration, time):
        if 0 <= time <= duration:
            return amplitude * (1 - abs((2 * time / duration) - 1))
        return 0

    def sinusoidal_pulse(self, amplitude, frequency, time):
        return amplitude * np.sin(2 * np.pi * frequency * time)

    def exponential_pulse(self, amplitude, decay, time):
        return amplitude * np.exp(-decay * time) if time >= 0 else 0

    def custom_pulse(self, expression, time):
        try:
            return eval(expression, {"t": time, "np": np})
        except Exception as e:
            return f"Error: {e}"

    def get_pulse_series(self, load_type, params, t_array):
        series = []
        for t in t_array:
            if load_type == "矩形脉冲":
                series.append(self.rectangular_pulse(*params, t))
            elif load_type == "三角脉冲":
                series.append(self.triangular_pulse(*params, t))
            elif load_type == "正弦脉冲":
                series.append(self.sinusoidal_pulse(*params, t))
            elif load_type == "指数衰减脉冲":
                series.append(self.exponential_pulse(*params, t))
            elif load_type == "自定义函数脉冲":
                val = self.custom_pulse(params[0], t)
                series.append(val if isinstance(val, (int, float)) else 0)
        return np.array(series)


class StructuralResponseCalculator:
    def single_dof_response(self, force_series, t_array, mass=1.0, damping=0.05, stiffness=10.0):
        dt = t_array[1] - t_array[0]
        n = len(t_array)
        u = np.zeros(n)  # displacement
        v = np.zeros(n)  # velocity
        a = np.zeros(n)  # acceleration

        a[0] = (force_series[0] - damping * v[0] - stiffness * u[0]) / mass

        for i in range(1, n):
            a[i] = (force_series[i] - damping * v[i-1] - stiffness * u[i-1]) / mass
            v[i] = v[i-1] + a[i] * dt
            u[i] = u[i-1] + v[i] * dt

        return u


class SteelStructureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Structural Impact Toolbox V1.0")
        self.setGeometry(100, 100, 900, 600)

        self.calculator = PulseLoadCalculator()
        self.response_calc = StructuralResponseCalculator()

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.load_type_combo = QComboBox()
        self.load_type_combo.addItems([
            "矩形脉冲", "三角脉冲", "正弦脉冲", "指数衰减脉冲", "自定义函数脉冲"
        ])
        self.load_type_combo.currentIndexChanged.connect(self.update_input_fields)
        form_layout.addRow("选择脉冲类型:", self.load_type_combo)

        self.input_fields = {
            "amplitude": QLineEdit(),
            "duration": QLineEdit(),
            "frequency": QLineEdit(),
            "decay": QLineEdit(),
            "expression": QLineEdit(),
            "mass": QLineEdit("1.0"),
            "damping": QLineEdit("0.05"),
            "stiffness": QLineEdit("10.0"),
            "t_max": QLineEdit("5.0"),
            "dt": QLineEdit("0.01")
        }

        form_layout.addRow("振幅 (amplitude):", self.input_fields["amplitude"])
        form_layout.addRow("持续时间/周期 (duration):", self.input_fields["duration"])
        form_layout.addRow("频率 (frequency):", self.input_fields["frequency"])
        form_layout.addRow("衰减系数 (decay):", self.input_fields["decay"])
        form_layout.addRow("自定义表达式 f(t):", self.input_fields["expression"])
        form_layout.addRow("质量 mass:", self.input_fields["mass"])
        form_layout.addRow("阻尼 damping:", self.input_fields["damping"])
        form_layout.addRow("刚度 stiffness:", self.input_fields["stiffness"])
        form_layout.addRow("分析最大时间 t_max (s):", self.input_fields["t_max"])
        form_layout.addRow("时间步长 dt (s):", self.input_fields["dt"])

        self.update_input_fields()

        controls_layout.addLayout(form_layout)

        self.calc_button = QPushButton("计算并绘图")
        self.calc_button.clicked.connect(self.calculate_and_plot)
        controls_layout.addWidget(self.calc_button)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        controls_layout.addWidget(QLabel("结果输出:"))
        controls_layout.addWidget(self.result_output)

        layout.addLayout(controls_layout, 3)

        # Plot canvas
        self.figure, self.ax = plt.subplots(2, 1, figsize=(5, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 5)

        central_widget.setLayout(layout)

    def update_input_fields(self):
        load_type = self.load_type_combo.currentText()

        for key, field in self.input_fields.items():
            field.hide()

        if load_type == "矩形脉冲" or load_type == "三角脉冲":
            self.input_fields["amplitude"].show()
            self.input_fields["duration"].show()
        elif load_type == "正弦脉冲":
            self.input_fields["amplitude"].show()
            self.input_fields["frequency"].show()
        elif load_type == "指数衰减脉冲":
            self.input_fields["amplitude"].show()
            self.input_fields["decay"].show()
        elif load_type == "自定义函数脉冲":
            self.input_fields["expression"].show()

        self.input_fields["mass"].show()
        self.input_fields["damping"].show()
        self.input_fields["stiffness"].show()
        self.input_fields["t_max"].show()
        self.input_fields["dt"].show()

    def calculate_and_plot(self):
        load_type = self.load_type_combo.currentText()
        t_max = float(self.input_fields["t_max"].text())
        dt = float(self.input_fields["dt"].text())
        t_array = np.arange(0, t_max, dt)

        try:
            if load_type == "矩形脉冲":
                A = float(self.input_fields["amplitude"].text())
                D = float(self.input_fields["duration"].text())
                params = (A, D)
            elif load_type == "三角脉冲":
                A = float(self.input_fields["amplitude"].text())
                D = float(self.input_fields["duration"].text())
                params = (A, D)
            elif load_type == "正弦脉冲":
                A = float(self.input_fields["amplitude"].text())
                f = float(self.input_fields["frequency"].text())
                params = (A, f)
            elif load_type == "指数衰减脉冲":
                A = float(self.input_fields["amplitude"].text())
                d = float(self.input_fields["decay"].text())
                params = (A, d)
            elif load_type == "自定义函数脉冲":
                expr = self.input_fields["expression"].text()
                params = (expr,)
            else:
                raise ValueError("不支持的脉冲类型")

            force_series = self.calculator.get_pulse_series(load_type, params, t_array)

            mass = float(self.input_fields["mass"].text())
            damping = float(self.input_fields["damping"].text())
            stiffness = float(self.input_fields["stiffness"].text())

            disp_series = self.response_calc.single_dof_response(force_series, t_array, mass, damping, stiffness)

            # 绘图
            self.ax[0].clear()
            self.ax[0].plot(t_array, force_series, label="荷载 f(t)")
            self.ax[0].set_title("荷载时程")
            self.ax[0].set_ylabel("荷载")
            self.ax[0].grid(True)

            self.ax[1].clear()
            self.ax[1].plot(t_array, disp_series, label="结构响应 u(t)", color='orange')
            self.ax[1].set_title("结构响应")
            self.ax[1].set_ylabel("位移")
            self.ax[1].set_xlabel("时间 (s)")
            self.ax[1].grid(True)

            self.figure.tight_layout()
            self.canvas.draw()

            self.result_output.setText("计算完成，已绘图显示荷载与结构响应。")

        except Exception as e:
            self.result_output.setText(f"计算或绘图出错: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SteelStructureApp()
    window.show()
    sys.exit(app.exec_())
