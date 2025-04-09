
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox,
    QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QFormLayout,
    QFileDialog, QMessageBox, QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt
import matplotlib
import csv
import re
from sympy import symbols, lambdify, sympify

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class MultiImpactLoad:
    def __init__(self):
        self.custom_expr_func = None

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
                elif load_type == "高斯冲击":
                    A, D = params
                    force[j] += A * np.exp(-((t_local - D / 2) ** 2) / (2 * (D / 6) ** 2))
                elif load_type == "半正弦冲击":
                    A, D = params
                    if 0 <= t_local <= D:
                        force[j] += A * np.sin((np.pi * t_local) / D)
                elif load_type == "自定义冲击":
                    force[j] += self.evaluate_expression(t_local)
        return force

    def set_custom_expression(self, expr):
        try:
            t_sym = symbols('t')
            expr_sym = sympify(expr)
            self.custom_expr_func = lambdify(t_sym, expr_sym, 'numpy')
        except Exception as e:
            raise ValueError(f"自定义表达式无效: {e}")

    def evaluate_expression(self, t):
        if self.custom_expr_func is None:
            raise ValueError("自定义表达式未设置")
        return self.custom_expr_func(t)


class MultiDOFResponse:
    def __init__(self, num_dofs):
        self.num_dofs = num_dofs

    def response(self, force, t, m, c, k):
        dt = t[1] - t[0]
        n = len(t)
        u = np.zeros((self.num_dofs, n))
        v = np.zeros((self.num_dofs, n))
        a = np.zeros((self.num_dofs, n))

        M = np.diag(m)
        C = np.diag(c)
        K = np.diag(k)

        for i in range(self.num_dofs):
            a[i, 0] = (force[0] - C[i, i] * v[i, 0] - K[i, i] * u[i, 0]) / M[i, i]

        for i in range(1, n):
            for j in range(self.num_dofs):
                a[j, i] = (force[i] - C[j, j] * v[j, i - 1] - K[j, j] * u[j, i - 1]) / M[j, j]
                v[j, i] = v[j, i - 1] + a[j, i] * dt
                u[j, i] = u[j, i - 1] + v[j, i] * dt

        return u, v, a


class ImpactResponseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Structural Impact Toolbox V2.0")
        self.setGeometry(100, 100, 1000, 700)

        self.load_model = MultiImpactLoad()
        self.num_dofs = 1
        self.response_model = MultiDOFResponse(self.num_dofs)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout()
        controls = QVBoxLayout()
        form = QFormLayout()

        self.load_type = QComboBox()
        self.load_type.addItems(["矩形冲击", "三角冲击", "正弦冲击", "高斯冲击", "半正弦冲击", "自定义冲击"])
        self.load_type.currentTextChanged.connect(self.update_custom_expression_state)
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
            "dt": QLineEdit("0.001"),
            "num_dofs": QLineEdit("1"),
            "custom_expression": QLineEdit("10 * exp(-t**2 / 0.1)")
        }

        for key, widget in self.inputs.items():
            form.addRow(f"{key.replace('_', ' ').capitalize()}:", widget)

        controls.addLayout(form)

        self.calc_button = QPushButton("计算并绘图")
        self.calc_button.clicked.connect(self.calculate_and_plot)
        controls.addWidget(self.calc_button)

        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self.export_results)
        controls.addWidget(self.export_button)

        self.response_spectrum_checkbox = QCheckBox("计算响应谱")
        controls.addWidget(self.response_spectrum_checkbox)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        controls.addWidget(QLabel("结果输出:"))
        controls.addWidget(self.result_box)

        layout.addLayout(controls, 3)

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_plot_tab(), "响应曲线")
        self.tab_widget.addTab(self.create_spectrum_tab(), "响应谱")
        layout.addWidget(self.tab_widget, 6)

        main_widget.setLayout(layout)

        self.update_custom_expression_state()

    def update_custom_expression_state(self):
        if self.load_type.currentText() == "自定义冲击":
            self.inputs["custom_expression"].setEnabled(True)
            try:
                self.load_model.set_custom_expression(self.inputs["custom_expression"].text())
            except Exception as e:
                QMessageBox.warning(self, "自定义表达式错误", str(e))
        else:
            self.inputs["custom_expression"].setEnabled(False)

    def create_plot_tab(self):
        plot_widget = QWidget()
        layout = QVBoxLayout()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(6, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        plot_widget.setLayout(layout)
        return plot_widget

    def create_spectrum_tab(self):
        spectrum_widget = QWidget()
        layout = QVBoxLayout()
        self.spectrum_fig, self.spectrum_ax = plt.subplots()
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        layout.addWidget(self.spectrum_canvas)
        spectrum_widget.setLayout(layout)
        return spectrum_widget

    def calculate_and_plot(self):
        try:
            load_type = self.load_type.currentText()
            A = float(self.inputs["amplitude"].text())
            D = float(self.inputs["duration"].text())
            f = float(self.inputs["frequency"].text())
            interval = float(self.inputs["interval"].text())
            num = int(self.inputs["num_impacts"].text())
            m = [float(x) for x in self.inputs["mass"].text().split(",")]
            c = [float(x) for x in self.inputs["damping"].text().split(",")]
            k = [float(x) for x in self.inputs["stiffness"].text().split(",")]
            t_max = float(self.inputs["t_max"].text())

            dt = float(self.inputs["dt"].text())
            self.num_dofs = int(self.inputs["num_dofs"].text())

            if len(m) != self.num_dofs or len(c) != self.num_dofs or len(k) != self.num_dofs:
                raise ValueError("自由度参数数量不匹配")

            t = np.arange(0, t_max, dt)

            if load_type == "正弦冲击":
                params = (A, f, D)
            elif load_type == "自定义冲击":
                params = self.inputs["custom_expression"].text()
                self.load_model.set_custom_expression(params)
            else:
                params = (A, D)

            force = self.load_model.generate_impact_series(load_type, params, t, num, interval)
            self.response_model = MultiDOFResponse(self.num_dofs)
            u, v, a = self.response_model.response(force, t, m, c, k)

            # 清除之前的绘图
            for ax in self.axs:
                ax.clear()

            # 绘制冲击荷载
            self.axs[0].plot(t, force, label="冲击荷载")
            self.axs[0].set_title("冲击荷载")
            self.axs[0].set_xlabel("时间 (s)")
            self.axs[0].set_ylabel("荷载 (N)")
            self.axs[0].grid(True)

            # 绘制响应曲线
            for i in range(self.num_dofs):
                self.axs[1].plot(t, u[i], label=f"位移 DOF {i + 1}")
                self.axs[2].plot(t, v[i], label=f"速度 DOF {i + 1}")
                self.axs[3].plot(t, a[i], label=f"加速度 DOF {i + 1}")

            self.axs[1].set_title("位移响应")
            self.axs[1].set_xlabel("时间 (s)")
            self.axs[1].set_ylabel("位移 (m)")
            self.axs[1].legend()
            self.axs[1].grid(True)

            self.axs[2].set_title("速度响应")
            self.axs[2].set_xlabel("时间 (s)")
            self.axs[2].set_ylabel("速度 (m/s)")
            self.axs[2].legend()
            self.axs[2].grid(True)

            self.axs[3].set_title("加速度响应")
            self.axs[3].set_xlabel("时间 (s)")
            self.axs[3].set_ylabel("加速度 (m/s²)")
            self.axs[3].legend()
            self.axs[3].grid(True)

            # 刷新绘图
            self.canvas.draw()

            # 如果勾选了响应谱分析
            if self.response_spectrum_checkbox.isChecked():
                self.calculate_response_spectrum(a, t)

            self.result_box.setText("计算完成，已显示各响应图像。")

        except Exception as e:
            self.result_box.setText(f"错误: {str(e)}")

    def calculate_response_spectrum(self, acceleration, t):
        try:
            dt = t[1] - t[0]
            freq = np.fft.fftfreq(len(t), d=dt)
            spectrum = np.fft.fft(acceleration, axis=1)

            # 只取正频率部分
            half_n = int(len(freq) / 2)
            freq = freq[:half_n]
            spectrum = np.abs(spectrum[:, :half_n])

            # 清除之前的绘图
            self.spectrum_ax.clear()

            # 绘制响应谱
            for i in range(self.num_dofs):
                self.spectrum_ax.plot(freq, spectrum[i], label=f"DOF {i + 1}")
            self.spectrum_ax.set_title("加速度响应谱")
            self.spectrum_ax.set_xlabel("频率 (Hz)")
            self.spectrum_ax.set_ylabel("幅值")
            self.spectrum_ax.legend()
            self.spectrum_ax.grid(True)

            # 刷新绘图
            self.spectrum_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "响应谱计算失败", f"响应谱计算过程中发生错误: {str(e)}")

    def export_results(self):
        try:
            file_name, _ = QFileDialog.getSaveFileName(self, "导出结果", "", "CSV 文件 (*.csv)")
            if not file_name:
                return

            load_type = self.load_type.currentText()
            A = float(self.inputs["amplitude"].text())
            D = float(self.inputs["duration"].text())
            f = float(self.inputs["frequency"].text())
            interval = float(self.inputs["interval"].text())
            num = int(self.inputs["num_impacts"].text())
            m = [float(x) for x in self.inputs["mass"].text().split(",")]
            c = [float(x) for x in self.inputs["damping"].text().split(",")]
            k = [float(x) for x in self.inputs["stiffness"].text().split(",")]
            t_max = float(self.inputs["t_max"].text())
            dt = float(self.inputs["dt"].text())
            self.num_dofs = int(self.inputs["num_dofs"].text())

            if len(m) != self.num_dofs or len(c) != self.num_dofs or len(k) != self.num_dofs:
                raise ValueError("自由度参数数量不匹配")

            t = np.arange(0, t_max, dt)

            if load_type == "正弦冲击":
                params = (A, f, D)
            elif load_type == "自定义冲击":
                params = self.inputs["custom_expression"].text()
                self.load_model.set_custom_expression(params)
            else:
                params = (A, D)

            force = self.load_model.generate_impact_series(load_type, params, t, num, interval)
            self.response_model = MultiDOFResponse(self.num_dofs)
            u, v, a = self.response_model.response(force, t, m, c, k)

            # 导出结果到CSV文件
            with open(file_name, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                header = ["时间"]
                for i in range(self.num_dofs):
                    header.append(f"位移 DOF {i + 1}")
                    header.append(f"速度 DOF {i + 1}")
                    header.append(f"加速度 DOF {i + 1}")
                writer.writerow(header)

                for i in range(len(t)):
                    row = [t[i]]
                    for j in range(self.num_dofs):
                        row.append(u[j, i])
                        row.append(v[j, i])
                        row.append(a[j, i])
                    writer.writerow(row)

            QMessageBox.information(self, "导出成功", "结果已成功导出到文件。")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中发生错误: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImpactResponseApp()
    window.show()
    sys.exit(app.exec_())


