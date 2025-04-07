# -*- coding: UTF-8 -*-
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
                             QWidget, QMainWindow)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义脉冲荷载函数
def pulse_load(t, pulse_type, amplitude, duration):
    """根据脉冲类型返回荷载值"""
    if pulse_type == "矩形脉冲":
        # 在0到duration时间内幅值为amplitude，其后为0
        return amplitude if 0 <= t <= duration else 0
    elif pulse_type == "三角脉冲":
        # 三角形脉冲，0到duration/2上升，duration/2到duration下降
        if 0 <= t < duration/2:
            return 2 * amplitude * t / duration
        elif duration/2 <= t <= duration:
            return 2 * amplitude * (duration - t) / duration
        else:
            return 0
    elif pulse_type == "正弦脉冲":
        # 正弦脉冲，在0到duration内用半个周期正弦函数（从0到π）
        return amplitude * np.sin(np.pi * t / duration) if 0 <= t <= duration else 0
    else:
        return 0

# 使用RK4方法求解单自由度系统的微分方程
def simulate_response(m, c, k, pulse_type, amplitude, duration, t_end=5.0, dt=0.001):
    """
    求解方程: m*x'' + c*x' + k*x = F(t)
    使用初始条件 x(0)=0, x'(0)=0
    """
    # 时间步长
    N = int(t_end/dt)
    t = np.linspace(0, t_end, N)
    x = np.zeros(N)
    v = np.zeros(N)
    
    # 定义微分方程：dx/dt = v, dv/dt = (F(t) - c*v - k*x) / m
    def deriv(xi, vi, time):
        F = pulse_load(time, pulse_type, amplitude, duration)
        ax = (F - c*vi - k*xi) / m
        return ax
    
    # RK4积分
    for i in range(N-1):
        ti = t[i]
        xi = x[i]
        vi = v[i]
        
        k1_v = deriv(xi, vi, ti)
        k1_x = vi
        
        k2_v = deriv(xi + dt*k1_x/2, vi + dt*k1_v/2, ti + dt/2)
        k2_x = vi + dt*k1_v/2
        
        k3_v = deriv(xi + dt*k2_x/2, vi + dt*k2_v/2, ti + dt/2)
        k3_x = vi + dt*k2_v/2
        
        k4_v = deriv(xi + dt*k3_x, vi + dt*k3_v, ti + dt)
        k4_x = vi + dt*k3_v
        
        x[i+1] = xi + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        v[i+1] = vi + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    
    return t, x

# 定义主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("单次冲击下单自由度结构动力学分析")
        self.initUI()
    
    def initUI(self):
        # 主界面Widget和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 参数输入区
        form_layout = QFormLayout()
        
        self.mass_input = QLineEdit("1.0")
        form_layout.addRow("质量 m (kg):", self.mass_input)
        
        self.damping_input = QLineEdit("0.2")
        form_layout.addRow("阻尼 c (N·s/m):", self.damping_input)
        
        self.stiffness_input = QLineEdit("10.0")
        form_layout.addRow("刚度 k (N/m):", self.stiffness_input)
        
        self.amplitude_input = QLineEdit("5.0")
        form_layout.addRow("脉冲幅值 (N):", self.amplitude_input)
        
        self.duration_input = QLineEdit("0.1")
        form_layout.addRow("脉冲持续时间 (s):", self.duration_input)
        
        # 荷载类型选择
        self.load_type_combo = QComboBox()
        self.load_type_combo.addItems(["矩形脉冲", "三角脉冲", "正弦脉冲"])
        form_layout.addRow("脉冲荷载形式:", self.load_type_combo)
        
        main_layout.addLayout(form_layout)
        
        # 分析按钮
        self.run_button = QPushButton("运行分析")
        self.run_button.clicked.connect(self.run_analysis)
        main_layout.addWidget(self.run_button)
        
        # Matplotlib图形嵌入
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
    
    def run_analysis(self):
        try:
            m = float(self.mass_input.text())
            c = float(self.damping_input.text())
            k = float(self.stiffness_input.text())
            amplitude = float(self.amplitude_input.text())
            duration = float(self.duration_input.text())
            load_type = self.load_type_combo.currentText()
        except ValueError:
            print("输入参数有误，请检查。")
            return
        
        # 计算响应（此处t_end可以根据需要延长）
        t, x = simulate_response(m, c, k, load_type, amplitude, duration, t_end=5.0, dt=0.001)
        
        # 绘制响应曲线
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t, x, label="位移响应")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("位移 (m)")
        ax.set_title("单自由度系统响应")
        ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
