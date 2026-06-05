'''
该程序的功能为：
（1）读取高速视频
（2）播放高速视频
（3）调节画面亮度、对比度、饱和度、gamma
（4）抽取任意帧的原始画面
（5）decord库的实现-2021停更
'''
import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QGroupBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高速视频逐帧提取与极速调谐工具")
        self.resize(1440, 900)

        self.vr = None
        self.current_frame_num = 0
        self.total_frames = 0
        self.fps = 0
        self.is_playing = False

        # 调色参数
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0
        self.preview_scale = 1.0   

        # 核心优化：全流程预缓存的查找表与色彩矩阵
        self.lut_table = None
        self.sat_matrix = None
        self.current_frame_rgb = None # 保存用原始分辨率RGB画面
        
        self.update_lut_and_matrix() # 初始化构建初始LUT
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        self.label = QLabel("点击 打开视频")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(1000, 600)
        main.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        # 优化点 1：解耦滑块事件，避免滑动时频繁 Seek 触发卡死
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.sliderReleased.connect(self.slider_released)
        main.addWidget(self.slider)

        # 控制按钮
        btns = QHBoxLayout()
        self.btn_open = QPushButton("打开视频")
        self.btn_play = QPushButton("播放")
        self.btn_prev = QPushButton("← 上一帧")
        self.btn_next = QPushButton("下一帧 →")
        self.btn_save = QPushButton("保存原始帧画面")
        self.btn_reset = QPushButton("重置参数")

        for b in [self.btn_open, self.btn_play, self.btn_prev, self.btn_next, self.btn_save, self.btn_reset]:
            btns.addWidget(b)
        main.addLayout(btns)

        # 预览分辨率选择
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("预览分辨率:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["原分辨率 (默认)", "2k (推荐)", "1080p (流畅)", "720p (最快)"])
        self.combo_scale.setCurrentIndex(0)
        self.combo_scale.currentIndexChanged.connect(self.change_preview_scale)
        scale_layout.addWidget(self.combo_scale)
        main.addLayout(scale_layout)

        # 参数调整
        param_group = QGroupBox("图像调整")
        param_lay = QHBoxLayout()

        for name, slider, range_ in [
            ("亮度", "bright", (-100, 100)),
            ("对比度", "contrast", (50, 300)),
            ("饱和度", "sat", (50, 300)),
            ("Gamma", "gamma", (50, 300))
        ]:
            vbox = QVBoxLayout()
            vbox.addWidget(QLabel(name))
            s = QSlider(Qt.Orientation.Vertical)
            s.setRange(range_[0], range_[1])
            s.setValue(100 if name != "亮度" else 0)
            s.valueChanged.connect(self.update_params)
            vbox.addWidget(s)
            setattr(self, f"slider_{slider}", s)
            param_lay.addLayout(vbox)

        param_group.setLayout(param_lay)
        main.addWidget(param_group)

        self.info_label = QLabel("未加载视频")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(self.info_label)

        # 信号绑定
        self.btn_open.clicked.connect(self.open_video)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_prev.clicked.connect(lambda: self.show_frame(self.current_frame_num - 1))
        self.btn_next.clicked.connect(lambda: self.show_frame(self.current_frame_num + 1))
        self.btn_save.clicked.connect(self.save_current_frame)
        self.btn_reset.clicked.connect(self.reset_params)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame_auto)

    def change_preview_scale(self):
        scales = [1.0, 0.5, 0.375, 0.25]
        self.preview_scale = scales[self.combo_scale.currentIndex()]
        if self.vr:
            self.show_frame(self.current_frame_num)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "视频 (*.mp4 *.avi *.mkv *.mov)")
        if not path: return

        try:
            # 使用 cpu 缓存提升连续读取性能
            self.vr = VideoReader(path, ctx=cpu(0))
            self.total_frames = len(self.vr)
            self.fps = self.vr.get_avg_fps()

            self.slider.setRange(0, self.total_frames - 1)
            self.current_frame_num = 0
            self.reset_params()
            self.show_frame(0)

            self.info_label.setText(f"视频已加载 | 总帧数: {self.total_frames:,} | FPS: {self.fps:.1f}")

        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))

    def update_lut_and_matrix(self):
        """优化点 2：将亮度、对比度、Gamma 合并进单张查找表(LUT)，滑块移动时仅计算一次"""
        # 1. 预计算 亮度 + 对比度 + Gamma 的组合映射
        inv_gamma = 1.0 / self.gamma
        i = np.arange(256, dtype=np.float32)
        
        # 线性应用 对比度 和 亮度
        res = i * self.contrast + self.brightness
        res = np.clip(res, 0, 255)
        
        # 应用 Gamma 校正
        if abs(self.gamma - 1.0) > 0.01:
            res = ((res / 255.0) ** inv_gamma) * 255.0
            
        self.lut_table = np.clip(res, 0, 255).astype("uint8")

        # 2. 优化点 3：预计算 RGB 线性饱和度矩阵，规避昂贵的 HSV 转换空间
        # 转换原理使用权重：R:0.299, G:0.587, B:0.114
        s = self.saturation
        self.sat_matrix = np.array([
            [0.299 + 0.701*s, 0.587 - 0.587*s, 0.114 - 0.114*s],
            [0.299 - 0.299*s, 0.587 + 0.413*s, 0.114 - 0.114*s],
            [0.299 - 0.299*s, 0.587 - 0.587*s, 0.114 + 0.886*s]
        ], dtype=np.float32).T

    def apply_enhancement_fast(self, frame_rgb):
        """高效纯 NumPy / OpenCV 增强链（无全图颜色空间转换拷贝）"""
        # 1. 一步到位应用 亮度/对比度/Gamma 联合 LUT 映射
        img = cv2.LUT(frame_rgb, self.lut_table)

        # 2. 使用矩阵点积快速调整饱和度 (比 cv2.cvtColor(HSV) 快 3-4 倍)
        if abs(self.saturation - 1.0) > 0.01:
            img = np.dot(img.astype(np.float32), self.sat_matrix)
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def show_frame(self, frame_num):
        if not self.vr: return
        self.current_frame_num = max(0, min(frame_num, self.total_frames - 1))

        # 读取原始帧 (Decord 默认返回高维 RGB 格式)
        frame_rgb = self.vr[self.current_frame_num].asnumpy()

        # 图像实时增强
        enhanced_rgb = self.apply_enhancement_fast(frame_rgb)
        self.current_frame_rgb = enhanced_rgb  # 备份供导出，免除 BGR 转换

        # 预览分辨率缩放
        if self.preview_scale < 1.0:
            h, w = enhanced_rgb.shape[:2]
            preview = cv2.resize(enhanced_rgb, (int(w * self.preview_scale), int(h * self.preview_scale)), interpolation=cv2.INTER_NEAREST)
        else:
            preview = enhanced_rgb

        # 显示（直接使用 RGB 构造 QImage，无转 BGR 再转回的操作）
        h, w, c = preview.shape
        qt_img = QImage(preview.data, w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        
        self.label.setPixmap(pixmap)
        
        # 阻止信号避免死循环触发 valueChanged
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_num)
        self.slider.blockSignals(False)

        self.info_label.setText(f"帧: {self.current_frame_num:,}/{self.total_frames:,}   "
                               f"亮度:{self.brightness}  对比度:{self.contrast:.2f}  饱和度:{self.saturation:.2f}  Gamma:{self.gamma:.2f}")

    def update_params(self):
        self.brightness = self.slider_bright.value()
        self.contrast = self.slider_contrast.value() / 100.0
        self.saturation = self.slider_sat.value() / 100.0
        self.gamma = self.slider_gamma.value() / 100.0

        # 参数改变，重新计算全局映射矩阵
        self.update_lut_and_matrix()

        if self.vr and not self.is_playing:
            self.show_frame(self.current_frame_num)

    def reset_params(self):
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0

        self.slider_bright.setValue(0)
        self.slider_contrast.setValue(100)
        self.slider_sat.setValue(100)
        self.slider_gamma.setValue(100)

        self.update_lut_and_matrix()

        if self.vr and not self.is_playing:
            self.show_frame(self.current_frame_num)

    def slider_moved(self, val):
        # 仅仅做信息提示，不进行图像解码，防止卡死
        self.info_label.setText(f"释放鼠标跳转至第 {val:,} 帧...")

    def slider_released(self):
        # 鼠标松开的一瞬间触发单次精确 Seek 解码
        if self.vr:
            self.show_frame(self.slider.value())

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("暂停" if self.is_playing else "播放")
        if self.is_playing:
            self.timer.start(1)       
        else:
            self.timer.stop()

    def next_frame_auto(self):
        # 依据视频原本帧率，自动匹配跳帧步长
        step = 1 if self.fps < 120 else (int(self.fps // 60))
        if self.current_frame_num + step < self.total_frames:
            self.show_frame(self.current_frame_num + step)
        else:
            self.toggle_play()

    def save_current_frame(self):
        if self.current_frame_rgb is None:
            return
        filename = f"frame_{self.current_frame_num:06d}.png"
        
        # 唯独在保存磁盘写入时，才进行一次转 BGR 契合 opencv 的写出规范
        save_bgr = cv2.cvtColor(self.current_frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, save_bgr)
        QMessageBox.information(self, "保存成功", f"已保存原始 4K/高清 图像至:\n{filename}")


if __name__ == "__main__":
    if not DECORD_AVAILABLE:
        QMessageBox.critical(None, "缺少库", "未检测到 decord 库")
        sys.exit(1)

    app = QApplication(sys.argv)
    win = VideoPlayer()
    win.show()
    sys.exit(app.exec())