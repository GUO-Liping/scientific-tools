'''
该程序的功能为：
（1）读取高速视频
（2）播放高速视频
（3）调节画面亮度、对比度、饱和度、gamma
（4）抽取任意帧的原始画面
（5）与v1.0相比为pyav库的实现，运行速度最快，且非常稳定
'''


import sys
import cv2
import numpy as np
from pathlib import Path
import av 

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QGroupBox, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高速视频画面逐帧提取 (PyAV 极速版)")
        self.resize(1440, 900)

        # PyAV 相关变量
        self.container = None
        self.video_stream = None
        self.frame_generator = None  # 用于连续播放的生成器

        self.current_frame_num = 0
        self.total_frames = 0
        self.fps = 0.0
        self.is_playing = False

        # 图像处理参数
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0
        self.preview_scale = 1.0 

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
        # 使用 sliderReleased 或 blockSignals 避免滑动时造成频繁的 seek 卡顿
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
        self.btn_prev.clicked.connect(lambda: self.jump_to_frame(self.current_frame_num - 1))
        self.btn_next.clicked.connect(lambda: self.jump_to_frame(self.current_frame_num + 1))
        self.btn_save.clicked.connect(self.save_current_frame)
        self.btn_reset.clicked.connect(self.reset_params)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame_auto)

    def change_preview_scale(self):
        scales = [1.0, 0.5, 0.375, 0.25]
        self.preview_scale = scales[self.combo_scale.currentIndex()]
        if self.container:
            self.jump_to_frame(self.current_frame_num)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "视频 (*.mp4 *.avi *.mkv *.mov)")
        if not path: return

        try:
            # 初始化 PyAV Container
            self.container = av.open(path)
            self.video_stream = self.container.streams.video[0]
            
            # 开启底层多线程解码提高速度
            self.video_stream.thread_type = "AUTO"

            # 获取总帧数与 FPS
            self.total_frames = self.video_stream.frames
            if self.total_frames <= 0:
                # 某些容器可能无法直接读取总帧数，尝试估算
                self.total_frames = int(self.video_stream.duration * self.video_stream.average_rate)
            
            self.fps = float(self.video_stream.average_rate)

            # 配置进度条
            self.slider.setRange(0, self.total_frames - 1)
            self.current_frame_num = 0
            
            self.reset_params()
            self.jump_to_frame(0)

            self.info_label.setText(f"视频已加载 | 总帧数: {self.total_frames:,} | FPS: {self.fps:.1f}")

        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))

    def apply_enhancement(self, frame_bgr):
        # 1. 亮度与对比度
        img = cv2.convertScaleAbs(frame_bgr, alpha=self.contrast, beta=self.brightness)

        # 2. Gamma 校正
        if abs(self.gamma - 1.0) > 0.01:
            inv_gamma = 1.0 / self.gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img = cv2.LUT(img, table)

        # 3. 饱和度
        if abs(self.saturation - 1.0) > 0.01:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
            hsv[:, :, 1] *= self.saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

        return img

    def jump_to_frame(self, frame_num):
        """精准定位到某一帧并渲染"""
        if not self.container: return
        self.current_frame_num = max(0, min(frame_num, self.total_frames - 1))

        # 计算对应的时间戳
        time_base = self.video_stream.time_base
        pts = int(self.current_frame_num / self.fps / time_base)

        # 1. Seek 到关键帧 (Keyframe)
        self.container.seek(pts, stream=self.video_stream)
        
        # 2. 顺序解码步进，直到到达目标帧
        self.frame_generator = self.container.decode(video=0)
        for frame in self.frame_generator:
            # 依靠 pts 或估算帧号定位
            current_pts_frame = int(frame.pts * time_base * self.fps) if frame.pts else self.current_frame_num
            if current_pts_frame >= self.current_frame_num:
                self.render_frame(frame)
                break

    def render_frame(self, av_frame):
        """处理并显示单帧"""
        # PyAV 帧转换为 numpy (RGB 格式)
        frame_rgb = av_frame.to_ndarray(format='rgb24')
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 图像增强
        enhanced = self.apply_enhancement(frame_bgr)
        self.current_frame_bgr = enhanced.copy()  # 备份供导出

        # 预览缩放
        if self.preview_scale < 1.0:
            h, w = enhanced.shape[:2]
            preview = cv2.resize(enhanced, (int(w * self.preview_scale), int(h * self.preview_scale)))
        else:
            preview = enhanced

        # 显示到控件
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qt_img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio)

        self.label.setPixmap(pixmap)
        
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_num)
        self.slider.blockSignals(False)

        self.info_label.setText(f"帧: {self.current_frame_num:,}/{self.total_frames:,}   "
                               f"亮度:{self.brightness}  对比度:{self.contrast:.2f}  "
                               f"饱和度:{self.saturation:.2f}  Gamma:{self.gamma:.2f}")

    def update_params(self):
        self.brightness = self.slider_bright.value()
        self.contrast = self.slider_contrast.value() / 100.0
        self.saturation = self.slider_sat.value() / 100.0
        self.gamma = self.slider_gamma.value() / 100.0

        if self.container and not self.is_playing:
            self.jump_to_frame(self.current_frame_num)

    def reset_params(self):
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0

        self.slider_bright.setValue(0)
        self.slider_contrast.setValue(100)
        self.slider_sat.setValue(100)
        self.slider_gamma.setValue(100)

        if self.container and not self.is_playing:
            self.jump_to_frame(self.current_frame_num)

    def slider_moved(self, val):
        self.info_label.setText(f"正在定位至帧: {val} ...")

    def slider_released(self):
        if self.container:
            self.jump_to_frame(self.slider.value())

    def toggle_play(self):
        if not self.container: return
        self.is_playing = not self.is_playing
        self.btn_play.setText("暂停" if self.is_playing else "播放")
        
        if self.is_playing:
            # 播放时重置生成器，使其从当前帧连续向下读，杜绝卡顿
            self.jump_to_frame(self.current_frame_num)
            # 高速视频设置较小定时器间隔
            self.timer.start(1)  
        else:
            self.timer.stop()

    def next_frame_auto(self):
        """流式连续解码播放，这是 PyAV 流畅播放的核心逻辑"""
        if not self.is_playing or not self.frame_generator:
            return

        try:
            # 针对 500fps-1000fps 等极端高速视频，如果逐帧播放会因为 UI 刷新率跟不上显得卡顿。
            # 这里允许连续跳帧播放（例如每次取第3帧或第5帧渲染）
            step = 1 if self.fps < 200 else 4
            
            frame = None
            for _ in range(step):
                frame = next(self.frame_generator)
                self.current_frame_num += 1

            if frame and self.current_frame_num < self.total_frames:
                self.render_frame(frame)
            else:
                self.toggle_play()  # 播完结束
        except StopIteration:
            self.toggle_play()

    def save_current_frame(self):
        if not hasattr(self, 'current_frame_bgr'):
            return
        filename = f"frame_{self.current_frame_num:06d}.png"
        cv2.imwrite(filename, self.current_frame_bgr)
        QMessageBox.information(self, "保存成功", f"已保存该帧原始画面至:\n{filename}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoPlayer()
    win.show()
    sys.exit(app.exec())