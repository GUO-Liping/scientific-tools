import sys
import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QWidget, QLabel, QFileDialog, QInputDialog, QMessageBox, QSlider, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QWheelEvent, QMouseEvent, QPaintEvent, QPainter

# ==========================================
# 1. 高性能视口交互组件
# ==========================================
class VideoWidget(QWidget):
    clicked_pos = pyqtSignal(int, int)
    moved_pos = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444;")
        self.setCursor(Qt.CursorShape.BlankCursor) 
        self.setMouseTracking(True) 

        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
        self.is_panning = False
        self.last_mouse_pos = None
        self.canvas_w = 0
        self.canvas_h = 0
        self.current_qimage = None

    def update_image(self, qimg):
        self.current_qimage = qimg
        self.update() 

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        if self.current_qimage:
            painter.drawImage(self.rect(), self.current_qimage)
        painter.end()

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0

    def _get_canvas_coords(self, pos):
        if self.canvas_w <= 0 or self.canvas_h <= 0: return None
        lw, lh = self.width(), self.height()
        view_w = self.canvas_w * self.zoom_level
        view_h = self.canvas_h * self.zoom_level
        x_offset = (lw - view_w) / 2 + self.pan_offset_x
        y_offset = (lh - view_h) / 2 + self.pan_offset_y
        canvas_x = (pos.x() - x_offset) / self.zoom_level
        canvas_y = (pos.y() - y_offset) / self.zoom_level
        if 0 <= canvas_x < self.canvas_w and 0 <= canvas_y < self.canvas_h:
            return int(canvas_x), int(canvas_y)
        return None

    def wheelEvent(self, event: QWheelEvent):
        old_coords = self._get_canvas_coords(event.position().toPoint())
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_zoom = self.zoom_level * zoom_factor
        if 0.5 <= new_zoom <= 40.0:
            self.zoom_level = new_zoom
            if old_coords:
                cx, cy = old_coords
                lw, lh = self.width(), self.height()
                self.pan_offset_x = event.position().x() - (lw / 2 + (cx - self.canvas_w / 2) * self.zoom_level)
                self.pan_offset_y = event.position().y() - (lh / 2 + (cy - self.canvas_h / 2) * self.zoom_level)
            self.moved_pos.emit(-1, -1) 

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
        elif event.button() == Qt.MouseButton.LeftButton:
            coords = self._get_canvas_coords(event.position().toPoint())
            if coords: self.clicked_pos.emit(*coords)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_panning and self.last_mouse_pos:
            delta = event.position() - self.last_mouse_pos
            self.pan_offset_x += delta.x()
            self.pan_offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.moved_pos.emit(-1, -1)
        else:
            coords = self._get_canvas_coords(event.position().toPoint())
            if coords: 
                self.moved_pos.emit(*coords)


# ==========================================
# 2. 主窗口控制层
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高速视频运动分析工具 V1.0")
        self.resize(1600, 900)

        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = -1 
        self.base_frame = None       

        self.state = "IDLE" 
        self.calib_pt1 = None
        self.scale_factor = 1.0 
        self.physics_unit = "单位" 
        self.origin_pt = None   
        self.mouse_curr_pos = None 

        self.marks = {}
        # 零点时刻同步变量
        self.first_time_sec = None  # 记录捕捉模式第一个点的绝对时间(s)
        
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        self.video_widget = VideoWidget() 
        self.video_widget.clicked_pos.connect(self.on_video_clicked)
        self.video_widget.moved_pos.connect(self.on_video_moved)
        left_layout.addWidget(self.video_widget, stretch=1) 

        slider_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_prev = QPushButton("◀ 上一帧")
        self.btn_next = QPushButton("下一帧 ▶")
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setTracking(False) 
        
        self.label_frame_info = QLabel("帧进度: 0 / 0")
        
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_prev.clicked.connect(lambda: self.seek_frame(-1))
        self.btn_next.clicked.connect(lambda: self.seek_frame(1))
        self.slider.valueChanged.connect(self.on_slider_changed)

        slider_layout.addWidget(self.btn_play)
        slider_layout.addWidget(self.btn_prev)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.btn_next)
        slider_layout.addWidget(self.label_frame_info)
        left_layout.addLayout(slider_layout)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("1. 加载视频")
        self.btn_calib = QPushButton("2. 标定尺寸")
        self.btn_origin = QPushButton("3. 设原点")
        
        self.label_step = QLabel("捕捉步长:")
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 100)
        self.spin_step.setValue(1)
        self.spin_step.setSuffix(" 帧")
        self.spin_step.setEnabled(False)

        self.btn_mark = QPushButton("4. 开始捕捉")
        self.btn_export = QPushButton("5. 导出 Excel")
        self.btn_reset_view = QPushButton("复位画面") 
        
        for btn in [self.btn_play, self.btn_calib, self.btn_origin, self.btn_mark, self.btn_export, self.btn_reset_view]:
            btn.setEnabled(False)

        self.btn_load.clicked.connect(self.load_video)
        self.btn_calib.clicked.connect(lambda: self.set_state("CALIBRATING_1"))
        self.btn_origin.clicked.connect(lambda: self.set_state("SET_ORIGIN"))
        self.btn_mark.clicked.connect(lambda: self.set_state("MARKING"))
        self.btn_export.clicked.connect(self.export_to_excel)
        self.btn_reset_view.clicked.connect(self.reset_label_view)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_calib)
        btn_layout.addWidget(self.btn_origin)
        btn_layout.addSpacing(10)
        btn_layout.addWidget(self.label_step)
        btn_layout.addWidget(self.spin_step)
        btn_layout.addWidget(self.btn_mark)
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_reset_view)
        left_layout.addLayout(btn_layout)
        
        self.status_label = QLabel("状态: 等待加载视频...")
        self.status_label.setStyleSheet("font-weight: bold; color: #d35400;")
        left_layout.addWidget(self.status_label)

        right_layout = QVBoxLayout()
        self.plot_x = pg.PlotWidget(title="水平 (X轴) 位移图")
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)
        self.curve_x = self.plot_x.plot(pen=pg.mkPen('c', width=2), symbol='o', symbolSize=5)

        self.plot_y = pg.PlotWidget(title="垂直 (Y轴) 位移图")
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        self.curve_y = self.plot_y.plot(pen=pg.mkPen('m', width=2), symbol='o', symbolSize=5)

        self.update_chart_labels()
        right_layout.addWidget(self.plot_x)
        right_layout.addWidget(self.plot_y)

        layout.addLayout(left_layout, stretch=2)
        layout.addLayout(right_layout, stretch=1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_frame()

    def set_state(self, state):
        self.state = state
        msgs = {
            "IDLE": "空闲。滚轮缩放，右键按住拖动画面，左下角可点击播放/暂停。",
            "CALIBRATING_1": "【尺寸标定】: 请点击定标尺的【起点】",
            "CALIBRATING_2": "【尺寸标定】: 请点击定标尺的【终点】",
            "SET_ORIGIN": "【建立坐标系】: 请点击设定【物理坐标原点 (0,0)】",
            "MARKING": "【捕捉采集模式】: 请点击捕捉特征目标（点击后自动按设定的步长跳帧）"
        }
        self.status_label.setText(f"状态: {msgs.get(state, state)}")
        self.render_frame()

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶ 播放")
        else:
            interval = max(1, int(1000 / self.fps))
            self.play_timer.start(interval)
            self.btn_play.setText("⏸ 暂停")

    def play_next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.seek_frame(1)
        else:
            self.play_timer.stop()
            self.btn_play.setText("▶ 播放")

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if filename:
            if self.play_timer.isActive(): self.toggle_play()
            if self.cap: self.cap.release()
            
            self.cap = cv2.VideoCapture(filename)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            scale_pre = min(1280 / video_w, 720 / video_h)
            self.video_widget.canvas_w = int(video_w * scale_pre)
            self.video_widget.canvas_h = int(video_h * scale_pre)
            self.video_widget.reset_view()

            self.marks.clear()
            self.origin_pt = None
            self.calib_pt1 = None
            self.physics_unit = "单位"
            self.current_frame_idx = -1 
            self.first_time_sec = None # 重置时间零点基准
            
            self.update_chart_labels()
            self.update_plots()
            
            self.slider.blockSignals(True)
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            
            for btn in [self.slider, self.btn_play, self.btn_calib, self.btn_origin, self.btn_mark, self.btn_export, self.btn_reset_view, self.spin_step]:
                btn.setEnabled(True)
            
            self.seek_frame(0, absolute=True)
            self.set_state("IDLE")

    def reset_label_view(self):
        self.video_widget.reset_view()
        self.render_frame()

    def seek_frame(self, val, absolute=False):
        if not self.cap: return
        
        target_idx = val if absolute else self.current_frame_idx + val
        target_idx = max(0, min(target_idx, self.total_frames - 1))
        diff = target_idx - self.current_frame_idx
        
        ret = False
        frame = None
        
        if 0 < diff < 200:
            for _ in range(diff - 1):
                self.cap.grab()
            ret, frame = self.cap.read()
        elif diff == 0 and self.base_frame is not None:
            return
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = self.cap.read()
            
        if ret:
            self.current_frame_idx = target_idx
            self.base_frame = cv2.resize(frame, (self.video_widget.canvas_w, self.video_widget.canvas_h))
            
            self.slider.blockSignals(True)
            self.slider.setValue(target_idx)
            self.slider.blockSignals(False)
            
            self.label_frame_info.setText(f"帧进度: {target_idx} / {self.total_frames - 1}")
            self.render_frame()

    def on_slider_changed(self, position):
        self.seek_frame(position, absolute=True)

    def on_video_moved(self, x, y):
        if not (x == -1 and y == -1):
            self.mouse_curr_pos = (x, y)
        self.render_frame()

    def render_frame(self):
        if self.base_frame is None: return
        
        lw = self.video_widget.width()
        lh = self.video_widget.height()
        if lw <= 0 or lh <= 0: return 
        
        zoom = self.video_widget.zoom_level
        pan_x = self.video_widget.pan_offset_x
        pan_y = self.video_widget.pan_offset_y
        
        h_orig, w_orig = self.base_frame.shape[:2]
        view_w = int(w_orig * zoom)
        view_h = int(h_orig * zoom)
        view_img = cv2.resize(self.base_frame, (view_w, view_h))

        screen_output = np.zeros((lh, lw, 3), dtype=np.uint8) + 26 

        start_x = int((lw - view_w) / 2 + pan_x)
        start_y = int((lh - view_h) / 2 + pan_y)
        
        src_y1, src_x1 = max(0, -start_y), max(0, -start_x)
        src_y2, src_x2 = min(view_h, lh - start_y), min(view_w, lw - start_x)
        dst_y1, dst_x1 = max(0, start_y), max(0, start_x)
        dst_y2, dst_x2 = min(lh, start_y + view_h), min(lw, start_x + view_w)

        if (src_y2 > src_y1) and (src_x2 > src_x1):
            screen_output[dst_y1:dst_y2, dst_x1:dst_x2] = view_img[src_y1:src_y2, src_x1:src_x2]

        def to_screen_pixel(canvas_x, canvas_y):
            scr_x = int((lw - view_w) / 2 + pan_x + canvas_x * zoom)
            scr_y = int((lh - view_h) / 2 + pan_y + canvas_y * zoom)
            return scr_x, scr_y

        if self.state == "CALIBRATING_2" and self.calib_pt1 and self.mouse_curr_pos:
            p1 = to_screen_pixel(*self.calib_pt1)
            p2 = to_screen_pixel(*self.mouse_curr_pos)
            cv2.line(screen_output, p1, p2, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(screen_output, p1, 4, (0, 0, 255), -1)
        elif self.calib_pt1:
            p1 = to_screen_pixel(*self.calib_pt1)
            cv2.circle(screen_output, p1, 4, (0, 0, 255), -1)

        if self.origin_pt:
            ox, oy = to_screen_pixel(*self.origin_pt)
            cv2.line(screen_output, (ox, 0), (ox, lh), (0, 102, 255), 1, lineType=cv2.LINE_AA) 
            cv2.line(screen_output, (0, oy), (lw, oy), (0, 102, 255), 1, lineType=cv2.LINE_AA) 
            cv2.circle(screen_output, (ox, oy), 5, (0, 0, 255), -1)
            cv2.putText(screen_output, "Origin (0,0)", (ox + 8, oy - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        if self.current_frame_idx in self.marks:
            data = self.marks[self.current_frame_idx]
            px, py = to_screen_pixel(*data['pixel_pt'])
            rx, ry = data['rel_x'], data['rel_y']
            cv2.circle(screen_output, (px, py), 4, (0, 255, 0), -1)
            
            # 屏幕标签上如果设置了零点时刻，也显示相对时间便于核对
            if self.first_time_sec is not None:
                rel_time = data['time'] - self.first_time_sec
                text = f"X:{rx:.2f} Y:{ry:.2f} | t:{rel_time:.3f}s"
            else:
                text = f"X: {rx:.2f}, Y: {ry:.2f} {self.physics_unit}"
                
            cv2.putText(screen_output, text, (px + 10, py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        if self.mouse_curr_pos:
            mx, my = to_screen_pixel(*self.mouse_curr_pos)
            if 0 <= mx < lw and 0 <= my < lh:
                cv2.line(screen_output, (mx, 0), (mx, lh), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.line(screen_output, (0, my), (lw, my), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        img_rgb = cv2.cvtColor(screen_output, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, lw, lh, 3 * lw, QImage.Format.Format_RGB888)
        self.video_widget.update_image(qimg)

    def on_video_clicked(self, x, y):
        if self.state == "CALIBRATING_1":
            self.calib_pt1 = (x, y)
            self.set_state("CALIBRATING_2")
            
        elif self.state == "CALIBRATING_2":
            pixel_dist = math.hypot(x - self.calib_pt1[0], y - self.calib_pt1[1])
            if pixel_dist == 0: pixel_dist = 1.0
            
            real_dist, ok1 = QInputDialog.getDouble(self, "标定物理尺寸", f"像素跨度: {pixel_dist:.2f} px\n请输入实际物理长度:", 100.0, 0.001, 100000, 3)
            if ok1 and real_dist > 0:
                unit_str, ok2 = QInputDialog.getText(self, "设置单位名称", "请输入单位简称 (如 mm, m):", text="mm")
                if ok2 and unit_str.strip():
                    self.physics_unit = unit_str.strip()
                self.scale_factor = real_dist / pixel_dist
                self.update_chart_labels()
                QMessageBox.information(self, "成功", f"标定映射建立！\n1 像素 = {self.scale_factor:.4f} {self.physics_unit}")
            self.calib_pt1 = None
            self.set_state("IDLE")
            
        elif self.state == "SET_ORIGIN":
            self.origin_pt = (x, y)
            self.set_state("IDLE")
            
        elif self.state == "MARKING":
            if not self.origin_pt:
                QMessageBox.warning(self, "警告", "请先第3步点击屏幕建立坐标原点！")
                self.set_state("IDLE")
                return
                
            rel_x = (x - self.origin_pt[0]) * self.scale_factor
            rel_y = (self.origin_pt[1] - y) * self.scale_factor
            time_sec = self.current_frame_idx / self.fps
            
            # 核心新增逻辑（2）：如果是全场打下的第一个特征点，锁死该点的时间为时间零点基准
            if self.first_time_sec is None:
                self.first_time_sec = time_sec
                self.update_chart_labels() # 更新图表的 X轴 标签为相对物理时间(s)
            
            self.marks[self.current_frame_idx] = {
                'time': time_sec, 'rel_x': rel_x, 'rel_y': rel_y, 'pixel_pt': (x, y)
            }
            self.update_plots()
            
            step_size = self.spin_step.value() 
            if self.current_frame_idx + step_size < self.total_frames:
                self.seek_frame(step_size) 
            else:
                self.render_frame()

    def update_chart_labels(self):
        self.plot_x.getAxis('left').setLabel(f"X轴位移 ({self.physics_unit})", units=None)
        self.plot_y.getAxis('left').setLabel(f"Y轴位移 ({self.physics_unit})", units=None)
        
        # 核心新增逻辑（2）：根据是否捕获到第一个点，动态更改右侧图表的横轴标题
        if self.first_time_sec is not None:
            self.plot_x.getAxis('bottom').setLabel("相对物理时间", units='s')
            self.plot_y.getAxis('bottom').setLabel("相对物理时间", units='s')
        else:
            self.plot_x.getAxis('bottom').setLabel("时间/进度", units='帧数')
            self.plot_y.getAxis('bottom').setLabel("时间/进度", units='帧数')

    def update_plots(self):
        """核心重构点（2）：绘图时若存在零点基准，横轴直接减去基准值做平移映射"""
        sorted_keys = sorted(self.marks.keys())
        
        if self.first_time_sec is not None:
            # 建立零点后，横坐标使用平移后的相对时间(s)
            x_axis_data = [self.marks[k]['time'] - self.first_time_sec for k in sorted_keys]
        else:
            # 未建立零点前，横轴默认依然显示帧数
            x_axis_data = [k for k in sorted_keys]
            
        xs = [self.marks[k]['rel_x'] for k in sorted_keys]
        ys = [self.marks[k]['rel_y'] for k in sorted_keys]
        
        self.curve_x.setData(x_axis_data, xs)
        self.curve_y.setData(x_axis_data, ys)

    def export_to_excel(self):
        if not self.marks: return
        if self.play_timer.isActive(): self.toggle_play() 
        filename, _ = QFileDialog.getSaveFileName(self, "数据存档为 Excel", "avalanche_vibration_report.xlsx", "Excel Files (*.xlsx)")
        if filename:
            try:
                sorted_keys = sorted(self.marks.keys())
                data_list = []
                for k in sorted_keys:
                    d = self.marks[k]
                    # 核心新增逻辑（3）：在 Excel 中同时写入绝对时间、绝对帧数、以及平移后的“相对试验时间”
                    rel_time_val = d['time'] - self.first_time_sec if self.first_time_sec is not None else 0.0
                    
                    data_list.append({
                        'Frame_Index (绝对帧数)': k, 
                        'Absolute_Time (绝对时间-秒)': d['time'],
                        'Relative_Time (相对时间-秒/零点平移)': rel_time_val,
                        f'Physical_X ({self.physics_unit})': d['rel_x'],
                        f'Physical_Y ({self.physics_unit})': d['rel_y'],
                        'Canvas_Pixel_X': d['pixel_pt'][0], 
                        'Canvas_Pixel_Y': d['pixel_pt'][1]
                    })
                pd.DataFrame(data_list).to_excel(filename, index=False)
                QMessageBox.information(self, "成功", f"报告成功写入到本地：\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
