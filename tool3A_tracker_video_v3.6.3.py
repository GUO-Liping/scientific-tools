import sys
import os
import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
import math
from collections import OrderedDict
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QWidget, QLabel, QFileDialog, QInputDialog, QMessageBox, QSlider, QSpinBox,
                             QFrame, QSizePolicy, QToolBar, QCheckBox, QComboBox, QToolButton, QMenu)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRectF, QPointF, QSize
from PyQt6.QtGui import (QImage, QWheelEvent, QMouseEvent, QPaintEvent, QPainter, QColor, QPen, QFont,
                         QAction, QKeySequence, QIcon, QPixmap, QActionGroup, QBrush, QPolygonF)

cv2.setUseOptimized(True)
pg.setConfigOptions(background="#101418", foreground="#d7dde8", antialias=True)

# 核心修复：分离 X 轴和 Y 轴的曲线颜色库，完美继承旧版(蓝/粉)的视觉体验
TRACK_COLORS_X = ["#5ac8fa", "#ff9800", "#8cc63f", "#00ffff", "#c678dd", "#ffeb3b", "#00bcd4", "#4caf50"]
TRACK_COLORS_Y = ["#ff5c8a", "#ff4d4f", "#f7931e", "#9e005d", "#3f51b5", "#e91e63", "#00ff00", "#ffffff"]
TRACK_COLORS_VIEW = ["#ff4d4f", "#5ac8fa", "#ffeb3b", "#8cc63f", "#c678dd", "#f7931e", "#00ffff", "#ff5c8a"]

APP_STYLE = """
QMainWindow, QWidget { background: #0f141b; color: #d7dde8; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; font-size: 10pt; }
QMenuBar { background: #111821; color: #d7dde8; border-bottom: 1px solid #263241; }
QMenuBar::item { padding: 6px 10px; background: transparent; }
QMenuBar::item:selected { background: #202a36; }
QMenu { background: #151c25; border: 1px solid #344255; color: #d7dde8; }
QMenu::item { padding: 7px 28px; }
QMenu::item:selected { background: #263241; }
QToolBar { background: #151c25; border: 0; border-bottom: 1px solid #263241; spacing: 4px; padding: 5px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 8px; color: #d7dde8; }
QToolButton:hover { background: #202a36; border-color: #344255; }
QToolButton:pressed { background: #111821; }
QToolButton:disabled { color: #687484; }
QToolButton::menu-indicator { image: none; width: 0px; }
QFrame#controlBar, QFrame#statusBar { background: #151c25; border: 1px solid #263241; border-radius: 8px; }
QPushButton { background: #202a36; border: 1px solid #344255; border-radius: 6px; color: #edf2f8; padding: 7px 12px; min-height: 24px; }
QPushButton::icon { color: #edf2f8; }
QPushButton:hover { background: #2a3747; border-color: #4c5e74; }
QPushButton:pressed { background: #18212c; }
QPushButton:disabled { color: #687484; background: #151b23; border-color: #202936; }
QPushButton#primaryButton { background: #1267d3; border-color: #3181ee; font-weight: 600; }
QPushButton#primaryButton:hover { background: #1976ed; }
QSlider::groove:horizontal { height: 6px; background: #263241; border-radius: 3px; }
QSlider::handle:horizontal { width: 16px; margin: -5px 0; border-radius: 8px; background: #5aa7ff; }
QSpinBox { background: #111821; border: 1px solid #344255; border-radius: 6px; padding: 5px 28px 5px 8px; min-height: 28px; }
QSpinBox::up-button, QSpinBox::down-button { width: 26px; border-left: 1px solid #344255; background: #202a36; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #2a3747; }
QComboBox { background: #111821; border: 1px solid #344255; border-radius: 6px; padding: 4px 10px; color: #d7dde8; min-height: 28px; }
QComboBox::drop-down { border: 0; }
QComboBox QAbstractItemView { background: #151c25; border: 1px solid #344255; color: #d7dde8; selection-background-color: #263241; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QLabel#statusLabel { color: #ffd166; font-weight: 600; }
"""

class VideoWidget(QWidget):
    clicked_pos = pyqtSignal(float, float)
    moved_pos = pyqtSignal(float, float)
    zoom_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #05070a; border: 1px solid #263241; border-radius: 8px;")
        self.setCursor(Qt.CursorShape.BlankCursor) 
        self.setMouseTracking(True) 
        self.setMinimumSize(720, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
        self.is_panning = False
        self.last_mouse_pos = None
        self.canvas_w = 0 
        self.canvas_h = 0
        self.frame_qimage = None
        self.overlay_state = {}
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_frame(self, qimg):
        self.frame_qimage = qimg
        self.update() 

    def set_overlay_state(self, **state):
        self.overlay_state = state
        self.update()

    def fit_to_view(self):
        if self.canvas_w <= 0 or self.canvas_h <= 0 or self.width() <= 0: return
        zw = self.width() / self.canvas_w
        zh = self.height() / self.canvas_h
        self.zoom_level = min(zw, zh) * 0.98
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.zoom_changed.emit(int(self.zoom_level * 100))
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom_level < 3.0)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#05070a"))

        if self.frame_qimage:
            view_rect = self._view_rect()
            painter.drawImage(view_rect, self.frame_qimage, QRectF(self.frame_qimage.rect()))
            self._paint_overlays(painter, view_rect)
        painter.end()

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.zoom_changed.emit(int(self.zoom_level * 100))
        self.update()

    def _view_rect(self):
        view_w = self.canvas_w * self.zoom_level
        view_h = self.canvas_h * self.zoom_level
        x_offset = (self.width() - view_w) / 2 + self.pan_offset_x
        y_offset = (self.height() - view_h) / 2 + self.pan_offset_y
        return QRectF(x_offset, y_offset, view_w, view_h)

    def _get_canvas_coords(self, pos):
        if self.canvas_w <= 0 or self.canvas_h <= 0: return None
        view_rect = self._view_rect()
        x_offset = view_rect.x()
        y_offset = view_rect.y()
        canvas_x = (pos.x() - x_offset) / self.zoom_level
        canvas_y = (pos.y() - y_offset) / self.zoom_level
        if 0 <= canvas_x < self.canvas_w and 0 <= canvas_y < self.canvas_h:
            return float(canvas_x), float(canvas_y) 
        return None

    def _to_screen_point(self, canvas_pt):
        x, y = canvas_pt
        rect = self._view_rect()
        return QPointF(rect.x() + x * self.zoom_level, rect.y() + y * self.zoom_level)

    def _paint_overlays(self, painter, view_rect):
        state = self.overlay_state
        if not state: return
        widget_rect = self.rect()
        painter.setFont(QFont("Segoe UI", 10))

        def draw_textbox(text, x, y, bg_color=QColor(0,0,0,180)):
            painter.setFont(QFont("Segoe UI", 9))
            fm = painter.fontMetrics()
            lines = text.split('\n')
            text_w = max(fm.horizontalAdvance(ln) for ln in lines) + 24
            text_h = fm.lineSpacing() * len(lines) + 16
            rect = QRectF(x, y, text_w, text_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor("#edf2f8"))
            painter.drawText(rect.adjusted(12, 8, -12, -8), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)

        def draw_point(pt, color, radius=4):
            screen = self._to_screen_point(pt)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(screen, radius, radius)
            return screen

        # 提取标定状态
        scale_factor = state.get("scale_factor", 1.0)
        physics_unit = state.get("physics_unit", "单位")
        # 修复2：采用 calib_line 是否存在来严格判断是否已经标定，而不是依赖单位名称
        is_calibrated = (state.get("calib_line") is not None)
        coord_origin = state.get("coord_origin")
        coord_angle = state.get("coord_angle", 0.0)

        def get_formatted_coords(pt):
            if is_calibrated and coord_origin:
                dx = pt[0] - coord_origin[0]
                dy = coord_origin[1] - pt[1]
                rx = (dx * math.cos(coord_angle) + dy * math.sin(coord_angle)) * scale_factor
                ry = (-dx * math.sin(coord_angle) + dy * math.cos(coord_angle)) * scale_factor
                return f"X:{rx:.1f} Y:{ry:.1f} {physics_unit}"
            elif is_calibrated:
                return f"X:{pt[0]*scale_factor:.1f} Y:{pt[1]*scale_factor:.1f} {physics_unit}"
            else:
                return f"X:{int(pt[0])} Y:{int(pt[1])} px"

        # 1. 视频信息
        video_info = state.get("video_info")
        if video_info:
            display_path = video_info['path']
            if len(display_path) > 100: display_path = display_path[:30] + " ... " + display_path[-70:]
            info_text = (
                f"✌️视频地址 {display_path}\n"
                f"🖼️分辨率 {video_info['resolution']}  |  🎬 帧率 {video_info['fps']} FPS  |  "
                f"📊 总帧数 {video_info['frames']}  |  ⏱️ 时长 {video_info['duration']}"
            )
            draw_textbox(info_text, 8, 8)

        # 2. 标定线与十字叉辅助线
        aux_lines = state.get("aux_lines", [])
        aux_current = state.get("aux_current_points", [])
        aux_intersections = state.get("aux_intersections", [])
        aux_midpoints = state.get("aux_midpoints", [])
        
        painter.setPen(QPen(QColor("#c678dd"), 2, Qt.PenStyle.DashLine))
        for line in aux_lines:
            painter.drawLine(self._to_screen_point(line[0]), self._to_screen_point(line[1]))
            draw_point(line[0], "#c678dd", 3); draw_point(line[1], "#c678dd", 3)
            
        if aux_current and len(aux_current) == 1:
            draw_point(aux_current[0], "#c678dd", 3)
            if state.get("mouse_curr_pos"):
                painter.setPen(QPen(QColor("#c678dd"), 2, Qt.PenStyle.DashLine))
                painter.drawLine(self._to_screen_point(aux_current[0]), self._to_screen_point(state["mouse_curr_pos"]))
        
        # 修复1：完整补回透视交点与网格阵列的绘制逻辑！
        if len(aux_intersections) == 4:
            C0, C1, C2, C3 = aux_intersections
            painter.setPen(QPen(QColor(0, 255, 255, 200), 1.5, Qt.PenStyle.SolidLine))
            p0 = self._to_screen_point(C0)
            p1 = self._to_screen_point(C1)
            p2 = self._to_screen_point(C2)
            p3 = self._to_screen_point(C3)
            painter.drawLine(p0, p1)
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)
            painter.drawLine(p3, p0)
            
            grid_size = state.get("grid_size", 10.0)
            if grid_size <= 0: grid_size = 10.0 # 防止除以0
            
            len_top = math.hypot(C1[0]-C0[0], C1[1]-C0[1])
            len_left = math.hypot(C3[0]-C0[0], C3[1]-C0[1])
            phys_w = len_top * scale_factor
            phys_h = len_left * scale_factor
            steps_x = min(200, max(1, int(round(phys_w / grid_size))))
            steps_y = min(200, max(1, int(round(phys_h / grid_size))))
            
            painter.setPen(QPen(QColor(0, 255, 255, 60), 1.0, Qt.PenStyle.DashLine))
            for i in range(1, steps_x):
                t = i / steps_x
                pt_top_x = C0[0] + t * (C1[0] - C0[0])
                pt_top_y = C0[1] + t * (C1[1] - C0[1])
                pt_bot_x = C3[0] + t * (C2[0] - C3[0])
                pt_bot_y = C3[1] + t * (C2[1] - C3[1])
                painter.drawLine(self._to_screen_point((pt_top_x, pt_top_y)), self._to_screen_point((pt_bot_x, pt_bot_y)))
            for i in range(1, steps_y):
                t = i / steps_y
                pt_left_x = C0[0] + t * (C3[0] - C0[0])
                pt_left_y = C0[1] + t * (C3[1] - C0[1])
                pt_right_x = C1[0] + t * (C2[0] - C1[0])
                pt_right_y = C1[1] + t * (C2[1] - C1[1])
                painter.drawLine(self._to_screen_point((pt_left_x, pt_left_y)), self._to_screen_point((pt_right_x, pt_right_y)))

        for pt in aux_intersections:
            p_scr = draw_point(pt, "#ff4d4f", 5)
            painter.setPen(QPen(QColor("#ff4d4f")))
            painter.drawText(p_scr + QPointF(6, 6), "交点")

        for i, pt in enumerate(aux_midpoints):
            p_scr = draw_point(pt, "#e5c07b", 5)
            painter.setPen(QPen(QColor("#e5c07b")))
            lbl = "视场中心" if i == 0 else "中点"
            painter.drawText(p_scr + QPointF(6, -6), lbl)

        # 标定主线
        if state.get("calib_pt1"):
            p1 = self._to_screen_point(state["calib_pt1"])
            draw_point(state["calib_pt1"], "#ff4d4f", 4)
            if state.get("mode") == "CALIBRATING_2" and state.get("mouse_curr_pos"):
                p2 = self._to_screen_point(state["mouse_curr_pos"])
                painter.setPen(QPen(QColor("#ffd43b"), 1.3))
                painter.drawLine(p1, p2)

        calib_line = state.get("calib_line")
        if calib_line:
            p1 = self._to_screen_point(calib_line["p1"])
            p2 = self._to_screen_point(calib_line["p2"])
            painter.setPen(QPen(QColor("#ffd43b"), 1.5))
            painter.drawLine(p1, p2)
            draw_point(calib_line["p1"], "#ffd43b", 4)
            draw_point(calib_line["p2"], "#ffd43b", 4)
            painter.setPen(QPen(QColor("#ffd43b")))
            mid = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
            painter.drawText(mid + QPointF(8, -8), calib_line["text"])

        # 3. 自定义坐标系 (支持旋转)
        if state.get("mode") == "SET_COORD_2" and coord_origin and state.get("mouse_curr_pos"):
            painter.setPen(QPen(QColor("#ff9f1c"), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(self._to_screen_point(coord_origin), self._to_screen_point(state["mouse_curr_pos"]))
            
        if coord_origin:
            origin_scr = self._to_screen_point(coord_origin)
            L = max(widget_rect.width(), widget_rect.height()) * 2 
            ux, uy = math.cos(coord_angle), -math.sin(coord_angle)
            vx, vy = -math.sin(coord_angle), -math.cos(coord_angle)

            painter.setPen(QPen(QColor(255, 159, 28, 180), 1.5))
            painter.drawLine(self._to_screen_point((coord_origin[0] - L*ux, coord_origin[1] - L*uy)), 
                             self._to_screen_point((coord_origin[0] + L*ux, coord_origin[1] + L*uy)))
            painter.setPen(QPen(QColor(76, 175, 80, 180), 1.5)) 
            painter.drawLine(self._to_screen_point((coord_origin[0] - L*vx, coord_origin[1] - L*vy)), 
                             self._to_screen_point((coord_origin[0] + L*vx, coord_origin[1] + L*vy)))

            painter.setPen(QPen(QColor("#ff9f1c")))
            painter.drawText(self._to_screen_point((coord_origin[0] + 120*ux, coord_origin[1] + 120*uy)) + QPointF(5, -5), "X")
            painter.setPen(QPen(QColor("#4caf50")))
            painter.drawText(self._to_screen_point((coord_origin[0] + 120*vx, coord_origin[1] + 120*vy)) + QPointF(5, -5), "Y")
            draw_point(coord_origin, "#ff4d4f", 5)

        # 4. 测量与查询工具
        mode = state.get("mode")
        mouse_pos = state.get("mouse_curr_pos")
        measure_pt1 = state.get("measure_pt1")
        measure_line = state.get("measure_line")

        if mode in ["MEASURE_1", "MEASURE_2"] and mouse_pos:
            scr_m = self._to_screen_point(mouse_pos)
            coord_str = get_formatted_coords(mouse_pos)
            draw_textbox(f"📍 当前坐标: {coord_str}", scr_m.x() + 15, scr_m.y() + 15, QColor(0, 188, 212, 200))

        if mode == "MEASURE_2" and measure_pt1 and mouse_pos:
            p1_scr = self._to_screen_point(measure_pt1)
            p2_scr = self._to_screen_point(mouse_pos)
            painter.setPen(QPen(QColor("#00bcd4"), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(p1_scr, p2_scr)
            draw_point(measure_pt1, "#00bcd4", 4); draw_point(mouse_pos, "#00bcd4", 4)
            
            pixel_dist = math.hypot(mouse_pos[0]-measure_pt1[0], mouse_pos[1]-measure_pt1[1])
            dist_str = f"{pixel_dist * scale_factor:.2f} {physics_unit}" if is_calibrated else f"{pixel_dist:.1f} px"
            mid_scr = QPointF((p1_scr.x() + p2_scr.x()) / 2, (p1_scr.y() + p2_scr.y()) / 2)
            draw_textbox(f"📏 距离: {dist_str}", mid_scr.x(), mid_scr.y())

        if measure_line:
            p1_scr = self._to_screen_point(measure_line["p1"])
            p2_scr = self._to_screen_point(measure_line["p2"])
            painter.setPen(QPen(QColor("#00bcd4"), 2.0))
            painter.drawLine(p1_scr, p2_scr)
            draw_point(measure_line["p1"], "#00bcd4", 5); draw_point(measure_line["p2"], "#00bcd4", 5)
            
            mid_scr = QPointF((p1_scr.x() + p2_scr.x()) / 2, (p1_scr.y() + p2_scr.y()) / 2)
            box_text = f"📏 测量距离: {measure_line['dist_str']}\n🔸 起点: {measure_line['c1_str']}\n🔹 终点: {measure_line['c2_str']}"
            draw_textbox(box_text, mid_scr.x() + 10, mid_scr.y() + 10, QColor(0, 188, 212, 180))

        # 5. 渲染全部轨迹与标记点
        all_trails = state.get("all_trails", [])
        for i, trail in enumerate(all_trails):
            color_hex = TRACK_COLORS_VIEW[i % len(TRACK_COLORS_VIEW)]
            color = QColor(color_hex)
            color.setAlpha(150)
            painter.setPen(QPen(color, 1.5))
            for j in range(1, len(trail)):
                painter.drawLine(self._to_screen_point(trail[j-1]), self._to_screen_point(trail[j]))
            for pt in trail: draw_point(pt, color_hex, 2)

        current_marks = state.get("current_marks", [])
        for i, mark in enumerate(current_marks):
            color_hex = TRACK_COLORS_VIEW[i % len(TRACK_COLORS_VIEW)]
            p = draw_point(mark["pixel_pt"], color_hex, 5) 
            painter.setPen(QPen(QColor(color_hex)))
            painter.drawText(p + QPointF(10, -10), mark["text"])

        if mouse_pos and mode not in ["MEASURE_1", "MEASURE_2"]:
            m = self._to_screen_point(mouse_pos)
            if widget_rect.contains(m.toPoint()):
                painter.setPen(QPen(QColor(255, 255, 255, 175), 1))
                painter.drawLine(QPointF(m.x(), 0), QPointF(m.x(), widget_rect.height()))
                painter.drawLine(QPointF(0, m.y()), QPointF(widget_rect.width(), m.y()))

    def wheelEvent(self, event: QWheelEvent):
        old_coords = self._get_canvas_coords(event.position().toPoint())
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_zoom = self.zoom_level * zoom_factor
        if 0.01 <= new_zoom <= 40.0:
            self.zoom_level = new_zoom
            if old_coords:
                cx, cy = old_coords
                lw, lh = self.width(), self.height()
                self.pan_offset_x = event.position().x() - (lw / 2 + (cx - self.canvas_w / 2) * self.zoom_level)
                self.pan_offset_y = event.position().y() - (lh / 2 + (cy - self.canvas_h / 2) * self.zoom_level)
            self.zoom_changed.emit(int(self.zoom_level * 100))
            self.update()
            self.moved_pos.emit(-1.0, -1.0) 

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
            self.update()
            self.moved_pos.emit(-1.0, -1.0)
        else:
            coords = self._get_canvas_coords(event.position().toPoint())
            if coords: self.moved_pos.emit(*coords)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高速视频运动分析工具 V5.1 (全面修复)")
        self.resize(1600, 900)

        self.cap = None
        self.video_path = "" 
        self.total_frames = 0
        self.fps = 30.0
        self.video_w = 0
        self.video_h = 0
        self.current_frame_idx = -1 
        self.decoder_next_frame_idx = -1
        self.base_frame = None       
        self.base_qimage = None
        self.adjusted_qimage = None
        self.frame_cache = OrderedDict()
        self.max_cached_frames = 24
        self.is_slider_dragging = False
        self.is_seeking = False

        self.state = "IDLE" 
        self.calib_pt1 = None
        self.calib_line = None
        self.show_calib_line = True
        self.scale_factor = 1.0 
        self.physics_unit = "单位"
        self.grid_size = 10.0   
        
        self.coord_origin = None   
        self.coord_angle = 0.0 
        
        self.measure_pt1 = None
        self.measure_line = None
        
        self.mouse_curr_pos = None 
        self.brightness = 0
        self.contrast = 1.0
        
        self.render_scale = 1.0 
        self.points_per_frame = 1 

        self.aux_lines = []
        self.aux_current_points = []
        self.aux_intersections = []
        self.aux_midpoints = []

        self.marks = {} 
        self.first_time_sec = None 
        
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        
        self.curves_x = []
        self.curves_y = []
        self.highlights_x = []
        self.highlights_y = []

        self.init_ui()

    def _custom_icon(self, icon_type):
        icon = QIcon()
        for mode, color_hex in [(QIcon.Mode.Normal, "#edf2f8"), (QIcon.Mode.Active, "#ffffff"), (QIcon.Mode.Disabled, "#687484")]:
            pix = QPixmap(32, 32)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = QColor(color_hex)
            pen = QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            
            if icon_type == "load":
                painter.drawRoundedRect(4, 12, 24, 14, 2, 2)
                painter.drawLine(4, 12, 12, 12)
                painter.drawLine(12, 12, 16, 6)
                painter.drawLine(16, 6, 28, 6)
                painter.drawLine(28, 6, 28, 12)
            elif icon_type == "calib":
                painter.translate(16, 16)
                painter.rotate(-45)
                painter.drawRoundedRect(-14, -6, 28, 12, 2, 2)
                painter.drawLine(-8, -6, -8, -2)
                painter.drawLine(-2, -6, -2, -2)
                painter.drawLine(4, -6, 4, -2)
                painter.drawLine(10, -6, 10, -2)
            elif icon_type == "origin":
                painter.drawLine(6, 6, 6, 26)
                painter.drawLine(6, 26, 26, 26)
                painter.drawLine(2, 10, 6, 6)
                painter.drawLine(10, 10, 6, 6)
                painter.drawLine(22, 22, 26, 26)
                painter.drawLine(22, 30, 26, 26)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(6, 26), 2.0, 2.0)
            elif icon_type == "measure":
                # 修复4：彻底重绘测量图标，改为“双箭头尺寸标注线”，不再是倾斜卡尺
                painter.drawLine(4, 16, 28, 16)
                painter.drawLine(4, 16, 10, 10)
                painter.drawLine(4, 16, 10, 22)
                painter.drawLine(28, 16, 22, 10)
                painter.drawLine(28, 16, 22, 22)
                painter.drawLine(16, 12, 16, 20)
                painter.drawLine(12, 14, 12, 18)
                painter.drawLine(20, 14, 20, 18)
            elif icon_type == "mark":
                painter.drawEllipse(8, 8, 16, 16)
                painter.drawLine(16, 2, 16, 6)
                painter.drawLine(16, 26, 16, 30)
                painter.drawLine(2, 16, 6, 16)
                painter.drawLine(26, 16, 30, 16)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(16, 16), 2.0, 2.0)
            elif icon_type == "aux":
                painter.drawLine(6, 26, 14, 8)
                painter.drawLine(26, 26, 18, 8)
                painter.drawLine(6, 26, 26, 26)
                painter.drawLine(14, 8, 18, 8)
                painter.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.drawLine(6, 8, 26, 26)
                painter.drawLine(26, 8, 6, 26)
            elif icon_type == "clear":
                painter.drawRoundedRect(8, 10, 16, 18, 2, 2)
                painter.drawLine(4, 10, 28, 10)
                painter.drawLine(12, 10, 12, 6)
                painter.drawLine(12, 6, 20, 6)
                painter.drawLine(20, 6, 20, 10)
                painter.drawLine(12, 14, 12, 24)
                painter.drawLine(16, 14, 16, 24)
                painter.drawLine(20, 14, 20, 24)
            elif icon_type == "export":
                painter.drawLine(16, 4, 16, 20)
                painter.drawLine(10, 14, 16, 20)
                painter.drawLine(22, 14, 16, 20)
                painter.drawLine(6, 24, 26, 24)
                painter.drawLine(6, 28, 26, 28)
            elif icon_type == "import":
                painter.drawLine(16, 20, 16, 4)
                painter.drawLine(10, 10, 16, 4)
                painter.drawLine(22, 10, 16, 4)
                painter.drawLine(6, 24, 26, 24)
                painter.drawLine(6, 28, 26, 28)
            elif icon_type == "frame":
                painter.drawRoundedRect(4, 6, 24, 20, 2, 2)
                painter.drawEllipse(8, 10, 3, 3)
                painter.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.drawLine(4, 20, 12, 14)
                painter.drawLine(12, 14, 18, 20)
                painter.drawLine(16, 18, 20, 14)
                painter.drawLine(20, 14, 28, 22)
            elif icon_type == "reset":
                painter.drawRoundedRect(4, 4, 24, 24, 2, 2)
                painter.drawLine(16, 16, 8, 8)
                painter.drawLine(8, 8, 14, 8)
                painter.drawLine(8, 8, 8, 14)
                painter.drawLine(16, 16, 24, 24)
                painter.drawLine(24, 24, 18, 24)
                painter.drawLine(24, 24, 24, 18)
            elif icon_type == "trail":
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(7, 16), 2.0, 2.0)
                painter.drawEllipse(QPointF(16, 16), 3.0, 3.0)
                painter.drawEllipse(QPointF(25, 16), 4.0, 4.0)
                painter.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.drawLine(7, 16, 25, 16)
            elif icon_type == "play":
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(QPolygonF([QPointF(10, 6), QPointF(24, 16), QPointF(10, 26)]))
            elif icon_type == "pause":
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(8, 8, 5, 16)
                painter.drawRect(19, 8, 5, 16)
            elif icon_type == "prev":
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(QPolygonF([QPointF(20, 8), QPointF(10, 16), QPointF(20, 24)]))
                painter.drawRect(6, 8, 3, 16)
            elif icon_type == "next":
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(QPolygonF([QPointF(12, 8), QPointF(22, 16), QPointF(12, 24)]))
                painter.drawRect(23, 8, 3, 16)
            painter.end()
            icon.addPixmap(pix, mode)
        return icon

    def create_actions(self):
        self.act_load = QAction(self._custom_icon("load"), "加载视频", self)
        self.act_load.setShortcut(QKeySequence.StandardKey.Open)
        self.act_load.triggered.connect(self.load_video)

        self.act_export = QAction(self._custom_icon("export"), "导出 Excel", self)
        self.act_export.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.act_export.triggered.connect(self.export_to_excel)

        self.act_export_frame = QAction(self._custom_icon("frame"), "导出当前帧", self)
        self.act_export_frame.triggered.connect(self.export_current_frame)

        self.act_import = QAction(self._custom_icon("import"), "导入 Excel", self)
        self.act_import.setShortcut(QKeySequence.StandardKey.Open)
        self.act_import.triggered.connect(self.import_from_excel)

        self.act_calib = QAction(self._custom_icon("calib"), "标定尺寸", self)
        self.act_calib.triggered.connect(lambda: self.set_state("CALIBRATING_1"))

        self.act_coord = QAction(self._custom_icon("origin"), "设置坐标", self)
        self.act_coord.triggered.connect(lambda: self.set_state("SET_COORD_1"))
        
        self.act_measure = QAction(self._custom_icon("measure"), "测量查询", self)
        self.act_measure.triggered.connect(lambda: self.set_state("MEASURE_1"))

        self.act_mark = QAction(self._custom_icon("mark"), "开始捕捉", self)
        self.act_mark.triggered.connect(self.start_marking_mode)
        
        self.act_aux_line = QAction(self._custom_icon("aux"), "透视辅助线", self)
        self.act_aux_line.triggered.connect(lambda: self.set_state("DRAW_AUX"))
        
        self.act_clear_aux = QAction(self._custom_icon("clear"), "清空辅助线", self)
        self.act_clear_aux.triggered.connect(self.clear_aux_lines)

        self.act_reset_view = QAction(self._custom_icon("reset"), "画面重置", self)
        self.act_reset_view.triggered.connect(self.reset_label_view)

        self.act_exit = QAction("退出", self)
        self.act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_exit.triggered.connect(self.close)

        self.video_actions = [
            self.act_calib, self.act_coord, self.act_measure, self.act_mark,
            self.act_aux_line, self.act_clear_aux,
            self.act_export, self.act_import,
            self.act_export_frame, self.act_reset_view,
        ]
        for action in self.video_actions:
            action.setEnabled(False)

    def create_menus(self):
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(self.act_load)
        file_menu.addAction(self.act_export)
        file_menu.addAction(self.act_import)
        file_menu.addAction(self.act_export_frame)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        track_menu = self.menuBar().addMenu("分析")
        track_menu.addAction(self.act_calib)
        track_menu.addAction(self.act_coord)
        track_menu.addAction(self.act_measure)
        track_menu.addAction(self.act_mark)
        
        tool_menu = self.menuBar().addMenu("工具")
        tool_menu.addAction(self.act_aux_line)
        tool_menu.addAction(self.act_clear_aux)

        view_menu = self.menuBar().addMenu("视图")
        view_menu.addAction(self.act_reset_view)

    def create_toolbars(self):
        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        
        toolbar.addAction(self.act_load)
        toolbar.addSeparator()
        
        toolbar.addAction(self.act_calib)
        toolbar.addAction(self.act_coord)
        toolbar.addAction(self.act_aux_line)
        toolbar.addAction(self.act_clear_aux)
        toolbar.addSeparator()
        
        toolbar.addAction(self.act_measure)
        toolbar.addSeparator()
        
        toolbar.addAction(self.act_mark)
        
        self.btn_trail = QToolButton(self)
        self.btn_trail.setIcon(self._custom_icon("trail"))
        self.btn_trail.setText("轨迹模式")
        self.btn_trail.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.btn_trail.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_trail.setEnabled(False)

        trail_menu = QMenu(self)
        self.trail_actions = QActionGroup(self)
        self.trail_mode = 1 
        self.act_trail_none = QAction("无轨迹", self, checkable=True)
        self.act_trail_local = QAction("局部轨迹 (近10点)", self, checkable=True)
        self.act_trail_all = QAction("全部轨迹", self, checkable=True)
        self.act_trail_local.setChecked(True)
        self.trail_actions.addAction(self.act_trail_none)
        self.trail_actions.addAction(self.act_trail_local)
        self.trail_actions.addAction(self.act_trail_all)
        trail_menu.addAction(self.act_trail_none)
        trail_menu.addAction(self.act_trail_local)
        trail_menu.addAction(self.act_trail_all)
        self.btn_trail.setMenu(trail_menu)

        def on_trail_changed(action):
            if action == self.act_trail_none: self.trail_mode = 0
            elif action == self.act_trail_local: self.trail_mode = 1
            elif action == self.act_trail_all: self.trail_mode = 2
            self.render_frame()

        self.trail_actions.triggered.connect(on_trail_changed)
        toolbar.addWidget(self.btn_trail)
        toolbar.addSeparator()
        
        toolbar.addAction(self.act_import)
        toolbar.addAction(self.act_export)
        toolbar.addAction(self.act_export_frame)
        toolbar.addSeparator()
        
        toolbar.addAction(self.act_reset_view)
        
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def start_marking_mode(self):
        self.measure_line = None 
        pts, ok = QInputDialog.getInt(self, "开启捕捉模式", "请输入每帧需要捕捉的目标点数：", self.points_per_frame, 1, 99, 1)
        if ok:
            if pts != self.points_per_frame:
                if self.marks and len(self.marks) > 0:
                    reply = QMessageBox.question(self, "修改追踪点数", "更改追踪点数会重置现有的动态图表。确定修改吗？",
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.No:
                        return
                self.points_per_frame = pts
                self._init_plot_curves(self.points_per_frame)
                self.update_plots()
            self.set_state("MARKING")

    def _init_plot_curves(self, num_points):
        self.plot_x.clear()
        self.plot_y.clear()
        self.curves_x.clear()
        self.curves_y.clear()
        self.highlights_x.clear()
        self.highlights_y.clear()
        
        for i in range(num_points):
            # 修复3：分离X轴和Y轴的调色板，确保无论多少点，X图和Y图的曲线颜色互不干扰
            c_hex_x = TRACK_COLORS_X[i % len(TRACK_COLORS_X)]
            c_hex_y = TRACK_COLORS_Y[i % len(TRACK_COLORS_Y)]
            
            pen_x = pg.mkPen(c_hex_x, width=2)
            brush_x = pg.mkBrush(c_hex_x)
            
            pen_y = pg.mkPen(c_hex_y, width=2)
            brush_y = pg.mkBrush(c_hex_y)
            
            c_x = self.plot_x.plot(pen=pen_x, symbol='o', symbolSize=5, symbolBrush=brush_x)
            c_y = self.plot_y.plot(pen=pen_y, symbol='o', symbolSize=5, symbolBrush=brush_y)
            c_x.setClipToView(True)
            c_y.setClipToView(True)
            c_x.sigPointsClicked.connect(self.on_plot_point_clicked)
            c_y.sigPointsClicked.connect(self.on_plot_point_clicked)
            
            hl_pen = pg.mkPen('#ffffff', width=1.0)
            hl_brush = pg.mkBrush('#ffffff')
            hl_x = self.plot_x.plot([], [], pen=None, symbol='star', symbolSize=10, symbolBrush=hl_brush, symbolPen=hl_pen)
            hl_y = self.plot_y.plot([], [], pen=None, symbol='star', symbolSize=10, symbolBrush=hl_brush, symbolPen=hl_pen)
            hl_x.setZValue(10)
            hl_y.setZValue(10)
            
            self.curves_x.append(c_x)
            self.curves_y.append(c_y)
            self.highlights_x.append(hl_x)
            self.highlights_y.append(hl_y)

    def init_ui(self):
        self.create_actions()
        self.create_menus()
        self.create_toolbars()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(0)
        self.video_widget = VideoWidget() 
        self.video_widget.clicked_pos.connect(self.on_video_clicked)
        self.video_widget.moved_pos.connect(self.on_video_moved)
        left_layout.addWidget(self.video_widget, stretch=1) 

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        self.plot_x = pg.PlotWidget(title="X轴位移 (依自定义坐标系)")
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)

        self.plot_y = pg.PlotWidget(title="Y轴位移 (依自定义坐标系)")
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        
        self._init_plot_curves(self.points_per_frame)
        self.update_chart_labels()
        
        right_layout.addWidget(self.plot_x)
        right_layout.addWidget(self.plot_y)

        content_layout.addLayout(left_layout, stretch=2)
        content_layout.addLayout(right_layout, stretch=1)
        root_layout.addLayout(content_layout, stretch=1)

        bottom_frame = QFrame()
        bottom_frame.setObjectName("controlBar")
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(10, 8, 10, 8)
        bottom_layout.setSpacing(8)

        self.btn_play = QPushButton("播放")
        self.btn_play.setIcon(self._custom_icon("play"))
        self.btn_prev = QPushButton("上一帧")
        self.btn_prev.setIcon(self._custom_icon("prev"))
        self.btn_next = QPushButton("下一帧")
        self.btn_next.setIcon(self._custom_icon("next"))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setTracking(False) 

        self.label_frame_text = QLabel("帧:")
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setEnabled(False)
        self.spin_frame.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_frame.setMinimumWidth(92)
        self.label_frame_total = QLabel("/ 0")
        self.label_step = QLabel("步长:")
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 100)
        self.spin_step.setValue(1)
        self.spin_step.setSuffix(" 帧")
        self.spin_step.setEnabled(False)

        self.label_render_scale = QLabel("画质(提速):")
        self.combo_render_quality = QComboBox()
        self.combo_render_quality.addItems(["原画质 (最准)", "1/2 缩放 (较快)", "1/4 缩放 (极速)"])
        self.combo_render_quality.setEnabled(False)
        self.combo_render_quality.currentIndexChanged.connect(self.on_render_quality_changed)

        self.label_brightness = QLabel("亮度:")
        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setFixedWidth(110)
        self.slider_brightness.setEnabled(False)

        self.label_contrast = QLabel("对比度:")
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(50, 200)
        self.slider_contrast.setValue(100)
        self.slider_contrast.setFixedWidth(110)
        self.slider_contrast.setEnabled(False)

        self.chk_keep_calib = QCheckBox("显示标尺")
        self.chk_keep_calib.setChecked(True)
        self.chk_keep_calib.setEnabled(False)

        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_prev.clicked.connect(lambda: self.seek_frame(-self.spin_step.value()))
        self.btn_next.clicked.connect(lambda: self.seek_frame(self.spin_step.value()))
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.spin_frame.editingFinished.connect(self.on_frame_spin_finished)
        self.slider_brightness.valueChanged.connect(self.on_adjustment_changed)
        self.slider_contrast.valueChanged.connect(self.on_adjustment_changed)
        self.chk_keep_calib.toggled.connect(self.on_keep_calib_toggled)

        for widget in [self.btn_play, self.btn_prev, self.btn_next]: widget.setEnabled(False)

        bottom_layout.addWidget(self.btn_play)
        bottom_layout.addWidget(self.btn_prev)
        bottom_layout.addWidget(self.btn_next)
        bottom_layout.addWidget(self.slider, stretch=1)
        bottom_layout.addWidget(self.label_frame_text)
        bottom_layout.addWidget(self.spin_frame)
        bottom_layout.addWidget(self.label_frame_total)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.label_step)
        bottom_layout.addWidget(self.spin_step)
        bottom_layout.addWidget(self.label_render_scale)
        bottom_layout.addWidget(self.combo_render_quality)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.label_brightness)
        bottom_layout.addWidget(self.slider_brightness)
        bottom_layout.addWidget(self.label_contrast)
        bottom_layout.addWidget(self.slider_contrast)
        bottom_layout.addWidget(self.chk_keep_calib)
        root_layout.addWidget(bottom_frame)

        status_frame = QFrame()
        status_frame.setObjectName("statusBar")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel("状态: 等待加载视频...")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)
        root_layout.addWidget(status_frame)

    def recalc_all_marks(self):
        if not self.coord_origin: return
        ox, oy = self.coord_origin
        theta = self.coord_angle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        for frame_idx, pts_list in self.marks.items():
            for pt in pts_list:
                px, py = pt['pixel_pt']
                dx = px - ox
                dy = oy - py
                
                rel_x = (dx * cos_t + dy * sin_t) * self.scale_factor
                rel_y = (-dx * sin_t + dy * cos_t) * self.scale_factor
                pt['rel_x'] = rel_x
                pt['rel_y'] = rel_y

        self.update_plots()
        self.auto_range_plots()

    def on_render_quality_changed(self, idx):
        if idx == 0: self.render_scale = 1.0
        elif idx == 1: self.render_scale = 0.5
        elif idx == 2: self.render_scale = 0.25
        if self.cap and self.current_frame_idx >= 0:
            self.frame_cache.clear() 
            self.seek_frame(self.current_frame_idx, absolute=True) 

    def on_plot_point_clicked(self, item, points, *args):
        if not points: return
        if self.play_timer.isActive(): self.toggle_play()
        pt = points[0]
        x_val = pt.pos().x() 
        best_frame = None
        min_diff = float('inf')
        for k, d_list in self.marks.items():
            if not d_list: continue
            curr_x = d_list[0]['time'] - self.first_time_sec if self.first_time_sec is not None else k
            diff = abs(curr_x - x_val)
            if diff < min_diff:
                min_diff = diff
                best_frame = k
        if best_frame is not None and min_diff < 1.0: 
            self.seek_frame(best_frame, absolute=True)

    def set_state(self, state):
        self.state = state
        pts = self.points_per_frame
        curr_len = len(self.marks.get(self.current_frame_idx, [])) if self.current_frame_idx in self.marks else 0
        if curr_len >= pts: curr_len = 0 
        
        msgs = {
            "IDLE": "空闲。滚轮缩放，右键按住拖动画面，左下角可点击播放/暂停。",
            "CALIBRATING_1": "【尺寸标定】: 请点击定标尺的【起点】 (自动吸附辅助交点/中点)",
            "CALIBRATING_2": "【尺寸标定】: 请点击定标尺的【终点】 (自动吸附辅助交点/中点)",
            "SET_COORD_1": "【建立坐标系】: 请点击设定【坐标原点 (0,0)】 (自动吸附辅助点)",
            "SET_COORD_2": "【建立坐标系】: 请点击画面确定【X轴正方向】(随后会弹窗供你微调角度)",
            "MEASURE_1": "【测量/查询】: 悬停查看坐标。请点击起点测量距离 (标定后显示真实物理尺寸)。",
            "MEASURE_2": "【测量/查询】: 请点击终点以固定测量线段。",
            "MARKING": f"【捕捉模式】: 设置为每帧 {pts} 个点。请点击目标（当前第 {curr_len+1}/{pts} 个点）",
            "DRAW_AUX": "【辅助线模式】: 请画出透视视场下的 4 条轮廓线段，将利用交比自动计算视场中心和透视中点。"
        }
        self.status_label.setText(f"状态: {msgs.get(state, state)}")
        if state != "DRAW_AUX":
            self.aux_current_points.clear()
        if state not in ["MEASURE_1", "MEASURE_2", "IDLE"]:
            self.measure_line = None 
        self.render_frame()

    def clear_aux_lines(self):
        self.aux_lines.clear()
        self.aux_current_points.clear()
        self.aux_intersections.clear()
        self.aux_midpoints.clear()
        self.render_frame()

    def _get_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = p3; x4, y4 = p4
        den = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(den) < 1e-5: return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
        return (px, py)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setIcon(self._custom_icon("play"))
            self.btn_play.setText("播放")
        else:
            self.play_timer.start(self._playback_interval_ms())
            self.btn_play.setIcon(self._custom_icon("pause"))
            self.btn_play.setText("暂停")

    def _playback_interval_ms(self):
        step_size = max(1, self.spin_step.value())
        return max(16, int(1000 * step_size / max(self.fps, 1.0)))

    def play_next_frame(self):
        if self.is_seeking: return
        next_interval = self._playback_interval_ms()
        if self.play_timer.interval() != next_interval:
            self.play_timer.setInterval(next_interval)
        step_size = self.spin_step.value()
        if self.current_frame_idx + step_size < self.total_frames:
            self.seek_frame(step_size)
        else:
            self.seek_frame(self.total_frames - 1, absolute=True)
            self.play_timer.stop()
            self.btn_play.setIcon(self._custom_icon("play"))
            self.btn_play.setText("播放")

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if not filename: return
        self.video_path = filename 

        if self.play_timer.isActive(): self.toggle_play()
        if self.cap: self.cap.release()
        
        self.cap = cv2.VideoCapture(filename)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_widget.canvas_w = self.video_w
        self.video_widget.canvas_h = self.video_h
        
        self.marks.clear()
        self.coord_origin = None
        self.coord_angle = 0.0
        self.calib_pt1 = None
        self.calib_line = None
        self.measure_pt1 = None
        self.measure_line = None
        self.clear_aux_lines()
        self.physics_unit = "单位"
        self.grid_size = 10.0
        self.current_frame_idx = -1 
        self.decoder_next_frame_idx = -1
        self.base_frame = None
        self.base_qimage = None
        self.adjusted_qimage = None
        self.frame_cache.clear()
        self.first_time_sec = None
        
        self.points_per_frame = 1 
        self._init_plot_curves(self.points_per_frame) 
        
        self.update_chart_labels()
        self.update_plots()
        
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.spin_frame.blockSignals(True)
        self.spin_frame.setRange(0, self.total_frames - 1)
        self.spin_frame.setValue(0)
        self.spin_frame.blockSignals(False)
        self.label_frame_total.setText(f"/ {self.total_frames - 1}")
        
        for widget in [self.slider, self.btn_play, self.btn_prev, self.btn_next, self.spin_step, self.combo_render_quality,
                       self.spin_frame, self.slider_brightness, self.slider_contrast, self.chk_keep_calib, self.btn_trail]:
            widget.setEnabled(True)
        for action in self.video_actions: action.setEnabled(True)
        
        self.reset_adjustments(update_frame=False)
        self.seek_frame(0, absolute=True)
        self.set_state("IDLE")
        self.video_widget.fit_to_view()

    def reset_label_view(self):
        self.video_widget.fit_to_view()
        self.reset_adjustments(update_frame=True)
        self.render_frame()

    def reset_adjustments(self, update_frame=True):
        self.brightness = 0
        self.contrast = 1.0
        for slider, value in [(self.slider_brightness, 0), (self.slider_contrast, 100)]:
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            
        self.combo_render_quality.blockSignals(True)
        self.combo_render_quality.setCurrentIndex(0)
        self.combo_render_quality.blockSignals(False)
        
        was_scaled = self.render_scale != 1.0
        self.render_scale = 1.0
            
        if update_frame and self.base_frame is not None:
            if was_scaled:
                self.frame_cache.clear()
                self.seek_frame(self.current_frame_idx, absolute=True)
            else:
                self._refresh_adjusted_frame()

    def seek_frame(self, val, absolute=False):
        if not self.cap: return
        if self.is_seeking: return
        
        target_idx = val if absolute else self.current_frame_idx + val
        target_idx = max(0, min(target_idx, self.total_frames - 1))
        if target_idx == self.current_frame_idx and self.base_qimage is not None: return

        if target_idx in self.frame_cache:
            self.base_frame, self.base_qimage = self.frame_cache.pop(target_idx)
            self.frame_cache[target_idx] = (self.base_frame, self.base_qimage)
            self._finish_seek(target_idx)
            return

        self.is_seeking = True
        try:
            diff = target_idx - self.current_frame_idx
            ret = False
            frame = None
            decoder_is_aligned = self.decoder_next_frame_idx == self.current_frame_idx + 1
            
            if diff == 1 and decoder_is_aligned:
                ret, frame = self.cap.read()
            elif 1 < diff <= 10 and decoder_is_aligned: 
                for _ in range(diff - 1): self.cap.grab()
                ret, frame = self.cap.read()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = self.cap.read()
                
            if ret: self._set_current_frame(target_idx, frame)
        finally:
            self.is_seeking = False

    def _set_current_frame(self, target_idx, frame):
        if self.render_scale != 1.0:
            new_w = int(self.video_w * self.render_scale)
            new_h = int(self.video_h * self.render_scale)
            self.base_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            self.base_frame = frame
            
        self.base_qimage = self._frame_to_qimage(self.base_frame)
        self.frame_cache[target_idx] = (self.base_frame, self.base_qimage)
        while len(self.frame_cache) > self.max_cached_frames:
            self.frame_cache.popitem(last=False)
        self.decoder_next_frame_idx = target_idx + 1
        self._finish_seek(target_idx)

    def _finish_seek(self, target_idx):
        self.current_frame_idx = target_idx
        self._refresh_adjusted_frame()
        self.slider.blockSignals(True)
        self.slider.setValue(target_idx)
        self.slider.blockSignals(False)
        self.spin_frame.blockSignals(True)
        self.spin_frame.setValue(target_idx)
        self.spin_frame.blockSignals(False)

        if self.state == "MARKING":
            pts = self.points_per_frame
            curr_len = len(self.marks.get(target_idx, []))
            if curr_len >= pts: curr_len = 0
            self.status_label.setText(f"状态: 【捕捉模式】 设置为每帧 {pts} 个点。请点击目标（当前第 {curr_len+1}/{pts} 个点）")

        self.render_frame()

    def _frame_to_qimage(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format.Format_RGB888).copy()

    def _refresh_adjusted_frame(self):
        if self.base_frame is None: return
        if self.brightness == 0 and abs(self.contrast - 1.0) < 0.001:
            self.adjusted_qimage = self.base_qimage
        else:
            adjusted = cv2.convertScaleAbs(self.base_frame, alpha=self.contrast, beta=self.brightness)
            self.adjusted_qimage = self._frame_to_qimage(adjusted)
        self.video_widget.set_frame(self.adjusted_qimage)

    def on_slider_pressed(self): self.is_slider_dragging = True

    def on_slider_value_changed(self, value):
        self.spin_frame.blockSignals(True)
        self.spin_frame.setValue(value)
        self.spin_frame.blockSignals(False)
        if not self.is_slider_dragging: self.seek_frame(value, absolute=True)

    def on_slider_released(self):
        self.is_slider_dragging = False
        self.seek_frame(self.slider.value(), absolute=True)

    def on_frame_spin_finished(self):
        if self.cap:
            val = self.spin_frame.value()
            self.slider.blockSignals(True)
            self.slider.setValue(val)
            self.slider.blockSignals(False)
            self.seek_frame(val, absolute=True)

    def on_adjustment_changed(self):
        self.brightness = self.slider_brightness.value()
        self.contrast = self.slider_contrast.value() / 100.0
        self._refresh_adjusted_frame()
        self.render_frame()

    def on_keep_calib_toggled(self, checked):
        self.show_calib_line = checked
        self.render_frame()

    def on_video_moved(self, x, y):
        if not (x == -1.0 and y == -1.0):
            snap_dist = 15.0 / self.video_widget.zoom_level
            min_d = float('inf')
            snap_pt = None
            
            all_snap_points = self.aux_intersections + self.aux_midpoints
            for pt in all_snap_points:
                d = math.hypot(x - pt[0], y - pt[1])
                if d < snap_dist and d < min_d:
                    min_d = d
                    snap_pt = pt
                        
            if snap_pt: self.mouse_curr_pos = snap_pt
            else: self.mouse_curr_pos = (x, y)
        self.render_frame()

    def render_frame(self):
        if self.base_qimage is None: return

        current_marks = []
        if self.current_frame_idx in self.marks:
            for d in self.marks[self.current_frame_idx]:
                rx, ry = d['rel_x'], d['rel_y']
                if self.first_time_sec is not None:
                    rel_time = d['time'] - self.first_time_sec
                    text = f"X:{rx:.2f} Y:{ry:.2f} | t:{rel_time:.3f}s"
                else:
                    text = f"X: {rx:.2f}, Y: {ry:.2f} {self.physics_unit}"
                current_marks.append({"pixel_pt": d["pixel_pt"], "text": text})

        video_info = None
        if self.video_path:
            filename = os.path.basename(self.video_path)
            duration = self.total_frames / max(self.fps, 1.0)
            video_info = {
                "name": filename,
                "path": self.video_path,
                "resolution": f"{self.video_w}×{self.video_h} (渲染: {int(self.video_w*self.render_scale)}×{int(self.video_h*self.render_scale)})",
                "fps": f"{self.fps:.2f}",
                "frames": self.total_frames,
                "duration": f"{duration:.2f}s"
            }

        trail_mode = self.trail_mode 
        sorted_keys = sorted(self.marks.keys())
        
        if trail_mode == 0: show_keys = []
        elif trail_mode == 1:
            past_keys = [k for k in sorted_keys if k <= self.current_frame_idx]
            show_keys = past_keys[-10:]
        else: show_keys = sorted_keys
            
        all_trails = [[] for _ in range(self.points_per_frame)]
        for k in show_keys:
            frame_points = self.marks[k]
            for i, pt_dict in enumerate(frame_points):
                if i < self.points_per_frame:
                    all_trails[i].append(pt_dict['pixel_pt'])

        self.video_widget.set_overlay_state(
            mode=self.state,
            calib_pt1=self.calib_pt1,
            calib_line=self.calib_line if self.show_calib_line else None,
            coord_origin=self.coord_origin,
            coord_angle=self.coord_angle,
            measure_pt1=self.measure_pt1,
            measure_line=self.measure_line,
            current_marks=current_marks,
            mouse_curr_pos=self.mouse_curr_pos,
            video_info=video_info,
            aux_lines=self.aux_lines,
            aux_current_points=self.aux_current_points,
            aux_intersections=self.aux_intersections,
            aux_midpoints=self.aux_midpoints,
            all_trails=all_trails,
            scale_factor=self.scale_factor,
            physics_unit=self.physics_unit,
            grid_size=self.grid_size
        )
        self.update_highlight()

    def get_formatted_coords(self, pt):
        is_calibrated = (self.calib_line is not None)
        if is_calibrated and self.coord_origin:
            dx = pt[0] - self.coord_origin[0]
            dy = self.coord_origin[1] - pt[1]
            theta = self.coord_angle
            rx = (dx * math.cos(theta) + dy * math.sin(theta)) * self.scale_factor
            ry = (-dx * math.sin(theta) + dy * math.cos(theta)) * self.scale_factor
            return f"{rx:.2f}, {ry:.2f} {self.physics_unit}"
        elif is_calibrated:
            return f"{pt[0]*self.scale_factor:.2f}, {pt[1]*self.scale_factor:.2f} {self.physics_unit}"
        else:
            return f"{int(pt[0])}, {int(pt[1])} px"

    def on_video_clicked(self, x, y):
        click_x, click_y = self.mouse_curr_pos if self.mouse_curr_pos else (x, y)

        if self.state == "DRAW_AUX":
            if len(self.aux_lines) == 4 and len(self.aux_current_points) == 0:
                self.clear_aux_lines()
            self.aux_current_points.append((click_x, click_y))
            if len(self.aux_current_points) == 2:
                A, B = self.aux_current_points
                self.aux_lines.append((A, B))
                self.aux_current_points.clear()
                
                if len(self.aux_lines) == 4:
                    M = [((l[0][0]+l[1][0])/2, (l[0][1]+l[1][1])/2) for l in self.aux_lines]
                    d = lambda p1, p2: math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    s1 = d(M[0], M[1]) + d(M[2], M[3]) 
                    s2 = d(M[0], M[2]) + d(M[1], M[3]) 
                    s3 = d(M[0], M[3]) + d(M[1], M[2]) 
                    
                    max_s = max(s1, s2, s3)
                    if max_s == s1: groupA, groupB = (0, 1), (2, 3)
                    elif max_s == s2: groupA, groupB = (0, 2), (1, 3)
                    else: groupA, groupB = (0, 3), (1, 2)
                    
                    self.aux_intersections.clear()
                    for i in groupA:
                        for j in groupB:
                            pt = self._get_intersection(*self.aux_lines[i], *self.aux_lines[j])
                            if pt: self.aux_intersections.append(pt)
                                
                    self.aux_midpoints.clear()
                    if len(self.aux_intersections) == 4:
                        corners = self.aux_intersections[:]
                        cx = sum(c[0] for c in corners) / 4.0
                        cy = sum(c[1] for c in corners) / 4.0
                        corners.sort(key=lambda c: math.atan2(c[1]-cy, c[0]-cx))
                        
                        self.aux_intersections = corners
                        C0, C1, C2, C3 = corners
                        O = self._get_intersection(C0, C2, C1, C3) 
                        
                        if O:
                            self.aux_midpoints.append(O) 
                            V1 = self._get_intersection(C0, C1, C2, C3) 
                            V2 = self._get_intersection(C1, C2, C3, C0) 
                            
                            def add_projective_midpoint(pA, pB, V):
                                if V:
                                    m = self._get_intersection(pA, pB, V, O)
                                    if m: self.aux_midpoints.append(m)
                                else:
                                    self.aux_midpoints.append(((pA[0]+pB[0])/2.0, (pA[1]+pB[1])/2.0))
                                    
                            add_projective_midpoint(C0, C1, V2)
                            add_projective_midpoint(C2, C3, V2)
                            add_projective_midpoint(C1, C2, V1)
                            add_projective_midpoint(C3, C0, V1)
            self.render_frame()

        elif self.state == "CALIBRATING_1":
            self.calib_pt1 = (click_x, click_y)
            self.set_state("CALIBRATING_2")
            
        elif self.state == "CALIBRATING_2":
            pixel_dist = math.hypot(click_x - self.calib_pt1[0], click_y - self.calib_pt1[1])
            if pixel_dist == 0: pixel_dist = 1.0
            
            real_dist, ok1 = QInputDialog.getDouble(self, "标定物理尺寸", f"像素跨度: {pixel_dist:.2f} px\n请输入实际物理长度:", 600.0, 0.001, 100000, 3)
            if ok1 and real_dist > 0:
                unit_str, ok2 = QInputDialog.getText(self, "设置单位名称", "请输入单位简称 (如 mm, m):", text="mm")
                if ok2 and unit_str.strip(): self.physics_unit = unit_str.strip()
                self.scale_factor = real_dist / pixel_dist
                self.calib_line = {
                    "p1": self.calib_pt1,
                    "p2": (click_x, click_y),
                    "text": f"{real_dist:.3f} {self.physics_unit}",
                }
                self.update_chart_labels()
                QMessageBox.information(self, "成功", f"标定映射建立！\n1 像素 = {self.scale_factor:.4f} {self.physics_unit}")
            self.calib_pt1 = None
            self.set_state("IDLE")
            
        elif self.state == "SET_COORD_1":
            self.coord_origin = (click_x, click_y)
            self.set_state("SET_COORD_2")
            
        elif self.state == "SET_COORD_2":
            dx = click_x - self.coord_origin[0]
            dy = self.coord_origin[1] - click_y 
            
            if math.hypot(dx, dy) < 3.0: 
                angle_deg = 0.0
            else: 
                angle_deg = math.degrees(math.atan2(dy, dx))
                
            final_deg, ok = QInputDialog.getDouble(
                self, "精调坐标轴角度", 
                "你已点击定义了X轴方向。\n在此微调逆时针旋转角度(度)或直接确认\n(0 为默认水平向右):", 
                angle_deg, -360, 360, 2
            )
            
            if ok:
                self.coord_angle = math.radians(final_deg)
                self.recalc_all_marks() 
                QMessageBox.information(self, "设置成功", 
                                        f"自定义坐标系已生效！\n"
                                        f"原点: ({self.coord_origin[0]:.1f}, {self.coord_origin[1]:.1f})\n"
                                        f"X轴倾角: {final_deg:.2f}°")
            self.set_state("IDLE")
            
        elif self.state == "MEASURE_1":
            self.measure_line = None
            self.measure_pt1 = (click_x, click_y)
            self.set_state("MEASURE_2")
            
        elif self.state == "MEASURE_2":
            p1 = self.measure_pt1
            p2 = (click_x, click_y)
            
            is_calibrated = (self.calib_line is not None)
            pixel_dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            if is_calibrated:
                dist_str = f"{pixel_dist * self.scale_factor:.2f} {self.physics_unit}"
            else:
                dist_str = f"{pixel_dist:.1f} px"
                
            self.measure_line = {
                'p1': p1, 'p2': p2, 
                'dist_str': dist_str,
                'c1_str': self.get_formatted_coords(p1),
                'c2_str': self.get_formatted_coords(p2)
            }
            self.measure_pt1 = None
            self.set_state("IDLE")
            
        elif self.state == "MARKING":
            if not self.coord_origin:
                QMessageBox.warning(self, "警告", "请先在上方工具栏点击【设置坐标】，在画面建立坐标系！")
                self.set_state("IDLE")
                return
                
            px, py = click_x, click_y
            ox, oy = self.coord_origin
            theta = self.coord_angle
            
            dx = px - ox
            dy = oy - py 
            
            rel_x = (dx * math.cos(theta) + dy * math.sin(theta)) * self.scale_factor
            rel_y = (-dx * math.sin(theta) + dy * math.cos(theta)) * self.scale_factor
            time_sec = self.current_frame_idx / self.fps
            
            if self.first_time_sec is None:
                self.first_time_sec = time_sec
                self.update_chart_labels() 

            frame_pts = self.marks.get(self.current_frame_idx, [])
            if len(frame_pts) >= self.points_per_frame: frame_pts = [] 
                
            frame_pts.append({
                'time': time_sec, 'rel_x': rel_x, 'rel_y': rel_y, 'pixel_pt': (click_x, click_y)
            })
            self.marks[self.current_frame_idx] = frame_pts

            self.update_plots()
            self.auto_range_plots()
            
            if len(frame_pts) >= self.points_per_frame:
                step_size = self.spin_step.value() 
                if self.current_frame_idx + step_size < self.total_frames:
                    self.seek_frame(step_size) 
                else:
                    self.render_frame()
            else:
                curr_len = len(frame_pts)
                self.status_label.setText(f"状态: 【捕捉模式】 设置为每帧 {self.points_per_frame} 个点。请点击目标（当前第 {curr_len+1}/{self.points_per_frame} 个点）")
                self.render_frame()

    def update_chart_labels(self):
        self.plot_x.getAxis('left').setLabel(f"X轴位移 ({self.physics_unit})", units=None)
        self.plot_y.getAxis('left').setLabel(f"Y轴位移 ({self.physics_unit})", units=None)
        
        if self.first_time_sec is not None:
            self.plot_x.getAxis('bottom').setLabel("相对时间", units='s')
            self.plot_y.getAxis('bottom').setLabel("相对时间", units='s')
        else:
            self.plot_x.getAxis('bottom').setLabel("时间/进度", units='帧数')
            self.plot_y.getAxis('bottom').setLabel("时间/进度", units='帧数')

    def update_plots(self):
        sorted_keys = sorted(self.marks.keys())
        if not sorted_keys: return
        
        if self.first_time_sec is not None:
            x_axis_data = [self.marks[k][0]['time'] - self.first_time_sec for k in sorted_keys if len(self.marks[k]) > 0]
        else:
            x_axis_data = [k for k in sorted_keys if len(self.marks[k]) > 0]
            
        for i in range(self.points_per_frame):
            if i >= len(self.curves_x): break 
            
            xs, ys, valid_x_axis = [], [], []
            for k, x_val in zip(sorted_keys, x_axis_data):
                if len(self.marks[k]) > i:
                    xs.append(self.marks[k][i]['rel_x'])
                    ys.append(self.marks[k][i]['rel_y'])
                    valid_x_axis.append(x_val)
                    
            self.curves_x[i].setData(valid_x_axis, xs)
            self.curves_y[i].setData(valid_x_axis, ys)

        self.update_highlight()

    def update_highlight(self):
        if self.current_frame_idx in self.marks:
            pts = self.marks[self.current_frame_idx]
            x_val = self.current_frame_idx
            if self.first_time_sec is not None and pts:
                x_val = pts[0]['time'] - self.first_time_sec
                
            for i in range(self.points_per_frame):
                if i < len(pts):
                    self.highlights_x[i].setData([x_val], [pts[i]['rel_x']])
                    self.highlights_y[i].setData([x_val], [pts[i]['rel_y']])
                else:
                    self.highlights_x[i].setData([], [])
                    self.highlights_y[i].setData([], [])
        else:
            for i in range(self.points_per_frame):
                if i < len(self.highlights_x):
                    self.highlights_x[i].setData([], [])
                    self.highlights_y[i].setData([], [])

    def auto_range_plots(self):
        self.plot_x.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_y.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_x.autoRange()
        self.plot_y.autoRange()

    def export_current_frame(self):
        if not self.cap or self.current_frame_idx < 0:
            QMessageBox.warning(self, "无法导出", "请先加载视频并定位到需要导出的帧。")
            return
        if self.play_timer.isActive(): self.toggle_play()

        default_name = f"frame_{self.current_frame_idx:06d}.png"
        filename, _ = QFileDialog.getSaveFileName(self, "导出当前帧原始画面", default_name, "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap Image (*.bmp)")
        if not filename: return
        
        lower_name = filename.lower()
        if not lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp")): filename += ".png"

        old_idx = self.current_frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, old_idx)
        ret, raw_frame = self.cap.read()
        self.decoder_next_frame_idx = old_idx + 1
        if not ret:
            QMessageBox.critical(self, "导出失败", "无法从视频中读取当前帧。")
            return
        if cv2.imwrite(filename, raw_frame): QMessageBox.information(self, "成功", f"当前帧原始画面已导出：\n{filename}")
        else: QMessageBox.critical(self, "导出失败", "写入图像文件失败，请检查文件路径或格式。")

    def export_to_excel(self):
        if not self.marks: return
        if self.play_timer.isActive(): self.toggle_play() 
        filename, _ = QFileDialog.getSaveFileName(self, "数据存档为 Excel", "_tracked_data.xlsx", "Excel Files (*.xlsx)")
        if filename:
            try:
                sorted_keys = sorted(self.marks.keys())
                data_list = []
                for k in sorted_keys:
                    pts = self.marks[k]
                    if not pts: continue
                    
                    rel_time_val = pts[0]['time'] - self.first_time_sec if self.first_time_sec is not None else 0.0
                    row_data = {
                        'Frame_Index (绝对帧数)': k, 
                        'Absolute_Time (绝对时间-秒)': pts[0]['time'],
                        'Relative_Time (相对时间-秒/零点平移)': rel_time_val
                    }
                    
                    for i, pt in enumerate(pts):
                        row_data[f'Physical_X_P{i+1} ({self.physics_unit})'] = pt['rel_x']
                        row_data[f'Physical_Y_P{i+1} ({self.physics_unit})'] = pt['rel_y']
                        row_data[f'Canvas_Pixel_X_P{i+1}'] = pt['pixel_pt'][0]
                        row_data[f'Canvas_Pixel_Y_P{i+1}'] = pt['pixel_pt'][1]
                        
                    data_list.append(row_data)
                df = pd.DataFrame(data_list)
                
                metadata = {
                    'Parameter': [
                        'points_per_frame', 'scale_factor', 'physics_unit', 'grid_size',
                        'coord_origin_x', 'coord_origin_y', 'coord_angle', 'first_time_sec',
                        'calib_p1_x', 'calib_p1_y', 'calib_p2_x', 'calib_p2_y', 'calib_text',
                        'video_path', 'video_w', 'video_h', 'fps', 'total_frames'
                    ],
                    'Value': [
                        self.points_per_frame, self.scale_factor, self.physics_unit, self.grid_size,
                        self.coord_origin[0] if self.coord_origin else None,
                        self.coord_origin[1] if self.coord_origin else None,
                        self.coord_angle,
                        self.first_time_sec,
                        self.calib_line['p1'][0] if self.calib_line else None,
                        self.calib_line['p1'][1] if self.calib_line else None,
                        self.calib_line['p2'][0] if self.calib_line else None,
                        self.calib_line['p2'][1] if self.calib_line else None,
                        self.calib_line['text'] if self.calib_line else None,
                        self.video_path, self.video_w, self.video_h, self.fps, self.total_frames
                    ]
                }
                df_meta = pd.DataFrame(metadata)
                
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='TrackData', index=False)
                    df_meta.to_excel(writer, sheet_name='Metadata', index=False)
                QMessageBox.information(self, "成功", f"报告成功写入到本地：\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))
    
    def import_from_excel(self):
        filename, _ = QFileDialog.getOpenFileName(self, "导入 Excel 捕捉数据", "", "Excel Files (*.xlsx)")
        if not filename: return
        if self.play_timer.isActive(): self.toggle_play()

        try:
            xls = pd.ExcelFile(filename)
            sheet_names = xls.sheet_names

            if 'Metadata' in sheet_names:
                df_meta = pd.read_excel(filename, sheet_name='Metadata')
                meta_dict = dict(zip(df_meta['Parameter'], df_meta['Value']))
                
                if 'points_per_frame' in meta_dict and pd.notna(meta_dict['points_per_frame']):
                    self.points_per_frame = int(meta_dict['points_per_frame'])
                    self._init_plot_curves(self.points_per_frame)
                if 'scale_factor' in meta_dict and pd.notna(meta_dict['scale_factor']):
                    self.scale_factor = float(meta_dict['scale_factor'])
                if 'physics_unit' in meta_dict and pd.notna(meta_dict['physics_unit']):
                    self.physics_unit = str(meta_dict['physics_unit'])
                if 'grid_size' in meta_dict and pd.notna(meta_dict['grid_size']):
                    self.grid_size = float(meta_dict['grid_size'])
                
                ox = meta_dict.get('coord_origin_x', meta_dict.get('origin_x'))
                oy = meta_dict.get('coord_origin_y', meta_dict.get('origin_y'))
                if pd.notna(ox) and pd.notna(oy):
                    self.coord_origin = (float(ox), float(oy))
                    
                c_angle = meta_dict.get('coord_angle')
                if pd.notna(c_angle):
                    self.coord_angle = float(c_angle)
                    
                if 'first_time_sec' in meta_dict and pd.notna(meta_dict['first_time_sec']):
                    self.first_time_sec = float(meta_dict['first_time_sec'])

                calib_keys = ('calib_p1_x', 'calib_p1_y', 'calib_p2_x', 'calib_p2_y')
                if all(k in meta_dict for k in calib_keys):
                    vals = [meta_dict[k] for k in calib_keys]
                    if all(pd.notna(v) for v in vals):
                        calib_text = meta_dict.get('calib_text', None)
                        if pd.notna(calib_text):
                            self.calib_line = {
                                'p1': (float(vals[0]), float(vals[1])),
                                'p2': (float(vals[2]), float(vals[3])),
                                'text': str(calib_text)
                            }

                imported_video = meta_dict.get('video_path', None)
                if imported_video and pd.notna(imported_video) and self.video_path:
                    imported_video = str(imported_video)
                    if imported_video != self.video_path:
                        reply = QMessageBox.question(
                            self, "视频不匹配",
                            f"导入数据来自:\n{imported_video}\n\n当前加载的是:\n{self.video_path}\n\n仍要恢复标定参数并导入数据吗？",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        if reply != QMessageBox.StandardButton.Yes: return

            data_sheet = 'TrackData' if 'TrackData' in sheet_names else [s for s in sheet_names if s != 'Metadata'][0]
            df = pd.read_excel(filename, sheet_name=data_sheet)

            phys_x_cols = [col for col in df.columns if 'physical_x' in col.lower()]
            num_points_in_file = len(phys_x_cols) if phys_x_cols else 1
            if num_points_in_file != self.points_per_frame:
                self.points_per_frame = num_points_in_file
                self._init_plot_curves(self.points_per_frame)

            frame_col = time_col = rel_time_col = None
            for col in df.columns:
                cl = col.lower()
                if frame_col is None and 'frame' in cl: frame_col = col
                elif time_col is None and 'absolute_time' in cl: time_col = col
                elif rel_time_col is None and 'relative_time' in cl: rel_time_col = col

            if frame_col is None:
                QMessageBox.critical(self, "导入失败", "无法识别帧索引列 (Frame_Index)")
                return

            self.marks.clear()
            skipped = 0
            for _, row in df.iterrows():
                frame_idx = int(row[frame_col])
                if self.total_frames > 0 and not (0 <= frame_idx < self.total_frames):
                    skipped += 1
                    continue

                t = float(row[time_col]) if time_col is not None and pd.notna(row.get(time_col)) else (frame_idx / max(self.fps, 1.0))
                pts_list = []
                for i in range(self.points_per_frame):
                    px_c = next((c for c in df.columns if f'physical_x_p{i+1}' in c.lower() or (i==0 and c.lower() == 'physical_x (mm)')), None)
                    py_c = next((c for c in df.columns if f'physical_y_p{i+1}' in c.lower() or (i==0 and c.lower() == 'physical_y (mm)')), None)
                    pix_x_c = next((c for c in df.columns if f'canvas_pixel_x_p{i+1}' in c.lower() or (i==0 and c.lower() == 'canvas_pixel_x')), None)
                    pix_y_c = next((c for c in df.columns if f'canvas_pixel_y_p{i+1}' in c.lower() or (i==0 and c.lower() == 'canvas_pixel_y')), None)
                    
                    rx = float(row[px_c]) if px_c is not None and pd.notna(row.get(px_c)) else 0.0
                    ry = float(row[py_c]) if py_c is not None and pd.notna(row.get(py_c)) else 0.0
                    pix_x = float(row[pix_x_c]) if pix_x_c is not None and pd.notna(row.get(pix_x_c)) else 0.0
                    pix_y = float(row[pix_y_c]) if pix_y_c is not None and pd.notna(row.get(pix_y_c)) else 0.0
                    
                    pts_list.append({ 'time': t, 'rel_x': rx, 'rel_y': ry, 'pixel_pt': (pix_x, pix_y) })

                self.marks[frame_idx] = pts_list

            if self.first_time_sec is None and self.marks and rel_time_col is not None:
                first_key = min(self.marks.keys())
                rel_t = float(df.loc[df[frame_col] == first_key, rel_time_col].iloc[0])
                if pd.notna(rel_t):
                    self.first_time_sec = self.marks[first_key][0]['time'] - rel_t

            self.update_chart_labels()
            self.update_plots()
            self.auto_range_plots()
            self.render_frame()

            msg = f"成功导入 {len(self.marks)} 帧记录 (每帧 {self.points_per_frame} 点)"
            if skipped > 0: msg += f"\n（跳过了 {skipped} 条超出当前视频帧范围的记录）"
            QMessageBox.information(self, "导入成功", msg)

        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"读取 Excel 文件出错：\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())