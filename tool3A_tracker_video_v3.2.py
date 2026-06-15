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
                             QFrame, QSizePolicy, QToolBar, QCheckBox, QStyle)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRectF, QPointF, QSize
from PyQt6.QtGui import (QImage, QWheelEvent, QMouseEvent, QPaintEvent, QPainter, QColor, QPen, QFont,
                         QAction, QKeySequence, QIcon, QPixmap)

cv2.setUseOptimized(True)
pg.setConfigOptions(background="#101418", foreground="#d7dde8", antialias=True)


APP_STYLE = """
QMainWindow, QWidget {
    background: #0f141b;
    color: #d7dde8;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 10pt;
}
QMenuBar {
    background: #111821;
    color: #d7dde8;
    border-bottom: 1px solid #263241;
}
QMenuBar::item {
    padding: 6px 10px;
    background: transparent;
}
QMenuBar::item:selected { background: #202a36; }
QMenu {
    background: #151c25;
    border: 1px solid #344255;
    color: #d7dde8;
}
QMenu::item { padding: 7px 28px; }
QMenu::item:selected { background: #263241; }
QToolBar {
    background: #151c25;
    border: 0;
    border-bottom: 1px solid #263241;
    spacing: 4px;
    padding: 5px;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 8px;
    color: #d7dde8;
}
QToolButton:hover { background: #202a36; border-color: #344255; }
QToolButton:pressed { background: #111821; }
QToolButton:disabled { color: #687484; }
QFrame#controlBar, QFrame#statusBar {
    background: #151c25;
    border: 1px solid #263241;
    border-radius: 8px;
}
QPushButton {
    background: #202a36;
    border: 1px solid #344255;
    border-radius: 6px;
    color: #edf2f8;
    padding: 7px 12px;
    min-height: 24px;
}
QPushButton::icon { color: #edf2f8; }
QPushButton:hover { background: #2a3747; border-color: #4c5e74; }
QPushButton:pressed { background: #18212c; }
QPushButton:disabled {
    color: #687484;
    background: #151b23;
    border-color: #202936;
}
QPushButton#primaryButton {
    background: #1267d3;
    border-color: #3181ee;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background: #1976ed; }
QSlider::groove:horizontal {
    height: 6px;
    background: #263241;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
    background: #5aa7ff;
}
QSpinBox {
    background: #111821;
    border: 1px solid #344255;
    border-radius: 6px;
    padding: 5px 30px 5px 8px;
    min-height: 28px;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 28px;
    border-left: 1px solid #344255;
    background: #202a36;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background: #2a3747;
}
QSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 8px solid #edf2f8;
}
QSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #edf2f8;
}
QSpinBox::up-arrow:disabled {
    border-bottom-color: #687484;
}
QSpinBox::down-arrow:disabled {
    border-top-color: #687484;
}
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QLabel#statusLabel {
    color: #ffd166;
    font-weight: 600;
}
"""

# ==========================================
# 1. 高性能视口交互组件
# ==========================================
class VideoWidget(QWidget):
    clicked_pos = pyqtSignal(float, float)
    moved_pos = pyqtSignal(float, float)

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
        if not state:
            return

        widget_rect = self.rect()
        painter.setFont(QFont("Segoe UI", 10))

        # 绘制视频信息（左上角）
        video_info = state.get("video_info")
        if video_info:
            # 路径截断，避免超长路径压扁信息框
            display_path = video_info['path']
            MAX_PATH = 100
            if len(display_path) > MAX_PATH:
                display_path = display_path[:40] + " ... " + display_path[-60:]

            info_text = (
                f"✌️视频地址 {display_path}\n"
                f"🖼️分辨率 {video_info['resolution']}  |  🎬 帧率 {video_info['fps']} FPS  |  "
                f"📊 总帧数 {video_info['frames']}  |  ⏱️ 时长 {video_info['duration']}"
            )

            # 动态背景：用当前字体度量计算实际需要的宽高
            painter.setFont(QFont("Segoe UI", 9))
            fm = painter.fontMetrics()
            lines = info_text.split('\n')
            text_w = max(fm.horizontalAdvance(ln) for ln in lines) + 28
            text_h = fm.lineSpacing() * len(lines) + 18
            info_bg_rect = QRectF(8, 8, text_w, text_h)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.drawRoundedRect(info_bg_rect, 6, 6)

            painter.setPen(QColor("#edf2f8"))
            painter.drawText(info_bg_rect.adjusted(12, 8, -12, -8), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, info_text)

        def draw_point(pt, color, radius=4):
            screen = self._to_screen_point(pt)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(screen, radius, radius)
            return screen

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

        origin_pt = state.get("origin_pt")
        if origin_pt:
            origin = self._to_screen_point(origin_pt)
            painter.setPen(QPen(QColor("#ff9f1c"), 1.1))
            painter.drawLine(QPointF(origin.x(), 0), QPointF(origin.x(), widget_rect.height()))
            painter.drawLine(QPointF(0, origin.y()), QPointF(widget_rect.width(), origin.y()))
            draw_point(origin_pt, "#ff4d4f", 5)
            painter.setPen(QPen(QColor("#ff4d4f")))
            painter.drawText(origin + QPointF(8, -8), "Origin (0,0)")

        mark = state.get("current_mark")
        if mark:
            p = draw_point(mark["pixel_pt"], "#2fef77", 4)
            painter.setPen(QPen(QColor("#2fef77")))
            painter.drawText(p + QPointF(10, -10), mark["text"])

        mouse = state.get("mouse_curr_pos")
        if mouse:
            m = self._to_screen_point(mouse)
            if widget_rect.contains(m.toPoint()):
                painter.setPen(QPen(QColor(255, 255, 255, 175), 1))
                painter.drawLine(QPointF(m.x(), 0), QPointF(m.x(), widget_rect.height()))
                painter.drawLine(QPointF(0, m.y()), QPointF(widget_rect.width(), m.y()))

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
            if coords: 
                self.moved_pos.emit(*coords)


# ==========================================
# 2. 主窗口控制层
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高速视频运动分析工具 V3.2")
        self.resize(1600, 900)

        self.cap = None
        self.video_path = ""  # 视频文件路径
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
        self.preview_max_w = 1152
        self.preview_max_h = 648
        self.preview_scale_percent = 100

        self.state = "IDLE" 
        self.calib_pt1 = None
        self.calib_line = None
        self.show_calib_line = True
        self.scale_factor = 1.0 
        self.physics_unit = "单位" 
        self.origin_pt = None   
        self.mouse_curr_pos = None 
        self.brightness = 0
        self.contrast = 1.0

        self.marks = {}
        # 零点时刻同步变量
        self.first_time_sec = None  # 记录捕捉模式第一个点的绝对时间(s)
        
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)

        self.init_ui()

    def _std_icon(self, standard_pixmap):
        source_icon = self.style().standardIcon(standard_pixmap)
        icon = QIcon()
        for mode, color in [
            (QIcon.Mode.Normal, QColor("#edf2f8")),
            (QIcon.Mode.Active, QColor("#ffffff")),
            (QIcon.Mode.Disabled, QColor("#687484")),
        ]:
            pixmap = source_icon.pixmap(24, 24)
            tinted = QPixmap(pixmap.size())
            tinted.fill(Qt.GlobalColor.transparent)
            painter = QPainter(tinted)
            painter.drawPixmap(0, 0, pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(tinted.rect(), color)
            painter.end()
            icon.addPixmap(tinted, mode)
        return icon

    def create_actions(self):
        style = QStyle.StandardPixmap

        self.act_load = QAction(self._std_icon(style.SP_DialogOpenButton), "加载视频", self)
        self.act_load.setShortcut(QKeySequence.StandardKey.Open)
        self.act_load.triggered.connect(self.load_video)

        self.act_export = QAction(self._std_icon(style.SP_DialogSaveButton), "导出 Excel", self)
        self.act_export.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.act_export.triggered.connect(self.export_to_excel)

        self.act_export_frame = QAction(self._std_icon(style.SP_DialogOkButton), "导出当前帧", self)
        self.act_export_frame.triggered.connect(self.export_current_frame)

        self.act_import = QAction(self._std_icon(style.SP_FileDialogContentsView), "导入 Excel", self)
        self.act_import.setShortcut(QKeySequence.StandardKey.Open)
        self.act_import.triggered.connect(self.import_from_excel)

        self.act_calib = QAction(self._std_icon(style.SP_FileDialogDetailedView), "标定尺寸", self)
        self.act_calib.triggered.connect(lambda: self.set_state("CALIBRATING_1"))

        self.act_origin = QAction(self._std_icon(style.SP_DialogNoButton), "设置原点", self)
        self.act_origin.triggered.connect(lambda: self.set_state("SET_ORIGIN"))

        self.act_mark = QAction(self._std_icon(style.SP_DialogApplyButton), "开始捕捉", self)
        self.act_mark.triggered.connect(lambda: self.set_state("MARKING"))

        self.act_reset_view = QAction(self._std_icon(style.SP_BrowserReload), "复位画面", self)
        self.act_reset_view.triggered.connect(self.reset_label_view)

        self.act_exit = QAction("退出", self)
        self.act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_exit.triggered.connect(self.close)

        self.video_actions = [
            self.act_calib,
            self.act_origin,
            self.act_mark,
            self.act_export,
            self.act_import,
            self.act_export_frame,
            self.act_reset_view,
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
        track_menu.addAction(self.act_origin)
        track_menu.addAction(self.act_mark)

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
        toolbar.addAction(self.act_origin)
        toolbar.addAction(self.act_mark)
        toolbar.addSeparator()
        toolbar.addAction(self.act_export)
        toolbar.addAction(self.act_import)
        toolbar.addAction(self.act_export_frame)
        toolbar.addAction(self.act_reset_view)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

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
        self.plot_x = pg.PlotWidget(title="输电塔水平 (X轴) 振动位移")
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)
        self.curve_x = self.plot_x.plot(pen=pg.mkPen('#5ac8fa', width=2), symbol='o', symbolSize=5, symbolBrush='#5ac8fa')
        self.curve_x.setClipToView(True)

        self.plot_y = pg.PlotWidget(title="输电塔竖向 (Y轴) 振动位移")
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        self.curve_y = self.plot_y.plot(pen=pg.mkPen('#ff5c8a', width=2), symbol='o', symbolSize=5, symbolBrush='#ff5c8a')
        self.curve_y.setClipToView(True)

        # === 新增：当前帧高亮数据层 ===
        # 采用醒目的金色五角星，带有白色描边，以区别于普通的圆形数据点
        highlight_pen = pg.mkPen('#ff9800', width=1.0)
        highlight_brush = pg.mkBrush(color='#ffeb3b') 
        self.highlight_x = self.plot_x.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush=highlight_brush, symbolPen=highlight_pen)
        self.highlight_x.setZValue(10) # 确保高亮点渲染在最上层
        
        self.highlight_y = self.plot_y.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush=highlight_brush, symbolPen=highlight_pen)
        self.highlight_y.setZValue(10)
        # ==============================

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
        self.btn_play.setIcon(self._std_icon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_prev = QPushButton("上一帧")
        self.btn_prev.setIcon(self._std_icon(QStyle.StandardPixmap.SP_MediaSeekBackward))
        self.btn_next = QPushButton("下一帧")
        self.btn_next.setIcon(self._std_icon(QStyle.StandardPixmap.SP_MediaSeekForward))

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

        self.label_preview_scale = QLabel("预览:")
        self.spin_preview_scale = QSpinBox()
        self.spin_preview_scale.setRange(25, 100)
        self.spin_preview_scale.setSingleStep(5)
        self.spin_preview_scale.setValue(100)
        self.spin_preview_scale.setSuffix(" %")
        self.spin_preview_scale.setEnabled(False)

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
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.spin_frame.editingFinished.connect(self.on_frame_spin_finished)
        self.slider_brightness.valueChanged.connect(self.on_adjustment_changed)
        self.slider_contrast.valueChanged.connect(self.on_adjustment_changed)
        self.chk_keep_calib.toggled.connect(self.on_keep_calib_toggled)
        self.spin_preview_scale.valueChanged.connect(self.on_preview_scale_changed)

        for widget in [self.btn_play, self.btn_prev, self.btn_next]:
            widget.setEnabled(False)

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
        bottom_layout.addWidget(self.label_preview_scale)
        bottom_layout.addWidget(self.spin_preview_scale)
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
            self.btn_play.setIcon(self._std_icon(QStyle.StandardPixmap.SP_MediaPlay))
            self.btn_play.setText("播放")
        else:
            self.play_timer.start(self._playback_interval_ms())
            self.btn_play.setIcon(self._std_icon(QStyle.StandardPixmap.SP_MediaStop))
            self.btn_play.setText("暂停")

    def _playback_interval_ms(self):
        step_size = max(1, self.spin_step.value())
        # 高帧率视频按真实时间会产生 1ms 级刷新，Qt 主线程会被解码/绘制压满。
        # 这里限制显示刷新上限约 60 FPS，同时保留按步长推进的帧号语义。
        return max(16, int(1000 * step_size / max(self.fps, 1.0)))

    def play_next_frame(self):
        if self.is_seeking:
            return
        next_interval = self._playback_interval_ms()
        if self.play_timer.interval() != next_interval:
            self.play_timer.setInterval(next_interval)
        step_size = self.spin_step.value()
        if self.current_frame_idx + step_size < self.total_frames:
            self.seek_frame(step_size)
        else:
            self.seek_frame(self.total_frames - 1, absolute=True)
            self.play_timer.stop()
            self.btn_play.setIcon(self._std_icon(QStyle.StandardPixmap.SP_MediaPlay))
            self.btn_play.setText("播放")

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if not filename:
            return
        if filename:
            self.video_path = filename  # 保存视频路径

            if self.play_timer.isActive(): self.toggle_play()
            if self.cap: self.cap.release()
            
            self.cap = cv2.VideoCapture(filename)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.apply_preview_scale()
            self.video_widget.reset_view()

            self.marks.clear()
            self.origin_pt = None
            self.calib_pt1 = None
            self.calib_line = None
            self.physics_unit = "单位"
            self.current_frame_idx = -1 
            self.decoder_next_frame_idx = -1
            self.base_frame = None
            self.base_qimage = None
            self.adjusted_qimage = None
            self.frame_cache.clear()
            self.first_time_sec = None # 重置时间零点基准
            self.reset_adjustments(update_frame=False)
            
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
            
            for widget in [self.slider, self.btn_play, self.btn_prev, self.btn_next, self.spin_step, self.spin_preview_scale,
                           self.spin_frame, self.slider_brightness, self.slider_contrast, self.chk_keep_calib]:
                widget.setEnabled(True)
            for action in self.video_actions:
                action.setEnabled(True)
            
            self.seek_frame(0, absolute=True)
            self.set_state("IDLE")

    def reset_label_view(self):
        self.video_widget.reset_view()
        self.reset_adjustments(update_frame=True)
        self.render_frame()

    def reset_adjustments(self, update_frame=True):
        self.brightness = 0
        self.contrast = 1.0
        for slider, value in [(self.slider_brightness, 0), (self.slider_contrast, 100)]:
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        if update_frame and self.base_frame is not None:
            self._refresh_adjusted_frame()

    def apply_preview_scale(self):
        base_scale = min(1.0, self.preview_max_w / max(self.video_w, 1), self.preview_max_h / max(self.video_h, 1))
        scale = base_scale * self.preview_scale_percent / 100.0
        self.video_widget.canvas_w = max(1, int(self.video_w * scale))
        self.video_widget.canvas_h = max(1, int(self.video_h * scale))

    def on_preview_scale_changed(self, value):
        if not self.cap:
            return
        self.preview_scale_percent = value
        target = max(0, self.current_frame_idx)
        self.apply_preview_scale()
        self.frame_cache.clear()
        self.base_frame = None
        self.base_qimage = None
        self.adjusted_qimage = None
        self.current_frame_idx = -1
        self.decoder_next_frame_idx = -1
        self.seek_frame(target, absolute=True)

    def seek_frame(self, val, absolute=False):
        if not self.cap: return
        if self.is_seeking:
            return
        
        target_idx = val if absolute else self.current_frame_idx + val
        target_idx = max(0, min(target_idx, self.total_frames - 1))
        if target_idx == self.current_frame_idx and self.base_qimage is not None:
            return

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
            elif 1 < diff < 200 and decoder_is_aligned:
                for _ in range(diff - 1):
                    self.cap.grab()
                ret, frame = self.cap.read()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = self.cap.read()
                
            if ret:
                self._set_current_frame(target_idx, frame)
        finally:
            self.is_seeking = False

    def _set_current_frame(self, target_idx, frame):
        resized = cv2.resize(frame, (self.video_widget.canvas_w, self.video_widget.canvas_h), interpolation=cv2.INTER_LINEAR)
        self.base_frame = resized
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

        self.render_frame()

    def _frame_to_qimage(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1], QImage.Format.Format_RGB888).copy()

    def _refresh_adjusted_frame(self):
        if self.base_frame is None:
            return
        if self.brightness == 0 and abs(self.contrast - 1.0) < 0.001:
            self.adjusted_qimage = self.base_qimage
        else:
            adjusted = cv2.convertScaleAbs(self.base_frame, alpha=self.contrast, beta=self.brightness)
            self.adjusted_qimage = self._frame_to_qimage(adjusted)
        self.video_widget.set_frame(self.adjusted_qimage)

    def on_slider_pressed(self):
        self.is_slider_dragging = True

    def on_slider_moved(self, position):
        self.spin_frame.blockSignals(True)
        self.spin_frame.setValue(position)
        self.spin_frame.blockSignals(False)

    def on_slider_released(self):
        self.is_slider_dragging = False
        self.seek_frame(self.slider.value(), absolute=True)

    def on_frame_spin_finished(self):
        if self.cap:
            self.seek_frame(self.spin_frame.value(), absolute=True)

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
            self.mouse_curr_pos = (x, y)
        self.render_frame()

    def render_frame(self):
        if self.base_qimage is None:
            return

        current_mark = None
        if self.current_frame_idx in self.marks:
            data = self.marks[self.current_frame_idx]
            rx, ry = data['rel_x'], data['rel_y']
            if self.first_time_sec is not None:
                rel_time = data['time'] - self.first_time_sec
                text = f"X:{rx:.2f} Y:{ry:.2f} | t:{rel_time:.3f}s"
            else:
                text = f"X: {rx:.2f}, Y: {ry:.2f} {self.physics_unit}"
            current_mark = {"pixel_pt": data["pixel_pt"], "text": text}

        # 视频信息
        video_info = None
        if self.video_path:
            filename = os.path.basename(self.video_path)
            duration = self.total_frames / max(self.fps, 1.0)
            video_info = {
                "name": filename,
                "path": self.video_path,
                "resolution": f"{self.video_w}×{self.video_h}",
                "fps": f"{self.fps:.2f}",
                "frames": self.total_frames,
                "duration": f"{duration:.2f}s"
            }

        self.video_widget.set_overlay_state(
            mode=self.state,
            calib_pt1=self.calib_pt1,
            calib_line=self.calib_line if self.show_calib_line else None,
            origin_pt=self.origin_pt,
            current_mark=current_mark,
            mouse_curr_pos=self.mouse_curr_pos,
            video_info=video_info,
        )

        # === 新增：画面刷新时，同步更新图表高亮点 ===
        self.update_highlight()

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
                self.calib_line = {
                    "p1": self.calib_pt1,
                    "p2": (x, y),
                    "text": f"{real_dist:.3f} {self.physics_unit}",
                }
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
            self.auto_range_plots()
            
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
            self.plot_x.getAxis('bottom').setLabel("相对时间", units='s')
            self.plot_y.getAxis('bottom').setLabel("相对时间", units='s')
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

        # === 新增：图表整体重绘时，刷新高亮点状态 ===
        self.update_highlight()

    def update_highlight(self):
        """核心新增：独立更新图表中的当前帧高亮点"""
        if self.current_frame_idx in self.marks:
            # 如果当前帧存在捕捉数据
            d = self.marks[self.current_frame_idx]
            
            # 判断横坐标使用的是绝对帧数还是相对时间(s)
            if self.first_time_sec is not None:
                x_val = d['time'] - self.first_time_sec
            else:
                x_val = self.current_frame_idx

            # 更新高亮点的坐标
            self.highlight_x.setData([x_val], [d['rel_x']])
            self.highlight_y.setData([x_val], [d['rel_y']])
        else:
            # 如果当前帧没有捕捉数据，则清空高亮层（隐藏高亮点）
            self.highlight_x.setData([], [])
            self.highlight_y.setData([], [])

    def auto_range_plots(self):
        self.plot_x.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_y.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_x.autoRange()
        self.plot_y.autoRange()

    def export_current_frame(self):
        if not self.cap or self.current_frame_idx < 0:
            QMessageBox.warning(self, "无法导出", "请先加载视频并定位到需要导出的帧。")
            return
        if self.play_timer.isActive():
            self.toggle_play()

        default_name = f"frame_{self.current_frame_idx:06d}.png"
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出当前帧原始画面", default_name, "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap Image (*.bmp)"
        )
        if not filename:
            return
        lower_name = filename.lower()
        if not lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            filename += ".png"

        old_idx = self.current_frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, old_idx)
        ret, raw_frame = self.cap.read()
        self.decoder_next_frame_idx = old_idx + 1
        if not ret:
            QMessageBox.critical(self, "导出失败", "无法从视频中读取当前帧。")
            return
        if cv2.imwrite(filename, raw_frame):
            QMessageBox.information(self, "成功", f"当前帧原始画面已导出：\n{filename}")
        else:
            QMessageBox.critical(self, "导出失败", "写入图像文件失败，请检查文件路径或格式。")


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
                df = pd.DataFrame(data_list)
                
                # --- 标定元数据表：保存当前分析参数，用于后续导入恢复 ---
                metadata = {
                    'Parameter': [
                        'scale_factor', 'physics_unit',
                        'origin_x', 'origin_y',
                        'first_time_sec',
                        'calib_p1_x', 'calib_p1_y', 'calib_p2_x', 'calib_p2_y', 'calib_text',
                        'video_path', 'video_w', 'video_h', 'fps', 'total_frames'
                    ],
                    'Value': [
                        self.scale_factor, self.physics_unit,
                        self.origin_pt[0] if self.origin_pt else None,
                        self.origin_pt[1] if self.origin_pt else None,
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
        """从 Excel 文件导入已有的捕捉数据，同时恢复标定参数，支持对已捕捉数据的继续修改和更新"""
        filename, _ = QFileDialog.getOpenFileName(self, "导入 Excel 捕捉数据", "", "Excel Files (*.xlsx)")
        if not filename:
            return

        if self.play_timer.isActive():
            self.toggle_play()

        try:
            xls = pd.ExcelFile(filename)
            sheet_names = xls.sheet_names

            # ---- 1. 读取元数据，恢复标定参数 ----
            if 'Metadata' in sheet_names:
                df_meta = pd.read_excel(filename, sheet_name='Metadata')
                meta_dict = dict(zip(df_meta['Parameter'], df_meta['Value']))

                if 'scale_factor' in meta_dict and pd.notna(meta_dict['scale_factor']):
                    self.scale_factor = float(meta_dict['scale_factor'])
                if 'physics_unit' in meta_dict and pd.notna(meta_dict['physics_unit']):
                    self.physics_unit = str(meta_dict['physics_unit'])
                if 'origin_x' in meta_dict and 'origin_y' in meta_dict:
                    ox, oy = meta_dict['origin_x'], meta_dict['origin_y']
                    if pd.notna(ox) and pd.notna(oy):
                        self.origin_pt = (float(ox), float(oy))
                if 'first_time_sec' in meta_dict and pd.notna(meta_dict['first_time_sec']):
                    self.first_time_sec = float(meta_dict['first_time_sec'])

                # 恢复标定线段 —— 即画面上显示的定标尺
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

                # 若当前已加载视频且路径不一致，提示用户确认
                imported_video = meta_dict.get('video_path', None)
                if imported_video and pd.notna(imported_video) and self.video_path:
                    imported_video = str(imported_video)
                    if imported_video != self.video_path:
                        reply = QMessageBox.question(
                            self, "视频不匹配",
                            f"导入数据来自:\n{imported_video}\n\n当前加载的是:\n{self.video_path}\n\n仍要恢复标定参数并导入数据吗？",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        if reply != QMessageBox.StandardButton.Yes:
                            return

            # ---- 2. 读取 TrackData 表（优先 TrackData sheet，否则取第一个非 Metadata sheet） ----
            data_sheet = 'TrackData' if 'TrackData' in sheet_names else [s for s in sheet_names if s != 'Metadata'][0]
            df = pd.read_excel(filename, sheet_name=data_sheet)

            # 自动识别列名
            frame_col = time_col = rel_time_col = phys_x_col = phys_y_col = pix_x_col = pix_y_col = None
            for col in df.columns:
                cl = col.lower()
                if frame_col is None and 'frame' in cl:
                    frame_col = col
                elif time_col is None and 'absolute_time' in cl:
                    time_col = col
                elif rel_time_col is None and 'relative_time' in cl:
                    rel_time_col = col
                elif phys_x_col is None and 'physical_x' in cl:
                    phys_x_col = col
                elif phys_y_col is None and 'physical_y' in cl:
                    phys_y_col = col
                elif pix_x_col is None and 'canvas_pixel_x' in cl:
                    pix_x_col = col
                elif pix_y_col is None and 'canvas_pixel_y' in cl:
                    pix_y_col = col

            if frame_col is None:
                QMessageBox.critical(self, "导入失败", "无法识别帧索引列 (Frame_Index)")
                return

            # ---- 3. 恢复捕捉数据 ----
            self.marks.clear()
            skipped = 0
            for _, row in df.iterrows():
                frame_idx = int(row[frame_col])
                if self.total_frames > 0 and not (0 <= frame_idx < self.total_frames):
                    skipped += 1
                    continue

                t = (
                    float(row[time_col]) if time_col is not None and pd.notna(row.get(time_col))
                    else (frame_idx / max(self.fps, 1.0))
                )
                rx = float(row[phys_x_col]) if phys_x_col is not None and pd.notna(row.get(phys_x_col)) else 0.0
                ry = float(row[phys_y_col]) if phys_y_col is not None and pd.notna(row.get(phys_y_col)) else 0.0
                px = float(row[pix_x_col]) if pix_x_col is not None and pd.notna(row.get(pix_x_col)) else 0.0
                py = float(row[pix_y_col]) if pix_y_col is not None and pd.notna(row.get(pix_y_col)) else 0.0

                self.marks[frame_idx] = {
                    'time': t, 'rel_x': rx, 'rel_y': ry, 'pixel_pt': (px, py)
                }

            # 若元数据未包含时间零点，尝试从相对时间列反推
            if self.first_time_sec is None and self.marks and rel_time_col is not None:
                first_key = min(self.marks.keys())
                rel_t = float(df.loc[df[frame_col] == first_key, rel_time_col].iloc[0])
                if pd.notna(rel_t):
                    self.first_time_sec = self.marks[first_key]['time'] - rel_t

            # ---- 4. 刷新界面 ----
            self.update_chart_labels()
            self.update_plots()
            self.auto_range_plots()
            self.render_frame()

            msg = f"成功导入 {len(self.marks)} 条记录"
            if skipped > 0:
                msg += f"（跳过 {skipped} 条超出当前视频帧范围的记录）"
            QMessageBox.information(self, "导入成功", msg)

        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"读取 Excel 文件出错：\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
