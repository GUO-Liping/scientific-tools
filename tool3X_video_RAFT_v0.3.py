import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path

from PyQt6.QtCore import Qt, QPointF, QLineF, QThread, pyqtSignal, QTimer, QRectF
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QTreeView, QStatusBar, QPushButton, QLineEdit,
                             QLabel, QSlider, QFileDialog, QFormLayout, QDoubleSpinBox,
                             QSpinBox, QCheckBox, QComboBox, QGroupBox, QToolBar, QStyle,
                             QInputDialog, QMessageBox)
from PyQt6.QtGui import QAction, QImage, QPixmap, QFileSystemModel, QPen, QColor, QPolygonF, QBrush, QPainter, QIntValidator
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPolygonItem, QGraphicsLineItem, QGraphicsEllipseItem

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 1. 动态双模现代化工业主题样式表 ======================
LIGHT_STYLE = """
QMainWindow { background-color: #F5F5F7; }
QWidget { color: #1C1C1E; font-family: 'Segoe UI', 'Microsoft YaHei'; font-size: 13px; }
QGroupBox { border: 1px solid #D1D1D6; border-radius: 6px; margin-top: 5px; font-weight: bold; color: #007AFF; background-color: #FFFFFF; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { background-color: #FFFFFF; border: 1px solid #C7C7CC; border-radius: 4px; padding: 5px 12px; color: #1C1C1E; font-weight: 500; }
QPushButton:hover { background-color: #E5E5EA; border-color: #007AFF; }
QSlider::groove:horizontal { border: 1px solid #C7C7CC; height: 6px; background: #E5E5EA; border-radius: 3px; }
QSlider::handle:horizontal { background: #007AFF; width: 14px; margin: -4px 0; border-radius: 7px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #FFFFFF; border: 1px solid #C7C7CC; border-radius: 4px; padding: 4px; color: #1C1C1E; }
QTreeView { background-color: #FFFFFF; border: 1px solid #D1D1D6; color: #1C1C1E; }
QToolBar { background-color: #FFFFFF; border-bottom: 1px solid #D1D1D6; spacing: 12px; padding: 5px; }
QStatusBar { background-color: #E5E5EA; color: #55555F; }
"""

DARK_STYLE = """
QMainWindow { background-color: #1E1E24; }
QWidget { color: #E0E0E6; font-family: 'Segoe UI', 'Microsoft YaHei'; font-size: 13px; }
QGroupBox { border: 1px solid #3A3A42; border-radius: 6px; margin-top: 5px; font-weight: bold; color: #00A8FF; background-color: #15151A; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { background-color: #2D2D38; border: 1px solid #4E4E5A; border-radius: 4px; padding: 5px 12px; color: #FFF; font-weight: 500; }
QPushButton:hover { background-color: #3E3E4F; border-color: #00A8FF; }
QSlider::groove:horizontal { border: 1px solid #4E4E5A; height: 6px; background: #2D2D38; border-radius: 3px; }
QSlider::handle:horizontal { background: #00A8FF; width: 14px; margin: -4px 0; border-radius: 7px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #2D2D38; border: 1px solid #4E4E5A; border-radius: 4px; padding: 4px; color: #FFF; }
QTreeView { background-color: #15151A; border: 1px solid #3A3A42; color: #CCC; }
QToolBar { background-color: #15151A; border-bottom: 1px solid #2D2D38; spacing: 12px; padding: 5px; }
QStatusBar { background-color: #15151A; color: #888; }
"""

class CollapsibleSection(QWidget):
    def __init__(self, title, is_dark=False, parent=None):
        super().__init__(parent)
        self.is_dark = is_dark
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 4)
        self.main_layout.setSpacing(2)

        self.btn_toggle = QPushButton(f"▼  {title}")
        self.update_header_style()
        self.main_layout.addWidget(self.btn_toggle)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(4, 4, 4, 6)
        self.main_layout.addWidget(self.container)

        self.btn_toggle.clicked.connect(self.on_toggle_clicked)
        self.is_collapsed = False

    def update_header_style(self):
        bg = "#2D2D38" if self.is_dark else "#E5E5EA"
        tc = "#FFF" if self.is_dark else "#1C1C1E"
        self.btn_toggle.setStyleSheet(
            f"QPushButton {{ text-align: left; font-weight: bold; padding: 6px 10px; "
            f"background-color: {bg}; color: {tc}; border: none; border-radius: 4px; }} "
            f"QPushButton:hover {{ background-color: #007AFF; color: white; }}"
        )

    def on_toggle_clicked(self):
        self.is_collapsed = not self.is_collapsed
        self.container.setVisible(not self.is_collapsed)
        prefix = "▶ " if self.is_collapsed else "▼ "
        pure_title = self.btn_toggle.text()[2:]
        self.btn_toggle.setText(f"{prefix} {pure_title}")

    def add_widget(self, widget):
        self.container_layout.addWidget(widget)

# ====================== 3. 闭环拓扑多模交互控制顶点 ======================
# 增大角点尺寸，优化点击与可视化体验
CORNER_VISUAL_SIZE = 22
CORNER_HIT_RADIUS = 50
FREE_MODE_ELLIPSE_R = 12
FREE_MODE_HIT_RADIUS = 50

class ControlPoint(QGraphicsItem):
    def __init__(self, x, y, idx, view_ref):
        super().__init__()
        self.setPos(x, y)
        self.idx = idx
        self.view_ref = view_ref
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(10)

    def boundingRect(self):
        r = FREE_MODE_HIT_RADIUS if self.view_ref.free_transform_enabled else CORNER_HIT_RADIUS
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.view_ref.free_transform_enabled:
            painter.setBrush(QBrush(QColor(0, 122, 255)))
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawEllipse(QRectF(-FREE_MODE_ELLIPSE_R, -FREE_MODE_ELLIPSE_R,
                                       2 * FREE_MODE_ELLIPSE_R, 2 * FREE_MODE_ELLIPSE_R))
        else:
            pen = QPen(QColor(255, 59, 48), 5)
            pen.setCapStyle(Qt.PenCapStyle.SquareCap)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            painter.setPen(pen)
            s = CORNER_VISUAL_SIZE
            if self.idx == 0:
                painter.drawLine(0, 0, s, 0); painter.drawLine(0, 0, 0, s)
            elif self.idx == 1:
                painter.drawLine(0, 0, -s, 0); painter.drawLine(0, 0, 0, s)
            elif self.idx == 2:
                painter.drawLine(0, 0, -s, 0); painter.drawLine(0, 0, 0, -s)
            elif self.idx == 3:
                painter.drawLine(0, 0, s, 0); painter.drawLine(0, 0, 0, -s)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and isinstance(value, QPointF):
            parent = self.parentItem()
            if parent and hasattr(parent, 'pixmap'):
                self.view_ref.on_handle_moved(self.idx, value)
                return self.view_ref.points[self.idx]
        return super().itemChange(change, value)

# ====================== 4. 高级工业图像渲染视口画布 ======================
class AdvancedGraphicsView(QGraphicsView):
    roi_changed = pyqtSignal(list)
    calibration_completed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor(240, 240, 243)))

        self.free_transform_enabled = False
        self.current_ratio_val = 4.0 / 3.0

        self.is_panning = False
        self.is_dragging_roi = False
        self.last_mouse_pos = None

        self.bg_item = None
        self.polygon_item = None
        self.grid_items = []
        self.points = []
        self.point_items = []

        self.calib_state = 0
        self.calib_pt1 = None
        self.calib_line_item = None
        self.calib_pt1_item = None

        self.lbl_osd = QLabel(self)
        self.lbl_osd.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.lbl_osd.move(15, 15)
        self.update_osd_theme(False)
        self.lbl_osd.setText("✌️ 流体力学工作站 | 等待载入视频...")
        self.lbl_osd.adjustSize()

    # ---------- 【v0.3】安全清理全部图形项引用 ----------
    def _safe_clear_scene_items(self):
        """彻底清理场景引用，防止访问已删除的 C++ 对象导致崩溃"""
        # 先移除角点（它们挂在 bg_item 下，clear 时会一起清理，但先收拾引用更安全）
        for cp in self.point_items:
            try:
                self.scene.removeItem(cp)
            except Exception:
                pass
        self.point_items.clear()
        self.points.clear()

        # 清理网格线
        for line in self.grid_items:
            try:
                self.scene.removeItem(line)
            except Exception:
                pass
        self.grid_items.clear()

        # 清理多边形叠加层
        if self.polygon_item is not None:
            try:
                self.scene.removeItem(self.polygon_item)
            except Exception:
                pass
        self.polygon_item = None

        # 清理标定临时元素
        if self.calib_line_item is not None:
            try:
                self.scene.removeItem(self.calib_line_item)
            except Exception:
                pass
        self.calib_line_item = None
        if self.calib_pt1_item is not None:
            try:
                self.scene.removeItem(self.calib_pt1_item)
            except Exception:
                pass
        self.calib_pt1_item = None

        # 最后移除并释放背景图
        if self.bg_item is not None:
            try:
                self.scene.removeItem(self.bg_item)
            except Exception:
                pass
        self.bg_item = None

        # 清空整个场景（双重保险）
        self.scene.clear()

    def update_osd_theme(self, is_dark_mode):
        bg = "rgba(20, 20, 25, 230)" if is_dark_mode else "rgba(240, 240, 245, 230)"
        color = "#00FFCC" if is_dark_mode else "#007AFF"
        border = "#00A8FF" if is_dark_mode else "#007AFF"
        self.lbl_osd.setStyleSheet(
            f"background-color: {bg}; color: {color}; padding: 12px; "
            f"border-radius: 6px; border: 1.5px solid {border}; font-family: 'Consolas', monospace; "
            f"font-size: 14px; font-weight: bold; line-height: 140%;"
        )
        self.lbl_osd.adjustSize()

    def update_osd_text(self, text):
        self.lbl_osd.setText(text)
        self.lbl_osd.adjustSize()

    def set_base_frame(self, frame):
        """设置新帧作为背景 —— v0.3 含安全初始化路径"""
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if self.bg_item is None:
            # 首次载入或视频切换后的重新初始化
            self._safe_clear_scene_items()
            self.scene.setSceneRect(0, 0, w, h)
            self.bg_item = self.scene.addPixmap(pixmap)
            self.reset_roi_to_default(w, h)
            self.fitInView(self.bg_item, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            # 同视频刷新帧
            self.bg_item.setPixmap(pixmap)
        self.update_mesh_elements()

    def reset_roi_to_default(self, w=None, h=None):
        if not self.bg_item: return
        if w is None:
            rect = self.bg_item.pixmap().rect()
            w, h = rect.width(), rect.height()

        cx, cy = w / 2, h / 2
        init_w = w * 0.4
        init_h = init_w / self.current_ratio_val
        if init_h > h * 0.8:
            init_h = h * 0.5
            init_w = init_h * self.current_ratio_val

        self.points = [
            QPointF(cx - init_w/2, cy - init_h/2), QPointF(cx + init_w/2, cy - init_h/2),
            QPointF(cx + init_w/2, cy + init_h/2), QPointF(cx - init_w/2, cy + init_h/2)
        ]

        for pt in self.point_items:
            try:
                self.scene.removeItem(pt)
            except Exception:
                pass
        self.point_items.clear()

        for i, pt in enumerate(self.points):
            cp = ControlPoint(pt.x(), pt.y(), i, self)
            cp.setParentItem(self.bg_item)
            self.point_items.append(cp)
        self.update_mesh_elements()

    def start_calibration_mode(self):
        self.calib_state = 1
        self.setCursor(Qt.CursorShape.CrossCursor)
        if self.calib_line_item:
            try:
                self.scene.removeItem(self.calib_line_item)
            except Exception:
                pass
            self.calib_line_item = None
        if self.calib_pt1_item:
            try:
                self.scene.removeItem(self.calib_pt1_item)
            except Exception:
                pass
            self.calib_pt1_item = None

    # 【v0.2/v0.3】比例约束矩形缩放 —— 统一边界裁剪 + 全部四个角点同步更新
    def on_handle_moved(self, idx, new_pos):
        # 边界裁剪
        if self.bg_item:
            rect = self.bg_item.pixmap().rect()
            x = max(0.0, min(float(rect.width() - 1), new_pos.x()))
            y = max(0.0, min(float(rect.height() - 1), new_pos.y()))
            new_pos = QPointF(x, y)

        if self.free_transform_enabled:
            self.points[idx] = new_pos
        else:
            # 按长宽比约束计算对角锚点缩放
            fixed_idx = (idx + 2) % 4
            fixed_pt = self.points[fixed_idx]

            dx = new_pos.x() - fixed_pt.x()
            dy = new_pos.y() - fixed_pt.y()

            sign_x = 1 if dx >= 0 else -1
            sign_y = 1 if dy >= 0 else -1

            if abs(dx) / (abs(dy) + 1e-6) > self.current_ratio_val:
                dy = abs(dx) / self.current_ratio_val * sign_y
            else:
                dx = abs(dy) * self.current_ratio_val * sign_x

            target_pos = QPointF(fixed_pt.x() + dx, fixed_pt.y() + dy)

            # 二次边界裁剪
            tx = max(0.0, min(float(rect.width() - 1), target_pos.x()))
            ty = max(0.0, min(float(rect.height() - 1), target_pos.y()))
            target_pos = QPointF(tx, ty)

            # 重算并再次比例约束
            dx = target_pos.x() - fixed_pt.x()
            dy = target_pos.y() - fixed_pt.y()
            if abs(dx) / (abs(dy) + 1e-6) > self.current_ratio_val:
                dy = abs(dx) / self.current_ratio_val * sign_y
            else:
                dx = abs(dy) * self.current_ratio_val * sign_x
            target_pos = QPointF(fixed_pt.x() + dx, fixed_pt.y() + dy)

            self.points[idx] = target_pos

            # 同步相邻角点保证矩形
            if idx == 0:
                self.points[1] = QPointF(fixed_pt.x(), target_pos.y())
                self.points[3] = QPointF(target_pos.x(), fixed_pt.y())
            elif idx == 1:
                self.points[0] = QPointF(fixed_pt.x(), target_pos.y())
                self.points[2] = QPointF(target_pos.x(), fixed_pt.y())
            elif idx == 2:
                self.points[1] = QPointF(target_pos.x(), fixed_pt.y())
                self.points[3] = QPointF(fixed_pt.x(), target_pos.y())
            elif idx == 3:
                self.points[0] = QPointF(target_pos.x(), fixed_pt.y())
                self.points[2] = QPointF(fixed_pt.x(), target_pos.y())

        # 同步全部四个角点视觉位置（含被拖动角点）
        for i, cp in enumerate(self.point_items):
            cp.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
            cp.setPos(self.points[i])
            cp.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self.update_mesh_elements()
        self.roi_changed.emit(self.points)

    def update_mesh_elements(self):
        """完全精确覆盖顶点数据 —— v0.3 含安全守卫"""
        if not self.bg_item:
            return

        # 同步可视化角点到计算数组
        if self.point_items:
            self.points = [cp.pos() for cp in self.point_items]

        if len(self.points) < 4:
            return

        p0, p1, p2, p3 = self.points[0], self.points[1], self.points[2], self.points[3]
        poly = QPolygonF([p0, p1, p2, p3])

        # 【v0.3 关键】若 polygon_item 已被 C++ 层回收，重新创建
        if self.polygon_item is None:
            self.polygon_item = QGraphicsPolygonItem(self.bg_item)
            self.polygon_item.setBrush(QBrush(QColor(0, 122, 255, 25)))
            self.polygon_item.setPen(QPen(QColor(0, 122, 255), 2))
        self.polygon_item.setPolygon(poly)

        res = 8
        needed = (res - 1) * 2

        # 【v0.3 关键】grid_items 为空或长度不匹配时重建
        if len(self.grid_items) != needed:
            for line in self.grid_items:
                try:
                    self.scene.removeItem(line)
                except Exception:
                    pass
            self.grid_items.clear()
            for _ in range(needed):
                line = QGraphicsLineItem(self.bg_item)
                line.setPen(QPen(QColor(52, 199, 89), 1.0, Qt.PenStyle.DashLine))
                self.grid_items.append(line)

        idx = 0
        for i in range(1, res):
            alpha = i / res
            self.grid_items[idx].setLine(QLineF(p0 * (1 - alpha) + p3 * alpha, p1 * (1 - alpha) + p2 * alpha))
            idx += 1
            self.grid_items[idx].setLine(QLineF(p0 * (1 - alpha) + p1 * alpha, p3 * (1 - alpha) + p2 * alpha))
            idx += 1

    def refresh_handles_rendering(self):
        for cp in self.point_items:
            cp.update()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        sp = self.mapToScene(event.position().toPoint())

        if self.calib_state == 1 and self.bg_item:
            self.calib_pt1 = sp
            self.calib_state = 2
            self.calib_pt1_item = QGraphicsEllipseItem(-4, -4, 8, 8, self.bg_item)
            self.calib_pt1_item.setPos(sp)
            self.calib_pt1_item.setBrush(QBrush(QColor(255, 0, 0)))
            self.calib_pt1_item.setPen(QPen(QColor(255, 255, 255)))
            self.calib_line_item = QGraphicsLineItem(self.bg_item)
            self.calib_line_item.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            return

        elif self.calib_state == 2 and self.bg_item:
            self.calib_state = 0
            self.setCursor(Qt.CursorShape.ArrowCursor)
            dist = np.hypot(sp.x() - self.calib_pt1.x(), sp.y() - self.calib_pt1.y())
            self.calibration_completed.emit(dist)
            if self.calib_line_item:
                try:
                    self.scene.removeItem(self.calib_line_item)
                except Exception:
                    pass
                self.calib_line_item = None
            if self.calib_pt1_item:
                try:
                    self.scene.removeItem(self.calib_pt1_item)
                except Exception:
                    pass
                self.calib_pt1_item = None
            return

        hit = self.scene.itemAt(sp, self.transform())
        if isinstance(hit, ControlPoint):
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton and self.polygon_item:
            if self.polygon_item.contains(self.polygon_item.mapFromScene(sp)):
                self.is_dragging_roi = True
                self.last_mouse_pos = sp
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return

        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        sp = self.mapToScene(event.position().toPoint())

        if self.calib_state == 2 and self.calib_line_item:
            self.calib_line_item.setLine(QLineF(self.calib_pt1, sp))
            return

        if self.is_dragging_roi and self.bg_item:
            delta = sp - self.last_mouse_pos
            rect = self.bg_item.pixmap().rect()

            min_x = min(p.x() for p in self.points)
            max_x = max(p.x() for p in self.points)
            min_y = min(p.y() for p in self.points)
            max_y = max(p.y() for p in self.points)

            if min_x + delta.x() < 0:
                delta.setX(-min_x)
            if max_x + delta.x() > rect.width():
                delta.setX(rect.width() - max_x)
            if min_y + delta.y() < 0:
                delta.setY(-min_y)
            if max_y + delta.y() > rect.height():
                delta.setY(rect.height() - max_y)

            for i, cp in enumerate(self.point_items):
                cp.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
                cp.setPos(cp.pos() + delta)
                cp.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
                self.points[i] = cp.pos()

            self.last_mouse_pos = sp
            self.update_mesh_elements()
            self.roi_changed.emit(self.points)
            return

        if self.is_panning:
            dt = event.position().toPoint() - self.last_mouse_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dt.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dt.y())
            self.last_mouse_pos = event.position().toPoint()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_panning = False
        self.is_dragging_roi = False
        if self.calib_state == 0:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

# ====================== 5. 独立科研级图表输出窗口 ======================
class ScientificFlowCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#F5F5F7')
        super().__init__(self.fig)

    def render_plots(self, u_phys, v_phys, mag_phys, frame_b, is_dark=False):
        self.fig.clear()
        bg = '#1E1E24' if is_dark else '#F5F5F7'
        tc = '#FFFFFF' if is_dark else '#1C1C1E'
        self.fig.set_facecolor(bg)

        axes = self.fig.subplots(2, 2)
        for r in axes:
            for ax in r:
                ax.axis('off')
                ax.set_facecolor(bg)

        axes[0, 0].imshow(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Target Area B (Warped)', color=tc, fontsize=10, fontweight='bold')

        im1 = axes[0, 1].imshow(mag_phys, cmap='jet')
        axes[0, 1].set_title('Velocity Magnitude (m/s)', color=tc, fontsize=10, fontweight='bold')
        cb1 = self.fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cb1.ax.yaxis.set_tick_params(color=tc, labelcolor=tc)

        im2 = axes[1, 0].imshow(u_phys, cmap='RdBu_r')
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_title('Horizontal Velocity u (m/s)', color=tc, fontsize=10, fontweight='bold')
        cb2 = self.fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cb2.ax.yaxis.set_tick_params(color=tc, labelcolor=tc)

        im3 = axes[1, 1].imshow(v_phys, cmap='RdBu_r')
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title('Vertical Velocity v (m/s)', color=tc, fontsize=10, fontweight='bold')
        cb3 = self.fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cb3.ax.yaxis.set_tick_params(color=tc, labelcolor=tc)

        self.fig.tight_layout()
        self.draw()

class ResultWindow(QMainWindow):
    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self.setWindowTitle("RAFT 动力学矢量分析报告窗")
        self.resize(1100, 800)

        self.canvas = ScientificFlowCanvas(self, width=8, height=6)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        bg = '#1E1E24' if is_dark else '#F5F5F7'
        self.setStyleSheet(f"QMainWindow {{ background-color: {bg}; }}")

# ====================== 6. 并行计算异步管道 ======================
class RaftPipelineWorker(QThread):
    done_signal = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    msg_signal = pyqtSignal(str)

    def __init__(self, generation, fa, fb, cfg):
        super().__init__()
        self.generation = generation      # v0.3: 代际标记，丢弃过期结果
        self.fa, self.fb, self.cfg = fa.copy(), fb.copy(), cfg

    def run(self):
        try:
            self.msg_signal.emit("正在处理全场基础图像增强滤波...")
            if self.cfg['enhance']:
                for img in [self.fa, self.fb]:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16)).apply(l)
                    img[:] = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

            h, w = self.fa.shape[:2]
            img_a = cv2.copyMakeBorder(self.fa, 0, ((h+7)//8)*8 - h, 0, ((w+7)//8)*8 - w, cv2.BORDER_REFLECT)
            img_b = cv2.copyMakeBorder(self.fb, 0, ((h+7)//8)*8 - h, 0, ((w+7)//8)*8 - w, cv2.BORDER_REFLECT)

            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, raft_small, Raft_Small_Weights

            t_a = (torch.from_numpy(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0) * 2.0 - 1.0
            t_b = (torch.from_numpy(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0) * 2.0 - 1.0

            model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEVICE) if self.cfg['model'] == 'large' else raft_small(weights=Raft_Small_Weights.DEFAULT).to(DEVICE)
            model.eval()

            self.msg_signal.emit("RAFT 计算架构启动，正在提取光流位移场...")
            with torch.no_grad():
                flow = model(t_a, t_b, num_flow_updates=20)[-1][0].cpu().numpy()

            u, v = flow[0], -flow[1]
            mag = np.sqrt(u**2 + v**2)
            mask = mag > (self.cfg['threshold'] * mag.max())
            u[~mask] = 0
            v[~mask] = 0
            u, v = cv2.medianBlur(u.astype(np.float32), 5), cv2.medianBlur(v.astype(np.float32), 5)
            mag = np.sqrt(u**2 + v**2)

            self.done_signal.emit(self.generation, u, v, mag, img_a, img_b)
        except Exception as e:
            self.msg_signal.emit(f"管道异常挂起: {str(e)}")


# ====================== 7. 系统总控主窗口核心 ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高速视频定量光流流体力学分析系统 V6 PRO  |  v0.3")
        self.resize(1400, 900)

        self.is_dark_mode = False
        self.setStyleSheet(LIGHT_STYLE)

        self.cap = None
        self.video_path = ""
        self.total_frames = 0
        self.fps = 120.0
        self.native_w, self.native_h = 0, 0
        self.current_idx = -1

        self.frame_a_idx, self.frame_b_idx = -1, -1
        self.frame_a_raw, self.frame_b_raw = None, None

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.advance_frame_loop)

        # v0.3: 管道代际标记，用于丢弃旧视频遗留的过期计算结果
        self._pipeline_generation = 0
        self._active_worker = None

        self.setup_menu_bar()
        self.assemble_ui_workbench()

    def setup_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件 (File)")

        act_open = QAction("导入高速视频...", self)
        act_open.triggered.connect(self.import_high_speed_video)
        file_menu.addAction(act_open)
        file_menu.addSeparator()

        act_exit = QAction("退出系统", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        view_menu = menubar.addMenu("视图 (View)")
        act_theme = QAction("切换深/浅色主题", self)
        act_theme.triggered.connect(self.toggle_workbench_theme)
        view_menu.addAction(act_theme)

        act_reset = QAction("复位画面与裁剪区", self)
        act_reset.triggered.connect(self.reset_entire_workspace)
        view_menu.addAction(act_reset)

    def assemble_ui_workbench(self):
        toolbar = QToolBar("主控制面板")
        self.addToolBar(toolbar)

        act_open = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "导入视频", self)
        act_open.triggered.connect(self.import_high_speed_video)
        toolbar.addAction(act_open)

        act_reset = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "画面与区域重置", self)
        act_reset.triggered.connect(self.reset_entire_workspace)
        toolbar.addAction(act_reset)
        toolbar.addSeparator()

        act_theme = QAction("🌓 深/浅色主题", self)
        act_theme.triggered.connect(self.toggle_workbench_theme)
        toolbar.addAction(act_theme)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        h_layout = QHBoxLayout(main_widget)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        h_layout.addWidget(self.splitter)

        # ==================== [ 左侧可折叠功能组件栏 ] ====================
        left_sidebar = QWidget()
        left_vbox = QVBoxLayout(left_sidebar)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(0)

        self.left_vsplitter = QSplitter(Qt.Orientation.Vertical)

        # 上半部分：文件工作空间树
        tree_holder = QWidget()
        tree_layout = QVBoxLayout(tree_holder)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        self.sec_tree = CollapsibleSection("本地工作空间树", self.is_dark_mode)
        self.file_tree = QTreeView()
        self.tree_model = QFileSystemModel()
        self.tree_model.setRootPath(os.getcwd())
        self.file_tree.setModel(self.tree_model)
        self.file_tree.setRootIndex(self.tree_model.index(os.getcwd()))
        self.file_tree.clicked.connect(self.on_tree_file_selected)
        self.sec_tree.add_widget(self.file_tree)
        tree_layout.addWidget(self.sec_tree)
        self.left_vsplitter.addWidget(tree_holder)

        # 下半部分：参数与控件区
        controls_holder = QWidget()
        controls_vbox = QVBoxLayout(controls_holder)
        controls_vbox.setContentsMargins(0, 0, 0, 0)
        controls_vbox.setSpacing(0)

        self.sec_adjust = CollapsibleSection("视频画面光学精细化调整", self.is_dark_mode)
        adj_form = QFormLayout()

        hb_b = QHBoxLayout()
        self.sld_b = QSlider(Qt.Orientation.Horizontal)
        self.sld_b.setRange(-100, 100)
        self.sld_b.setValue(0)
        self.sld_b.valueChanged.connect(self.on_hardware_param_modified)
        hb_b.addWidget(self.sld_b)
        adj_form.addRow("光场亮度增益:", hb_b)

        hb_c = QHBoxLayout()
        self.sld_c = QSlider(Qt.Orientation.Horizontal)
        self.sld_c.setRange(50, 300)
        self.sld_c.setValue(100)
        self.sld_c.valueChanged.connect(self.on_hardware_param_modified)
        hb_c.addWidget(self.sld_c)
        adj_form.addRow("全场对比度:", hb_c)

        self.chk_clahe = QCheckBox("并行激活局部纹理增强 (仅分析生效)")
        self.chk_clahe.setChecked(True)
        adj_form.addRow(self.chk_clahe)

        self.cb_ratio = QComboBox()
        self.cb_ratio.addItems(["4:3", "1:1", "16:9", "9:16", "3:4", "2:3", "3:2", "5:4", "4:5", "8:5", "5:8", "自由无约束"])
        self.cb_ratio.currentTextChanged.connect(self.on_ratio_preset_changed)
        adj_form.addRow("目标区域约束:", self.cb_ratio)

        self.chk_free_transform = QCheckBox("开启自由变换 (Perspective Free Mode)")
        self.chk_free_transform.setChecked(False)
        self.chk_free_transform.toggled.connect(self.on_free_transform_toggled)
        adj_form.addRow(self.chk_free_transform)

        container_adj = QWidget()
        container_adj.setLayout(adj_form)
        self.sec_adjust.add_widget(container_adj)
        controls_vbox.addWidget(self.sec_adjust)

        self.sec_params = CollapsibleSection("RAFT 计算流场动力超参", self.is_dark_mode)
        p_form = QFormLayout()

        self.sb_fps = QDoubleSpinBox()
        self.sb_fps.setRange(1, 200000)
        self.sb_fps.setValue(120.0)
        p_form.addRow("采集相机 FPS:", self.sb_fps)

        hb_calib = QHBoxLayout()
        self.sb_ppm = QDoubleSpinBox()
        self.sb_ppm.setRange(0.001, 200000)
        self.sb_ppm.setValue(1488.3)
        self.sb_ppm.setDecimals(3)
        self.btn_run_calib = QPushButton("📏 画面测距标定")
        self.btn_run_calib.clicked.connect(self.trigger_manual_calibration)
        hb_calib.addWidget(self.sb_ppm)
        hb_calib.addWidget(self.btn_run_calib)
        p_form.addRow("空间系数(px/m):", hb_calib)

        self.cb_model = QComboBox()
        self.cb_model.addItems(["large", "small"])
        p_form.addRow("深度解算模型:", self.cb_model)

        self.sb_thresh = QDoubleSpinBox()
        self.sb_thresh.setRange(0, 1)
        self.sb_thresh.setValue(0.1)
        p_form.addRow("低通滤波阈值:", self.sb_thresh)
        container_p = QWidget()
        container_p.setLayout(p_form)
        self.sec_params.add_widget(container_p)
        controls_vbox.addWidget(self.sec_params)

        self.sec_btn = CollapsibleSection("RAFT 分析控制", self.is_dark_mode)
        btn_layout = QVBoxLayout()

        btn_la = QPushButton("📍 捕获锁定为起点帧 (Frame A)")
        btn_lb = QPushButton("📍 捕获锁定为终点帧 (Frame B)")
        btn_la.clicked.connect(lambda: self.lock_analysis_frame_slot('A'))
        btn_lb.clicked.connect(lambda: self.lock_analysis_frame_slot('B'))

        self.lbl_slot_msg = QLabel("匹配槽位: 对齐尚未建立")
        self.lbl_slot_msg.setStyleSheet("color: #007AFF; font-weight: bold; margin-bottom: 10px;")

        self.btn_run = QPushButton("🚀 启动目标区域 RAFT定量计算\n(结果自动弹窗显示)")
        self.btn_run.setStyleSheet("background-color: #007AFF; color: white; font-weight: bold; min-height: 45px;")
        self.btn_run.clicked.connect(self.trigger_raft_pipeline)

        btn_layout.addWidget(btn_la)
        btn_layout.addWidget(btn_lb)
        btn_layout.addWidget(self.lbl_slot_msg)
        btn_layout.addWidget(self.btn_run)

        container_btn = QWidget()
        container_btn.setLayout(btn_layout)
        self.sec_btn.add_widget(container_btn)
        controls_vbox.addWidget(self.sec_btn)

        self.left_vsplitter.addWidget(controls_holder)
        self.left_vsplitter.setSizes([250, 450])

        left_vbox.addWidget(self.left_vsplitter)
        self.splitter.addWidget(left_sidebar)

        # ==================== [ 中间视频播放视口 ] ====================
        center_layout_container = QWidget()
        center_vbox = QVBoxLayout(center_layout_container)

        self.view_viewport = AdvancedGraphicsView()
        self.view_viewport.calibration_completed.connect(self.on_calibration_finished)
        center_vbox.addWidget(self.view_viewport, stretch=8)

        # 精简联动播放控制区 + 时间轴 —— 单行
        single_row_ctrl = QHBoxLayout()
        single_row_ctrl.setSpacing(6)

        self.btn_playback_play = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "播放")
        self.btn_playback_pause = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause), "暂停")

        self.btn_playback_play.clicked.connect(self.start_video_play)
        self.btn_playback_pause.clicked.connect(self.stop_video_play)

        single_row_ctrl.addWidget(self.btn_playback_play)
        single_row_ctrl.addWidget(self.btn_playback_pause)
        single_row_ctrl.addSpacing(12)

        single_row_ctrl.addWidget(QLabel("步长:"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(1, 10000)
        self.spin_stride.setValue(1)
        self.spin_stride.setFixedWidth(70)
        single_row_ctrl.addWidget(self.spin_stride)

        btn_step_b = QPushButton("⏮")
        btn_step_f = QPushButton("⏭")
        btn_step_b.setFixedWidth(40)
        btn_step_f.setFixedWidth(40)
        btn_step_b.clicked.connect(lambda: self.move_frame_index_by_offset(-self.spin_stride.value()))
        btn_step_f.clicked.connect(lambda: self.move_frame_index_by_offset(self.spin_stride.value()))
        single_row_ctrl.addWidget(btn_step_b)
        single_row_ctrl.addWidget(btn_step_f)

        single_row_ctrl.addSpacing(16)

        self.slider_time = QSlider(Qt.Orientation.Horizontal)
        self.slider_time.sliderMoved.connect(self.on_slider_time_dragged)
        single_row_ctrl.addWidget(self.slider_time, stretch=4)

        self.edit_curr_frame = QLineEdit("0")
        self.edit_curr_frame.setFixedWidth(70)
        self.edit_curr_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edit_curr_frame.setValidator(QIntValidator(0, 1000000))
        self.edit_curr_frame.returnPressed.connect(self.on_frame_edit_return)
        single_row_ctrl.addWidget(self.edit_curr_frame)

        self.lbl_total_frames = QLabel(" / 0")
        self.lbl_total_frames.setStyleSheet("font-family: Consolas; font-weight: bold;")
        single_row_ctrl.addWidget(self.lbl_total_frames)

        center_vbox.addLayout(single_row_ctrl)
        self.splitter.addWidget(center_layout_container)

        self.splitter.setSizes([320, 1080])
        self.statusBar().showMessage("系统全面启动成功。v0.3  |  多视频安全切换就绪")

    # ====================== 8. 闭环业务逻辑与数据驱动控制 ======================
    def toggle_workbench_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        style = DARK_STYLE if self.is_dark_mode else LIGHT_STYLE
        self.setStyleSheet(style)
        self.view_viewport.setBackgroundBrush(QBrush(QColor(30, 30, 35) if self.is_dark_mode else QColor(240, 240, 243)))
        self.view_viewport.update_osd_theme(self.is_dark_mode)

        for sec in [self.sec_tree, self.sec_adjust, self.sec_params, self.sec_btn]:
            sec.is_dark = self.is_dark_mode
            sec.update_header_style()

        if self.cap is not None:
            self.refresh_view_frame()

    def reset_entire_workspace(self):
        self.sld_b.setValue(0)
        self.sld_c.setValue(100)
        self.chk_free_transform.setChecked(False)
        self.cb_ratio.setCurrentText("4:3")
        if self.cap is not None:
            self.view_viewport.reset_roi_to_default()
            self.view_viewport.fitInView(self.view_viewport.bg_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.refresh_view_frame()

    def on_free_transform_toggled(self, checked):
        self.view_viewport.free_transform_enabled = checked
        self.view_viewport.refresh_handles_rendering()

    def on_ratio_preset_changed(self, text):
        if text == "自由无约束" or self.view_viewport.free_transform_enabled:
            return
        w, h = map(float, text.split(':'))
        self.view_viewport.current_ratio_val = w / h
        if self.cap is not None:
            self.view_viewport.on_handle_moved(0, self.view_viewport.points[0])

    def trigger_manual_calibration(self):
        if self.cap is None:
            QMessageBox.warning(self, "警告", "请先加载视频！")
            return
        self.statusBar().showMessage("标定仪激活：请在画面中点击【起点】以测量已知长度线段...")
        self.view_viewport.start_calibration_mode()

    def on_calibration_finished(self, px_dist):
        if px_dist < 1.0:
            self.statusBar().showMessage("标定终止：点位距离过小。")
            return
        real_dist, ok = QInputDialog.getDouble(
            self, "建立物理空间映射",
            f"画布截取跨度: {px_dist:.2f} px\n\n请输入该线段的实际物理长度 (米):",
            1.0, 0.001, 100000, 4
        )
        if ok and real_dist > 0:
            ppm = px_dist / real_dist
            self.sb_ppm.setValue(ppm)
            self.statusBar().showMessage(f"空间映射更新成功：1 米 = {ppm:.3f} 像素")
        else:
            self.statusBar().showMessage("用户取消标定操作。")

    def import_high_speed_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择导入高速视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if p:
            self.bind_video_resource(p)

    # 【v0.3 关键修复】多视频安全切换
    def bind_video_resource(self, path):
        """加载新视频 —— 安全停止旧资源、重置全部视图状态"""
        # 1. 停止播放循环
        self.stop_video_play()

        # 2. 使旧管道结果失效（代际递增）
        self._pipeline_generation += 1
        self._active_worker = None

        # 3. 释放旧视频捕捉器
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # 4. 打开新视频并校验
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "视频加载失败",
                                 f"无法打开视频文件:\n{path}\n\n请检查文件是否损坏或格式不支持。")
            return

        self.cap = cap
        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 120.0
        self.native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.total_frames <= 0 or self.native_w <= 0 or self.native_h <= 0:
            QMessageBox.critical(self, "视频数据异常",
                                 f"读取视频元数据失败:\n"
                                 f"帧数={self.total_frames}, 尺寸={self.native_w}x{self.native_h}")
            self.cap.release()
            self.cap = None
            return

        # 5. 重置界面参数
        self.sb_fps.setValue(self.fps)
        self.lbl_total_frames.setText(f" / {self.total_frames - 1}")
        self.slider_time.setRange(0, self.total_frames - 1)

        # 6. 清理帧槽位
        self.frame_a_idx, self.frame_b_idx = -1, -1
        self.frame_a_raw, self.frame_b_raw = None, None
        self.lbl_slot_msg.setText("匹配槽位: 对齐尚未建立")

        # 7. 重置视口（bg_item=None 触发完整的场景重建路径）
        self.view_viewport._safe_clear_scene_items()
        self.current_idx = -1
        self.refresh_view_frame(0)

        # 8. 重置比例预设
        self.cb_ratio.setCurrentText("4:3")
        self.on_ratio_preset_changed("4:3")

        self.statusBar().showMessage(f"视频加载完成: {os.path.basename(path)}  |  "
                                     f"{self.native_w}x{self.native_h}  |  {self.total_frames} 帧  |  {self.fps:.1f} FPS")

    def on_tree_file_selected(self, index):
        p = self.tree_model.filePath(index)
        if os.path.isfile(p) and p.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.bind_video_resource(p)

    def start_video_play(self):
        if self.cap is not None:
            self.play_timer.start(int(1000 / max(self.fps, 1.0)))

    def stop_video_play(self):
        self.play_timer.stop()

    def advance_frame_loop(self):
        if self.cap is None:
            self.stop_video_play()
            return
        if self.current_idx < self.total_frames - 1:
            self.slider_time.setValue(self.current_idx + 1)
            self.refresh_view_frame(self.current_idx + 1, sequential=True)
        else:
            self.stop_video_play()

    def move_frame_index_by_offset(self, offset):
        if self.cap is not None:
            t = max(0, min(self.total_frames - 1, self.current_idx + offset))
            self.slider_time.setValue(t)
            self.refresh_view_frame(t, sequential=(0 < offset <= 60))

    def on_slider_time_dragged(self, v):
        if self.current_idx != v:
            self.refresh_view_frame(v, sequential=False)

    def on_frame_edit_return(self):
        if self.cap is None:
            return
        try:
            val = int(self.edit_curr_frame.text())
            val = max(0, min(self.total_frames - 1, val))
            self.slider_time.setValue(val)
            self.refresh_view_frame(val, sequential=False)
        except ValueError:
            self.edit_curr_frame.setText(str(self.current_idx))

    def on_hardware_param_modified(self):
        if self.cap is not None:
            self.refresh_view_frame(self.current_idx)

    def refresh_view_frame(self, index=None, sequential=False):
        """20G 级大视频帧寻找核心性能优化 —— v0.3 增加空指针守卫"""
        if self.cap is None:
            return

        if index is not None:
            if sequential and index > self.current_idx:
                for _ in range(index - self.current_idx - 1):
                    self.cap.grab()
                ret, frame = self.cap.read()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = self.cap.read()
            self.current_idx = index
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_idx)
            ret, frame = self.cap.read()

        if ret and frame is not None:
            bv, cv = self.sld_b.value(), self.sld_c.value() / 100.0
            duration = self.total_frames / max(self.fps, 1.0)
            display_name = os.path.basename(self.video_path)

            osd_info = (
                f"🎬 视频源: {display_name}\n"
                f"🖼️ 分辨率: {self.native_w}x{self.native_h} px  |  🎞️ 原生帧率: {self.fps:.1f} FPS\n"
                f"📊 视频总帧: {self.total_frames} Frame  |  ⏱️ 预估时长: {duration:.2f} s"
            )
            self.view_viewport.update_osd_text(osd_info)

            adjusted = cv2.convertScaleAbs(frame, alpha=cv, beta=bv)
            self.view_viewport.set_base_frame(adjusted)

            if not self.edit_curr_frame.hasFocus():
                self.edit_curr_frame.setText(str(self.current_idx))

    def lock_analysis_frame_slot(self, label):
        if self.cap is None or self.current_idx == -1:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_idx)
        _, f = self.cap.read()
        if f is None:
            return
        proc = cv2.convertScaleAbs(f, alpha=self.sld_c.value()/100.0, beta=self.sld_b.value())
        if label == 'A':
            self.frame_a_idx = self.current_idx
            self.frame_a_raw = proc
        else:
            self.frame_b_idx = self.current_idx
            self.frame_b_raw = proc
        self.lbl_slot_msg.setText(f"匹配信息 ↔ Frame A [{self.frame_a_idx}] ⚡ Frame B [{self.frame_b_idx}]")

    # ====================== 9. 高阶透视拉伸重组与算法运算触发 ======================
    def warp_perspective_pipeline(self, raw_src):
        src_pts = np.float32([[pt.x(), pt.y()] for pt in self.view_viewport.points])
        w_t = np.hypot(src_pts[0][0] - src_pts[1][0], src_pts[0][1] - src_pts[1][1])
        w_b = np.hypot(src_pts[2][0] - src_pts[3][0], src_pts[2][1] - src_pts[3][1])
        max_w = int(max(w_t, w_b))
        h_l = np.hypot(src_pts[0][0] - src_pts[3][0], src_pts[0][1] - src_pts[3][1])
        h_r = np.hypot(src_pts[1][0] - src_pts[2][0], src_pts[1][1] - src_pts[2][1])
        max_h = int(max(h_l, h_r))

        ratio_preset = self.cb_ratio.currentText()
        if ratio_preset != "自由无约束":
            nw, nh = map(int, ratio_preset.split(':'))
            val = nw / nh
            if max_w / max_h > val:
                max_w = int(max_h * val)
            else:
                max_h = int(max_w / val)

        max_w, max_h = max(16, max_w), max(16, max_h)
        dst_pts = np.float32([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(raw_src, M, (max_w, max_h), flags=cv2.INTER_LINEAR)

    def trigger_raft_pipeline(self):
        if self.frame_a_raw is None or self.frame_b_raw is None:
            self.statusBar().showMessage("流场分析拒绝：请先在主轴上提取锁定有效的 A/B 对齐帧组！")
            return

        self.btn_run.setEnabled(False)
        self.statusBar().showMessage("正在启动 RAFT 并行计算管道...")

        wa = self.warp_perspective_pipeline(self.frame_a_raw)
        wb = self.warp_perspective_pipeline(self.frame_b_raw)

        cfg = {
            'model': self.cb_model.currentText(),
            'enhance': self.chk_clahe.isChecked(),
            'threshold': self.sb_thresh.value()
        }

        # 【v0.3】代际递增，旧管道结果将被自动丢弃
        self._pipeline_generation += 1
        gen = self._pipeline_generation

        self._active_worker = RaftPipelineWorker(gen, wa, wb, cfg)
        self._active_worker.msg_signal.connect(lambda m: self.statusBar().showMessage(m))
        self._active_worker.done_signal.connect(self.on_pipeline_completed)
        self._active_worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self._active_worker.start()

    def on_pipeline_completed(self, gen, u, v, mag, fa, fb):
        """【v0.3】代际校验 —— 丢弃旧视频遗留的过期结果"""
        if gen != self._pipeline_generation:
            # 旧管道结果，直接忽略
            return

        fps = self.sb_fps.value()
        ppm = self.sb_ppm.value()
        dt = abs(self.frame_b_idx - self.frame_a_idx) / max(fps, 1.0)

        u_p = u / dt / ppm
        v_p = v / dt / ppm
        mag_p = mag / dt / ppm

        self.result_window = ResultWindow(self, self.is_dark_mode)
        self.result_window.canvas.render_plots(u_p, v_p, mag_p, fb, self.is_dark_mode)
        self.result_window.show()

        self.statusBar().showMessage(f"RAFT 定量解算圆满完成。流体速度峰值: {mag_p.max():.2f} m/s")

if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
