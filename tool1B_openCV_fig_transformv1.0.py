# 该程序用于单张图片的自由变换裁剪变换
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QPointF, QLineF
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QFileDialog, QMessageBox, QGroupBox, QGridLayout,
                             QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
                             QGraphicsPolygonItem, QGraphicsLineItem)
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPolygonF, QBrush, QPainter

# ==================== 1. 自定义高级画布交互组件 ====================
class ControlPoint(QGraphicsEllipseItem):
    """可拖拽的控制点（坐标基于原图绝对像素）"""
    def __init__(self, x, y, idx, callback):
        super().__init__(-12, -12, 24, 24) 
        self.setPos(x, y)
        self.idx = idx
        self.callback = callback
        
        self.setBrush(QBrush(QColor(220, 53, 69))) # 默认红色
        self.setPen(QPen(QColor(255, 255, 255), 2)) # 白色边框
        
        # 激活拖拽和几何改变通知
        self.setFlags(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange and isinstance(value, QPointF):
            parent = self.parentItem()
            if parent:
                rect = parent.pixmap().rect()
                x = max(0.0, min(float(rect.width() - 1), value.x()))
                y = max(0.0, min(float(rect.height() - 1), value.y()))
                clamped_pos = QPointF(x, y)
                self.callback(self.idx, clamped_pos)
                return clamped_pos
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self.setBrush(QBrush(QColor(255, 193, 7))) # 按下变黄
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setBrush(QBrush(QColor(220, 53, 69))) # 释放恢复红色
        super().mouseReleaseEvent(event)


# ==================== 2. 主业务 GUI 窗口 ====================
class PerspectiveClipperSubstation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photoshop级 自由变换透视裁剪工作站 (双信息面板工业版)")
        self.setGeometry(100, 100, 1450, 950)
        self.setStyleSheet("QMainWindow { background-color: #2d2d2d; }")

        # --- 核心数据状态机 ---
        self.img_orig = None          # 原始 OpenCV BGR 图像
        self.points = []              # 存储绝对坐标 QPointF(x, y)，基于原图真实像素
        self.point_items = []         # 画布上的控制点对象实例
        
        # 画布图元引用
        self.bg_item = None           # 底图图元
        self.polygon_item = None      # 蓝色四边形
        self.grid_items = []          # 内部黄色网格线

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # ==================== 2.1 顶部工具栏 ====================
        top_bar_layout = QHBoxLayout()
        
        self.btn_open = QPushButton("📂 打开图片")
        self.btn_open.clicked.connect(self.open_image)
        
        self.btn_load_json = QPushButton("📄 导入 Labelme JSON")
        self.btn_load_json.clicked.connect(self.load_labelme_json)
        
        self.btn_save_json = QPushButton("💾 导出新 JSON 坐标")
        self.btn_save_json.clicked.connect(self.save_new_json)
        
        self.btn_crop = QPushButton("✂️ 执行裁剪并保存")
        self.btn_crop.clicked.connect(self.export_cropped_image)
        
        self.btn_clear = QPushButton("🔄 重置清空")
        self.btn_clear.clicked.connect(self.clear_points)

        buttons = [self.btn_open, self.btn_load_json, self.btn_save_json, self.btn_crop, self.btn_clear]
        colors = ["#4a4a4a", "#4a4a4a", "#28a745", "#007bff", "#dc3545"]
        for btn, color in zip(buttons, colors):
            btn.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; padding: 8px 15px; font-size: 13px; border-radius: 3px; font-weight: bold; }} QPushButton:hover {{ opacity: 0.9; }}")
            top_bar_layout.addWidget(btn)
        
        top_bar_layout.addStretch()
        main_layout.addLayout(top_bar_layout)

        # ==================== 2.2 核心双栏布局 ====================
        workspace_layout = QHBoxLayout()

        # 左侧图像画布容器
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(QColor(18, 18, 18)))
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing) 
        self.view.mousePressEvent = self.on_canvas_click         
        workspace_layout.addWidget(self.view, stretch=3)

        # 右侧参数控制与多维数据展示面板
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1. 原始图片信息展示面板（加载后立即显示）
        orig_info_group = QGroupBox("原始图片多维元数据")
        orig_info_group.setStyleSheet("QGroupBox { color: #00ff2b; font-weight: bold; border: 1px solid #00ff2b; padding: 10px; }")
        orig_info_layout = QVBoxLayout(orig_info_group)
        self.lbl_orig_info = QLabel("等待加载图片...")
        self.lbl_orig_info.setStyleSheet("""
            QLabel { 
                color: #00ff2b; 
                font-family: 'Consolas', 'Courier New'; 
                font-size: 12px; 
                line-height: 150%;
            }
        """)
        orig_info_layout.addWidget(self.lbl_orig_info)
        right_panel.addWidget(orig_info_group)

        # 2. 手动 4 点坐标输入表单
        form_group = QGroupBox("手动输入参数裁剪 (任意 4 点绝对坐标)")
        form_group.setStyleSheet("""
            QGroupBox { color: white; font-weight: bold; border: 1px solid #555; padding: 8px; } 
            QGroupBox::title { subcontrol-origin: margin; left: 10px; } 
            QLabel { color: #ddd; font-size: 11px; } 
            QLineEdit { background-color: #3d3d3d; color: white; border: 1px solid #666; padding: 2px; font-family: Consolas; }
        """)
        form_grid = QGridLayout(form_group)
        
        self.coord_entries = [] 
        for i in range(4):
            lbl_x = QLabel(f"P{i+1} X:")
            edit_x = QLineEdit()
            edit_x.setFixedWidth(65)
            
            lbl_y = QLabel(f"P{i+1} Y:")
            edit_y = QLineEdit()
            edit_y.setFixedWidth(65)
            
            form_grid.addWidget(lbl_x, i, 0)
            form_grid.addWidget(edit_x, i, 1)
            form_grid.addWidget(lbl_y, i, 2)
            form_grid.addWidget(edit_y, i, 3)
            self.coord_entries.append((edit_x, edit_y))

        self.btn_apply_pts = QPushButton("应用点位参数")
        self.btn_apply_pts.setStyleSheet("QPushButton { background-color: #6c757d; color: white; border: none; padding: 6px; font-weight: bold; margin-top: 5px; }")
        self.btn_apply_pts.clicked.connect(self.apply_manual_points)
        form_grid.addWidget(self.btn_apply_pts, 4, 0, 1, 4)
        right_panel.addWidget(form_group)

        # 3. 坐标捕获状态展示组
        coord_group = QGroupBox("实时捕获 4 点绝对像素坐标")
        coord_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #555; padding: 10px; }")
        coord_layout = QVBoxLayout(coord_group)
        self.lbl_coords = []
        for i in range(4):
            lbl = QLabel(f"点 {i+1}: 未指定")
            lbl.setStyleSheet("color: #aaa; font-family: Consolas; font-size: 12px;")
            coord_layout.addWidget(lbl)
            self.lbl_coords.append(lbl)
        right_panel.addWidget(coord_group)

        # 4. 实时裁剪预览与裁剪元数据组
        preview_group = QGroupBox("实时透视裁剪预览")
        preview_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #555; padding: 10px; }")
        preview_layout = QVBoxLayout(preview_group)
        
        self.lbl_preview = QLabel("等待标注满4个点后自动生成预览...")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setFixedSize(360, 240)
        self.lbl_preview.setStyleSheet("background-color: #121212; color: #666; border: 1px solid #444;")
        preview_layout.addWidget(self.lbl_preview)
        
        # 裁剪图像尺寸等元数据（维度与原图信息完全对齐）
        self.info_panel = QLabel("裁剪尺寸信息:\n- 宽度(W): -- px\n- 高度(H): -- px\n- 通道数: --\n- 宽高比: --\n- 总像素: -- MP")
        self.info_panel.setStyleSheet("""
            QLabel { 
                background-color: #1e1e1e; 
                color: #00ffff; 
                font-family: 'Consolas', 'Courier New'; 
                font-size: 12px; 
                border: 1px solid #333; 
                padding: 8px; 
                margin-top: 5px;
                line-height: 150%;
            }
        """)
        preview_layout.addWidget(self.info_panel)
        right_panel.addWidget(preview_group)

        workspace_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(workspace_layout)

    # ==================== 3. 响应式与画布图像逻辑 ====================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.auto_fit_image()

    def auto_fit_image(self):
        if self.bg_item is not None:
            self.view.fitInView(self.bg_item, Qt.AspectRatioMode.KeepAspectRatio)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not file_path: return
        
        # 兼容中文路径的 OpenCV 读取
        self.img_orig = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.img_orig is None:
            QMessageBox.critical(self, "错误", "无法解码该图像！")
            return
            
        self.current_img_path = Path(file_path)
        self.clear_points()
        
        h_orig, w_orig = self.img_orig.shape[:2]
        channels = self.img_orig.shape[2] if len(self.img_orig.shape) > 2 else 1
        
        # 🚀 新增功能：立即解析并渲染原始图片的各项核心指标
        orig_file_size = self.current_img_path.stat().st_size
        size_str = f"{orig_file_size / (1024*1024):.2f} MB" if orig_file_size >= 1024*1024 else f"{orig_file_size / 1024:.2f} KB"
        aspect_ratio_orig = w_orig / h_orig
        mega_pixels_orig = (w_orig * h_orig) / 1000000.0
        
        orig_info_text = (
            f"原始图像信息:\n"
            f"- 图片名称: {self.current_img_path.name}\n"
            f"- 宽度(W): {w_orig} px\n"
            f"- 高度(H): {h_orig} px\n"
            f"- 通道数: {channels}\n"
            f"- 宽高比: {aspect_ratio_orig:.3f} (1:{aspect_ratio_orig:.2f})\n"
            f"- 总像素: {mega_pixels_orig:.2f} MP\n"
            f"- 文件大小: {size_str}"
        )
        self.lbl_orig_info.setText(orig_info_text)

        # 渲染底层画布图片
        img_rgb = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w_orig, h_orig, w_orig * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.scene.clear()
        self.scene.setSceneRect(0, 0, w_orig, h_orig) 
        
        self.bg_item = self.scene.addPixmap(pixmap)
        self.bg_item.setPos(0, 0)
        
        self.polygon_item = None
        self.grid_items = []
        self.point_items = []

        self.auto_fit_image()

    def on_canvas_click(self, event):
        if self.img_orig is None or len(self.points) >= 4: 
            QGraphicsView.mousePressEvent(self.view, event)
            return
        
        scene_pos = self.view.mapToScene(event.pos())
        
        if self.scene.itemAt(scene_pos, self.view.transform()) in self.point_items:
            QGraphicsView.mousePressEvent(self.view, event)
            return

        h_orig, w_orig = self.img_orig.shape[:2]
        if 0 <= scene_pos.x() < w_orig and 0 <= scene_pos.y() < h_orig:
            idx = len(self.points)
            self.points.append(scene_pos)
            
            pt_item = ControlPoint(scene_pos.x(), scene_pos.y(), idx, self.on_point_dragged)
            pt_item.setParentItem(self.bg_item)
            self.point_items.append(pt_item)
            
            self.update_visual_elements()

    def on_point_dragged(self, idx, new_clamped_pos):
        self.points[idx] = new_clamped_pos
        self.update_visual_elements()

    def order_points_clockwise(self, pts):
        pts_np = np.array([[p.x(), p.y()] for p in pts], dtype=np.float32)
        s = pts_np.sum(axis=1)
        diff = np.diff(pts_np, axis=1)
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = pts_np[np.argmin(s)]
        rect[2] = pts_np[np.argmax(s)]
        rect[1] = pts_np[np.argmin(diff)]
        rect[3] = pts_np[np.argmax(diff)]
        return rect

    def update_visual_elements(self):
        if self.img_orig is None: return

        # 1. 刷新文本及输入表单反向同步
        for i in range(4):
            if i < len(self.points):
                px, py = int(self.points[i].x()), int(self.points[i].y())
                self.lbl_coords[i].setText(f"点 {i+1} (红): X={px}, Y={py}")
                self.lbl_coords[i].setStyleSheet("color: #28a745; font-family: Consolas;")
                
                if not self.coord_entries[i][0].hasFocus(): self.coord_entries[i][0].setText(str(px))
                if not self.coord_entries[i][1].hasFocus(): self.coord_entries[i][1].setText(str(py))
            else:
                self.lbl_coords[i].setText(f"点 {i+1}: 未指定")
                self.lbl_coords[i].setStyleSheet("color: #aaa; font-family: Consolas;")

        # 2. 清理旧图形元件
        if self.polygon_item: self.scene.removeItem(self.polygon_item); self.polygon_item = None
        for item in self.grid_items: self.scene.removeItem(item)
        self.grid_items.clear()

        # 3. 满 4 个点时绘制密集的 9x9 畸变网格
        if len(self.points) == 4:
            rect = self.order_points_clockwise(self.points)
            tl = QPointF(rect[0][0], rect[0][1])
            tr = QPointF(rect[1][0], rect[1][1])
            br = QPointF(rect[2][0], rect[2][1])
            bl = QPointF(rect[3][0], rect[3][1])

            poly = QPolygonF([tl, tr, br, bl])
            self.polygon_item = QGraphicsPolygonItem(poly, self.bg_item)
            self.polygon_item.setPen(QPen(QColor(0, 123, 255), 4)) 
            
            grid_count = 9
            for i in range(1, grid_count):
                alpha = i / grid_count
                
                left_pt = tl * (1 - alpha) + bl * alpha
                right_pt = tr * (1 - alpha) + br * alpha
                line_h = QGraphicsLineItem(QLineF(left_pt, right_pt), self.bg_item)
                line_h.setPen(QPen(QColor(0, 255, 255), 1.5, Qt.PenStyle.DashLine))
                self.grid_items.append(line_h)

                top_pt = tl * (1 - alpha) + tr * alpha
                bottom_pt = bl * (1 - alpha) + br * alpha
                line_v = QGraphicsLineItem(QLineF(top_pt, bottom_pt), self.bg_item)
                line_v.setPen(QPen(QColor(0, 255, 255), 1.5, Qt.PenStyle.DashLine))
                self.grid_items.append(line_v)

            self.generate_live_preview(rect)
        else:
            self.info_panel.setText("裁剪尺寸信息:\n- 宽度(W): -- px\n- 高度(H): -- px\n- 通道数: --\n- 宽高比: --\n- 总像素: -- MP")

    def generate_live_preview(self, rect):
        """生成预览切片并计算更新高精度元数据指标信息"""
        (tl, tr, br, bl) = rect
        w1 = np.linalg.norm(br - bl)
        w2 = np.linalg.norm(tr - tl)
        max_w = int(max(w1, w2))
        h1 = np.linalg.norm(tr - br)
        h2 = np.linalg.norm(tl - bl)
        max_h = int(max(h1, h2))
        max_w, max_h = max(10, max_w), max(10, max_h)

        # 透视拉伸变换
        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.img_orig, M, (max_w, max_h))

        # 解析裁剪后图像的通道数
        channels_warped = warped.shape[2] if len(warped.shape) > 2 else 1
        aspect_ratio = max_w / max_h
        mega_pixels = (max_w * max_h) / 1000000.0
        
        # 🚀 刷新右侧裁剪后图像的状态栏面板（维度格式与原图面板完美对齐）
        info_text = (
            f"自由变换裁剪信息:\n"
            f"- 宽度(W): {max_w} px\n"
            f"- 高度(H): {max_h} px\n"
            f"- 通道数: {channels_warped}\n"
            f"- 宽高比: {aspect_ratio:.3f} (1:{aspect_ratio:.2f})\n"
            f"- 总像素: {mega_pixels:.2f} MP"
        )
        self.info_panel.setText(info_text)

        # 缩放渲染至预览区
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        wh, ww = warped_rgb.shape[:2]
        scale = min(360 / ww, 240 / wh)
        nw, nh = max(1, int(ww * scale)), max(1, int(wh * scale))
        
        warped_resized = cv2.resize(warped_rgb, (nw, nh))
        qimg = QImage(warped_resized.data, nw, nh, nw * 3, QImage.Format.Format_RGB888)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qimg))

    def apply_manual_points(self):
        if self.img_orig is None:
            QMessageBox.warning(self, "警告", "请先加载图片底图！")
            return
        try:
            parsed_pts = []
            for i in range(4):
                x_str = self.coord_entries[i][0].text().strip()
                y_str = self.coord_entries[i][1].text().strip()
                if not x_str or not y_str:
                    raise ValueError("未填满四个控制点")
                parsed_pts.append(QPointF(float(x_str), float(y_str)))
            
            self.clear_points()
            self.points = parsed_pts
            
            for i, pt in enumerate(self.points):
                pt_item = ControlPoint(pt.x(), pt.y(), i, self.on_point_dragged)
                pt_item.setParentItem(self.bg_item)
                self.point_items.append(pt_item)
                
            self.update_visual_elements()
        except Exception:
            QMessageBox.critical(self, "错误", "四个坐标点输入不完整或包含非法字符，请输入纯数字！")

    def load_labelme_json(self):
        if self.img_orig is None:
            QMessageBox.warning(self, "警告", "请先加载图片！")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "导入 Labelme JSON", "", "JSON (*.json)")
        if not file_path: return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts = data["shapes"][0]["points"]
            if len(pts) != 4:
                QMessageBox.critical(self, "错误", "JSON 内多边形顶点不等于 4！")
                return
            
            self.clear_points()
            for i, pt in enumerate(pts):
                ox, oy = float(pt[0]), float(pt[1])
                self.points.append(QPointF(ox, oy))
                pt_item = ControlPoint(ox, oy, i, self.on_point_dragged)
                pt_item.setParentItem(self.bg_item)
                self.point_items.append(pt_item)
                
            self.update_visual_elements()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"解析失败:\n{e}")

    def save_new_json(self):
        if len(self.points) != 4:
            QMessageBox.warning(self, "提示", "请先生成 4 个控制点位置！")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存新坐标配置", "", "JSON (*.json)")
        if not file_path: return
        try:
            output = {
                "version": "PyQt6_Responsive_Clipper_v4.0",
                "shapes": [{
                    "label": "crop_area",
                    "points": [[int(pt.x()), int(pt.y())] for pt in self.points],
                    "shape_type": "polygon"
                }],
                "imageWidth": self.img_orig.shape[1],
                "imageHeight": self.img_orig.shape[0]
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4)
            QMessageBox.information(self, "成功", "新 JSON 配置文件导出成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def export_cropped_image(self):
        if len(self.points) != 4: return
        rect = self.order_points_clockwise(self.points)
        (tl, tr, br, bl) = rect
        w1 = np.linalg.norm(br - bl)
        w2 = np.linalg.norm(tr - tl)
        max_w = int(max(w1, w2))
        h1 = np.linalg.norm(tr - br)
        h2 = np.linalg.norm(tl - bl)
        max_h = int(max(h1, h2))
        max_w, max_h = max(10, max_w), max(10, max_h)

        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.img_orig, M, (max_w, max_h))

        file_path, _ = QFileDialog.getSaveFileName(self, "保存切片图片", "", "JPEG (*.jpg);;PNG (*.png)")
        if not file_path: return
        try:
            ext = Path(file_path).suffix
            cv2.imencode(ext, warped)[1].tofile(file_path)
            QMessageBox.information(self, "成功", "高质量切片图像已成功保存！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def clear_points(self):
        self.points.clear()
        self.point_items.clear()
        self.grid_items.clear()
        self.polygon_item = None
        if self.bg_item is not None:
            for item in self.bg_item.childItems():
                self.scene.removeItem(item)
        self.lbl_preview.clear()
        self.lbl_preview.setText("等待标注满4个点后自动生成预览...")
        self.info_panel.setText("裁剪尺寸信息:\n- 宽度(W): -- px\n- 高度(H): -- px\n- 通道数: --\n- 宽高比: --\n- 总像素: -- MP")
        for x_edit, y_edit in self.coord_entries:
            x_edit.clear()
            y_edit.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    station = PerspectiveClipperSubstation()
    station.show()
    sys.exit(app.exec())