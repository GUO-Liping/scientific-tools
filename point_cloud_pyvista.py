#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 + PyVista 点云/网格文件转换器（带输出路径选择）
支持输入: .xyz, .csv, .vtk, .stl, .ply, .obj
支持输出: .stl, .vtk, .vtp, .ply, .obj
"""

import sys
import os
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QCheckBox, QLabel, QLineEdit, QMessageBox
)

class MeshConverterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("点云/网格文件转换器")
        self.resize(900, 650)

        # 主布局
        layout = QVBoxLayout()
        self.setLayout(layout)

        # PyVista 可视化窗口
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # 文件显示
        self.label_file = QLabel("未选择文件")
        layout.addWidget(self.label_file)

        # 打开输入文件按钮
        btn_open = QPushButton("选择输入文件")
        btn_open.clicked.connect(self.load_file)
        layout.addWidget(btn_open)

        # 输出文件路径输入框 + 选择按钮
        self.line_out = QLineEdit()
        self.line_out.setPlaceholderText("输出文件路径（可选）")
        layout.addWidget(self.line_out)

        btn_out = QPushButton("选择输出文件路径")
        btn_out.clicked.connect(self.select_output_path)
        layout.addWidget(btn_out)

        # Delaunay 曲面生成选项
        self.checkbox_delaunay = QCheckBox("对点云生成曲面 (Delaunay三角剖分)")
        layout.addWidget(self.checkbox_delaunay)

        # 转换显示按钮
        btn_convert = QPushButton("转换并显示")
        btn_convert.clicked.connect(self.convert_and_show)
        layout.addWidget(btn_convert)

        self.input_file = None
        self.mesh = None

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择输入文件",
            "", 
            "Mesh Files (*.xyz *.csv *.vtk *.vtp *.stl *.ply *.obj);;All Files (*)"
        )
        if file_path:
            self.input_file = file_path
            self.label_file.setText(f"已选择文件: {os.path.basename(file_path)}")
            self.mesh = self.load_mesh(file_path)
            self.plotter.clear()
            self.plotter.add_mesh(self.mesh, show_edges=True)
            self.plotter.reset_camera()
            self.plotter.render()

    def select_output_path(self):
        if self.input_file:
            default_name = os.path.splitext(os.path.basename(self.input_file))[0] + "_converted.vtk"
        else:
            default_name = "converted.vtk"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件", default_name,
            "Mesh Files (*.stl *.vtk *.vtp *.ply *.obj)"
        )
        if out_path:
            self.line_out.setText(out_path)

    def load_mesh(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".vtk", ".vtp", ".stl", ".ply", ".obj"]:
            mesh = pv.read(filename)
        elif ext in [".xyz", ".txt", ".csv"]:
            data = np.loadtxt(filename)
            if data.shape[1] < 3:
                raise ValueError("输入文件必须至少包含3列 (x y z)")
            mesh = pv.PolyData(data[:, :3])
        else:
            raise ValueError(f"不支持的文件类型: {ext}")
        return mesh

    def convert_and_show(self):
        if self.mesh is None:
            QMessageBox.warning(self, "警告", "请先选择输入文件")
            return

        mesh_to_save = self.mesh.copy()
        if self.checkbox_delaunay.isChecked():
            try:
                mesh_to_save = mesh_to_save.delaunay_2d()
            except Exception as e:
                QMessageBox.warning(self, "警告", f"Delaunay 生成失败: {e}")

        output_file = self.line_out.text().strip()
        if not output_file:
            output_file = os.path.splitext(self.input_file)[0] + "_converted.vtk"

        try:
            mesh_to_save.save(output_file)
            QMessageBox.information(self, "完成", f"文件已导出: {output_file}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件失败: {e}")
            return

        # 可视化
        self.plotter.clear()
        self.plotter.add_mesh(mesh_to_save, show_edges=True)
        self.plotter.reset_camera()
        self.plotter.render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeshConverterApp()
    window.show()
    sys.exit(app.exec_())
