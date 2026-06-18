# -*- coding: utf-8 -*-
"""
软件名称：基于地震信号的推移质通量自动反演与可视化分析软件
版本号：V1.0
当前功能：
1. 项目管理页面：保存项目基本信息。
2. 地震预处理页面：读取 SAC，去仪器响应，保存处理后 SAC。
3. 时频分析页面：读取处理后 SAC，计算 Welch PSD，输出 PSD、时频图和 30–80 Hz 线性能量 CSV。
4. 泥沙粒径参数页面：输入 D16/D50/D84 或导入粒径分布，保存泥沙参数和粒径分布。
5. 水力参数页面：导入河道断面和水位序列，计算 H<sub>eff</sub>、W<sub>eff</sub> 以及代表断面的有效运动粒径范围。
6. 反演计算页面：自动调用前序模块输出，在内存中计算单位通量理论能量表，并反演推移质通量。
7. 简化台站配置，只保存长期固定台站属性。
8. 初步封装 StationProfile / RunTask / SystemSettingsCore，区分固定配置与本次任务输入。
"""

import sys
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd


def safe_trapz(y, x=None, dx=1.0, axis=-1):
    """兼容 NumPy 1.x/2.x 的梯形积分函数。

    NumPy 2.x 推荐使用 np.trapezoid；旧环境则回退到 np.trapz。
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x, dx=dx, axis=axis)
    y = np.asarray(y, dtype=float)
    if x is None:
        return np.sum((np.take(y, range(1, y.shape[axis]), axis=axis) + np.take(y, range(0, y.shape[axis]-1), axis=axis)) * 0.5 * dx, axis=axis)
    x = np.asarray(x, dtype=float)
    return np.trapezoid(y, x=x, axis=axis)


from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QCheckBox,
    QScrollArea,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QProgressBar,
)

from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

try:
    from scipy.special import gamma as gamma_func
except Exception:
    from math import gamma as gamma_func

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_json_safely(path):
    """读取 JSON；文件不存在或格式错误时返回空字典。"""
    try:
        path = Path(path)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json_safely(path, data):
    """保存 JSON，并自动创建父文件夹。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def deep_update_dict(base, update):
    """递归更新字典，用于合并 project_config.json。"""
    base = dict(base or {})
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def widget_text(widget, default=""):
    """安全读取 QLineEdit 等控件文本。

    注意：如果控件存在但文本为空，返回 default，而不是返回空字符串。
    这样“从当前模块填充固定信息”时，空的当前模块不会把台站配置页已有内容清空。
    """
    try:
        if widget is not None and hasattr(widget, "text"):
            text = widget.text().strip()
            return text if text else default
    except Exception:
        pass
    return default


def set_widget_text(widget, value):
    """安全写入 QLineEdit 文本。"""
    try:
        if widget is not None and hasattr(widget, "setText") and value is not None:
            widget.setText(str(value))
    except Exception:
        pass


def set_widget_text_if_nonempty(widget, value):
    """只有 value 非空时才写入，避免空配置覆盖已有模块路径。"""
    try:
        if value is None:
            return
        text = str(value).strip()
        if widget is not None and hasattr(widget, "setText") and text:
            widget.setText(text)
    except Exception:
        pass


def project_dir_from_page(project_page):
    """从项目页面读取项目目录。"""
    if project_page is None:
        return None
    text = widget_text(getattr(project_page, "project_dir", None))
    if not text:
        return None
    return Path(text)


def project_config_path_from_page(project_page):
    """获取 project_config.json 路径。"""
    project_dir = project_dir_from_page(project_page)
    if project_dir is None:
        return None
    return project_dir / "project_config.json"


def load_project_config_from_page(project_page):
    path = project_config_path_from_page(project_page)
    if path is None:
        return {}
    return load_json_safely(path)


def save_project_config_from_page(project_page, patch):
    path = project_config_path_from_page(project_page)
    if path is None:
        raise ValueError("请先在项目管理页面设置项目文件夹。")
    old = load_json_safely(path)
    new = deep_update_dict(old, patch)
    save_json_safely(path, new)
    return path


@dataclass
class BedloadSedimentParams:
    """沉积物物理参数。"""
    rho_s: float = 2700.0
    rho_f: float = 1000.0
    g: float = 9.81
    CSF: float = 0.8
    P: float = 3.5
    nu: float = 1e-6
    C1: float = 2.0 / 3.0


@dataclass
class BedloadSeismicParams:
    """地震波传播与冲击方向参数。"""
    v0: float = 2206.0
    z0: float = 1000.0
    f0: float = 1.0
    a: float = 0.272
    Q0: float = 20.0
    eta: float = 0.0
    phi: float = 0.0
    eb: float = 0.5
    fx: float = 0.146
    fy: float = 0.146
    fz: float = 0.539
    Nzz: float = 0.352

    @property
    def zeta(self):
        return self.a / (1.0 - self.a)

    @property
    def vc0(self):
        return (self.v0 * gamma_func(1.0 + self.a) / (2.0 * np.pi * self.z0 * self.f0) ** self.a) ** (1.0 / (1.0 - self.a))


@dataclass
class StationProfile:
    """长期固定的台站配置。

    只保存一个台站长期不变或很少变化的信息：台站身份、固定断面、固定级配、
    台站-河道距离和地震传播参数。原始地震数据、水位序列、输出目录等属于
    单次任务输入，不放入 StationProfile。
    """
    station_id: str = ""
    river_name: str = ""
    station_name: str = ""
    cross_section_file: str = ""
    grain_distribution_file: str = ""
    slope: float = 0.01
    r0_m: float = 17.0
    seismic_params: Optional[BedloadSeismicParams] = None

    @staticmethod
    def _float_value(value, default):
        try:
            if value is None or str(value).strip() == "":
                return float(default)
            return float(str(value).strip())
        except Exception:
            return float(default)

    @classmethod
    def from_dict(cls, data):
        data = data or {}

        # 兼容旧版 station_config.json：station / paths / seismic_params 三段式。
        station = data.get("station", {}) if isinstance(data.get("station", {}), dict) else {}
        fixed = data.get("fixed_files", {}) if isinstance(data.get("fixed_files", {}), dict) else {}
        paths = data.get("paths", {}) if isinstance(data.get("paths", {}), dict) else {}
        seismic = data.get("seismic_params", {}) if isinstance(data.get("seismic_params", {}), dict) else {}

        sp = BedloadSeismicParams(
            v0=cls._float_value(seismic.get("v0"), 2206.0),
            z0=cls._float_value(seismic.get("z0"), 1000.0),
            f0=cls._float_value(seismic.get("f0"), 1.0),
            a=cls._float_value(seismic.get("a"), 0.272),
            Q0=cls._float_value(seismic.get("Q0"), 20.0),
            eta=cls._float_value(seismic.get("eta"), 0.0),
            phi=cls._float_value(seismic.get("phi"), 0.0),
            eb=cls._float_value(seismic.get("eb"), 0.5),
            fx=cls._float_value(seismic.get("fx"), 0.146),
            fy=cls._float_value(seismic.get("fy"), 0.146),
            fz=cls._float_value(seismic.get("fz"), 0.539),
            Nzz=cls._float_value(seismic.get("Nzz"), 0.352),
        )

        return cls(
            station_id=str(data.get("station_id", station.get("station_id", ""))).strip(),
            river_name=str(data.get("river_name", station.get("river_name", ""))).strip(),
            station_name=str(data.get("station_name", station.get("station_name", ""))).strip(),
            cross_section_file=str(data.get("cross_section_file", fixed.get("cross_section_file", paths.get("cross_section_file", "")))).strip(),
            grain_distribution_file=str(data.get("grain_distribution_file", fixed.get("grain_distribution_file", paths.get("grain_distribution_file", "")))).strip(),
            slope=cls._float_value(data.get("slope", station.get("slope", 0.01)), 0.01),
            r0_m=cls._float_value(data.get("r0_m", station.get("r0_m", station.get("r0", 17.0))), 17.0),
            seismic_params=sp,
        )

    def to_dict(self):
        sp = self.seismic_params if self.seismic_params is not None else BedloadSeismicParams()
        return {
            "station_id": self.station_id,
            "river_name": self.river_name,
            "station_name": self.station_name,
            "cross_section_file": self.cross_section_file,
            "grain_distribution_file": self.grain_distribution_file,
            "slope": self.slope,
            "r0_m": self.r0_m,
            "seismic_params": {
                "v0": sp.v0,
                "z0": sp.z0,
                "f0": sp.f0,
                "a": sp.a,
                "Q0": sp.Q0,
                "eta": sp.eta,
                "phi": sp.phi,
                "eb": sp.eb,
                "fx": sp.fx,
                "fy": sp.fy,
                "fz": sp.fz,
                "Nzz": sp.Nzz,
            },
        }


@dataclass
class RunTask:
    """单次运行任务配置。

    这些信息随每次实时或批量计算改变，不写入台站固定配置。
    """
    station_id: str = ""
    seismic_input: str = ""
    water_level_file: str = ""
    output_dir: str = ""
    start_time: str = ""
    end_time: str = ""
    mode: str = "single"


@dataclass
class SystemSettingsCore:
    """软件级全局默认参数。"""
    rho_s_default: float = 2700.0
    rho_w_default: float = 1000.0
    theta_c_default: float = 0.045
    energy_fmin: float = 30.0
    energy_fmax: float = 80.0
    energy_nfreq: int = 51
    outlier_window_min: float = 60.0
    outlier_factor: float = 5.0
    cache_enabled: bool = True
    skip_existing: bool = True
    log_dir: str = ""


class SaltationBedloadModel:
    """
    基于盐跃颗粒冲击的推移质地震 PSD 正演模型。

    注意：qb 为单位宽体积通量，单位 m²/s。
    如果希望使用 1 kg m⁻¹ s⁻¹ 的单位质量通量，需要传入 qb = 1 / rho_s。
    """

    def __init__(self, sediment_params=None, seismic_params=None):
        self.sediment_params = sediment_params if sediment_params is not None else BedloadSedimentParams()
        self.seismic_params = seismic_params if seismic_params is not None else BedloadSeismicParams()

    def estimate_critical_shear(self, theta, method="gimbert"):
        theta = float(theta)
        if theta <= 0:
            return 0.045
        if method == "gimbert":
            dummy = 0.407 * np.log(142.0 * theta)
            return np.exp(2.59e-2 * dummy**4 + 8.94e-2 * dummy**3 + 0.142 * dummy**2 + 0.41 * dummy - 3.14)
        if method == "lamb":
            return 0.15 * theta**0.25
        return 0.045

    def estimate_drag_coeff(self, D):
        D = np.asarray(D, dtype=float)
        R = (self.sediment_params.rho_s - self.sediment_params.rho_f) / self.sediment_params.rho_f
        D = np.maximum(D, 1e-9)
        D_star = (R * self.sediment_params.g * D**3) / self.sediment_params.nu**2
        D_star = np.log10(np.maximum(D_star, 1e-30))

        R1 = (-3.76715 + 1.92944 * D_star - 0.09815 * D_star**2 - 0.00575 * D_star**3 + 0.00056 * D_star**4)
        R2 = (
            np.log10(1.0 - ((1.0 - self.sediment_params.CSF) / 0.85))
            - (1.0 - self.sediment_params.CSF) ** 2.3 * np.tanh(D_star - 4.6)
            + 0.3 * (0.5 - self.sediment_params.CSF) * (1.0 - self.sediment_params.CSF) ** 2 * (D_star - 4.6)
        )
        R3 = (0.65 - ((self.sediment_params.CSF / 2.83) * np.tanh(D_star - 4.6))) ** (
            1.0 + ((3.5 - self.sediment_params.P) / 2.5)
        )
        W_star = R3 * 10.0 ** (R1 + R2)
        w_s = (W_star * R * self.sediment_params.g * self.sediment_params.nu) ** (1.0 / 3.0)
        cD = (4.0 / 3.0) * (R * self.sediment_params.g * D) / np.maximum(w_s**2, 1e-30)
        return cD

    def calculate_bedload_parameters(self, D, H, theta, tau_c):
        D = np.asarray(D, dtype=float)
        H = float(H)
        theta = float(theta)
        R = (self.sediment_params.rho_s - self.sediment_params.rho_f) / self.sediment_params.rho_f
        D = np.maximum(D, 1e-9)
        tau_c = np.asarray(tau_c, dtype=float)

        if H <= 0 or theta <= 0:
            zeros = np.zeros_like(D, dtype=float)
            return np.pi * D**3 / 6.0, zeros, zeros, zeros

        u_star = np.sqrt(self.sediment_params.g * H * theta)
        tau = u_star**2 / R / self.sediment_params.g / D
        transport_stage = tau / np.maximum(tau_c, 1e-12)

        ks = 3.0 * D
        U = 8.1 * u_star * (H / np.maximum(ks, 1e-12)) ** 1.6
        excess = np.maximum(transport_stage - 1.0, 0.0)
        Vp = np.pi * D**3 / 6.0
        Ub = 1.56 * np.sqrt(R * self.sediment_params.g * D) * excess**0.56
        Hb = 1.44 * D * excess**0.50
        Ub = np.minimum(np.maximum(Ub, 0.0), U)
        Hb = np.minimum(np.maximum(Hb, 0.0), H)
        return Vp, Ub, Hb, transport_stage

    def calculate_seismic_properties(self, f, r0):
        f = np.asarray(f, dtype=float)
        f = np.maximum(f, 1e-9)
        r0 = max(float(r0), 1e-9)
        sp = self.seismic_params
        vc = sp.vc0 * (f / sp.f0) ** (-sp.zeta)
        vu = vc / (1.0 + sp.zeta)
        beta = (
            2.0 * np.pi * r0 * (1.0 + sp.zeta)
            * f ** (1.0 + sp.zeta - sp.eta)
            / (sp.vc0 * sp.Q0 * sp.f0 ** (sp.zeta - sp.eta))
        )
        beta = np.maximum(beta, 1e-12)
        chi = (
            2.0 * np.log(1.0 + 1.0 / beta) * np.exp(-2.0 * beta)
            + (1.0 - np.exp(-beta)) * np.exp(-beta) * np.sqrt(2.0 * np.pi / beta)
        )
        return vc, vu, chi

    def _psd(self, f, D, H, W, theta, r0, qb, tau_c=None, clip_tau_c=False, D50=None, tau_c50=None):
        f_arr = np.atleast_1d(np.asarray(f, dtype=float))
        D_arr = np.atleast_1d(np.asarray(D, dtype=float))
        H = float(H)
        W = float(W)
        theta = float(theta)
        r0 = float(r0)
        qb = float(qb)

        if H <= 0 or W <= 0 or theta <= 0 or r0 <= 0 or qb <= 0 or D_arr.size == 0:
            if D_arr.size == 1:
                return np.zeros_like(f_arr, dtype=float)
            if f_arr.size == 1:
                return np.zeros_like(D_arr, dtype=float)
            return np.zeros((D_arr.size, f_arr.size), dtype=float)

        D_arr = np.maximum(D_arr, 1e-9)
        if D50 is None or not np.isfinite(D50) or float(D50) <= 0:
            D50 = np.nanmedian(D_arr)
        if tau_c50 is None or not np.isfinite(tau_c50) or float(tau_c50) <= 0:
            tau_c50 = self.estimate_critical_shear(theta)

        if tau_c is None:
            tau_c_arr = float(tau_c50) * (D_arr / float(D50)) ** (-0.9)
        else:
            tau_c_arr = np.asarray(tau_c, dtype=float)
            if tau_c_arr.size == 1:
                tau_c_arr = np.full_like(D_arr, float(tau_c_arr))

        if clip_tau_c:
            tau_c_arr = np.clip(tau_c_arr, 0.03, 0.06)

        Vp, Ub, Hb, transport_stage = self.calculate_bedload_parameters(D_arr, H, theta, tau_c_arr)
        valid = (transport_stage >= 1.0) & (Ub > 0) & (Hb > 0) & np.isfinite(Ub) & np.isfinite(Hb)

        vc, vu, chi = self.calculate_seismic_properties(f_arr, r0)
        psd_matrix = np.zeros((D_arr.size, f_arr.size), dtype=float)
        if not np.any(valid):
            if D_arr.size == 1:
                return psd_matrix[0, :]
            if f_arr.size == 1:
                return psd_matrix[:, 0]
            return psd_matrix

        Dv = D_arr[valid]
        Vpv = Vp[valid]
        Ubv = Ub[valid]
        Hbv = Hb[valid]
        cD = self.estimate_drag_coeff(Dv)
        spd = self.sediment_params
        m = spd.rho_s * Vpv

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            wst = np.sqrt(4.0 * (spd.rho_s - spd.rho_f) / spd.rho_f * spd.g * Dv / 3.0 / np.maximum(cD, 1e-30))
            Hb_c = 3.0 * cD * spd.rho_f * Hbv / (2.0 * spd.rho_s * Dv * np.cos(np.arctan(theta)))
            Hb_c = np.maximum(Hb_c, 0.0)
            wi = wst * np.cos(np.arctan(theta)) * np.sqrt(np.maximum(1.0 - np.exp(-Hb_c), 0.0))
            denom = 2.0 * np.log(np.exp(Hb_c / 2.0) + np.sqrt(np.maximum(np.exp(Hb_c) - 1.0, 0.0)))
            ws = Hb_c * wst * np.cos(np.arctan(theta)) / np.maximum(denom, 1e-30)
            rate = spd.C1 * W * qb * ws / np.maximum(Vpv * Ubv * Hbv, 1e-30)
            const_D = rate * (np.pi**2 * m**2 * wi**2) / spd.rho_s**2
            const_f = f_arr**3 / np.maximum(vc**3 * vu**2, 1e-30) * chi
            psd_matrix[valid, :] = const_D[:, None] * const_f[None, :]

        psd_matrix = np.where(np.isfinite(psd_matrix) & (psd_matrix > 0), psd_matrix, 0.0)
        if D_arr.size == 1:
            return psd_matrix[0, :]
        if f_arr.size == 1:
            return psd_matrix[:, 0]
        return psd_matrix

    def forward_psd(self, f, D, H, W, theta, r0, qb, tau_c=None, clip_tau_c=False, D50=None, tau_c50=None, pdf=None):
        f_arr = np.atleast_1d(np.asarray(f, dtype=float))
        D_arr = np.atleast_1d(np.asarray(D, dtype=float))
        if D_arr.size == 0:
            return np.zeros_like(f_arr, dtype=float)
        if D_arr.size == 1:
            return self._psd(f_arr, D_arr[0], H, W, theta, r0, qb, tau_c, clip_tau_c, D50=D50, tau_c50=tau_c50)

        if pdf is None:
            weights = np.ones_like(D_arr, dtype=float)
            area = safe_trapz(weights, x=D_arr)
            pdf_arr = weights / area if area > 0 else np.ones_like(D_arr) / D_arr.size
        else:
            pdf_arr = np.asarray(pdf, dtype=float)
            pdf_arr = np.where(np.isfinite(pdf_arr) & (pdf_arr > 0), pdf_arr, 0.0)
            area = safe_trapz(pdf_arr, x=D_arr)
            if area > 0:
                pdf_arr = pdf_arr / area
            else:
                weights = np.ones_like(D_arr, dtype=float)
                area = safe_trapz(weights, x=D_arr)
                pdf_arr = weights / area if area > 0 else np.ones_like(D_arr) / D_arr.size

        PSD = np.zeros_like(f_arr, dtype=float)
        for i, fi in enumerate(f_arr):
            psd_D = self._psd(fi, D_arr, H, W, theta, r0, qb, tau_c, clip_tau_c, D50=D50, tau_c50=tau_c50)
            PSD[i] = safe_trapz(pdf_arr * psd_D, x=D_arr)
        return np.where(np.isfinite(PSD) & (PSD > 0), PSD, 0.0)



class ProjectPage(QWidget):
    """项目管理页面：保存项目基本信息、固定台站信息和默认反演参数。"""

    def __init__(self):
        super().__init__()
        self.sediment_page = None
        self.hydraulic_page = None
        self.inversion_page = None
        self.build_ui()

    def set_context_pages(self, sediment_page=None, hydraulic_page=None, inversion_page=None):
        self.sediment_page = sediment_page
        self.hydraulic_page = hydraulic_page
        self.inversion_page = inversion_page

    def build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("项目管理")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        subtitle = QLabel("保存项目基本信息和固定台站参数。完成配置后，后续泥沙粒径、水力参数、反演计算和批量反演会自动继承这些固定信息。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        project_group = QGroupBox("项目基本信息")
        project_form = QFormLayout(project_group)
        self.project_name = QLineEdit()
        self.project_dir = QLineEdit()
        self.output_dir = QLineEdit()
        self.description = QTextEdit()
        self.description.setFixedHeight(85)
        self.description.setPlaceholderText("填写项目说明，例如监测地点、数据时间范围、反演目的等。")
        project_form.addRow("项目名称：", self.project_name)
        project_form.addRow("项目文件夹：", self.make_path_row(self.project_dir, self.choose_project_dir, is_dir=True))
        project_form.addRow("默认输出文件夹：", self.make_path_row(self.output_dir, self.choose_output_dir, is_dir=True))
        project_form.addRow("项目说明：", self.description)
        layout.addWidget(project_group)

        station_group = QGroupBox("固定台站信息")
        station_form = QFormLayout(station_group)
        self.station_id = QLineEdit()
        self.river_name = QLineEdit()
        self.station_name = QLineEdit()
        self.cross_section_file = QLineEdit()
        self.grain_distribution_file = QLineEdit()
        self.slope_edit = QLineEdit("0.01")
        self.r0_edit = QLineEdit("17")
        station_form.addRow("台站编号：", self.station_id)
        station_form.addRow("河流名称：", self.river_name)
        station_form.addRow("站点名称：", self.station_name)
        station_form.addRow("河道断面文件：", self.make_path_row(self.cross_section_file, self.choose_cross_section_file))
        station_form.addRow("泥沙级配文件：", self.make_path_row(self.grain_distribution_file, self.choose_grain_distribution_file))
        station_form.addRow("河床坡降 <i>S</i>，m/m：", self.slope_edit)
        station_form.addRow("台站-河道距离 r<sub>0</sub>，m：", self.r0_edit)
        layout.addWidget(station_group)

        seismic_group = QGroupBox("地震传播参数")
        seismic_form = QFormLayout(seismic_group)
        self.v0_edit = QLineEdit("2206")
        self.z0_edit = QLineEdit("1000")
        self.f0_edit = QLineEdit("1")
        self.a_edit = QLineEdit("0.272")
        self.Q0_edit = QLineEdit("20")
        self.eta_edit = QLineEdit("0")
        self.phi_edit = QLineEdit("0")
        self.eb_edit = QLineEdit("0.5")
        self.fx_edit = QLineEdit("0.146")
        self.fy_edit = QLineEdit("0.146")
        self.fz_edit = QLineEdit("0.539")
        self.Nzz_edit = QLineEdit("0.352")
        seismic_form.addRow("v<sub>0</sub>（参考相速度，m/s）：", self.v0_edit)
        seismic_form.addRow("z<sub>0</sub>（参考深度，m）：", self.z0_edit)
        seismic_form.addRow("f<sub>0</sub>（参考频率，Hz）：", self.f0_edit)
        seismic_form.addRow("a（频散指数）：", self.a_edit)
        seismic_form.addRow("Q<sub>0</sub>（品质因子）：", self.Q0_edit)
        seismic_form.addRow("η（Q 的频率指数）：", self.eta_edit)
        seismic_form.addRow("φ（源–台站方位角）：", self.phi_edit)
        seismic_form.addRow("e<sub>b</sub>（反弹系数）：", self.eb_edit)
        seismic_form.addRow("f<sub>x</sub>（x 向冲量系数）：", self.fx_edit)
        seismic_form.addRow("f<sub>y</sub>（y 向冲量系数）：", self.fy_edit)
        seismic_form.addRow("f<sub>z</sub>（z 向冲量系数）：", self.fz_edit)
        seismic_form.addRow("N<sub>zz</sub>（垂直辐射项）：", self.Nzz_edit)
        layout.addWidget(seismic_group)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("保存项目配置")
        self.load_button = QPushButton("加载项目配置")
        self.apply_button = QPushButton("应用到各模块")
        self.clear_button = QPushButton("清空页面")
        self.save_button.clicked.connect(self.save_project)
        self.load_button.clicked.connect(self.load_project)
        self.apply_button.clicked.connect(self.apply_project_config_to_modules)
        self.clear_button.clicked.connect(self.clear_form)
        for btn in [self.save_button, self.load_button, self.apply_button, self.clear_button]:
            button_layout.addWidget(btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(100)
        self.log_box.setPlaceholderText("项目配置保存、加载和自动应用信息会显示在这里。")
        layout.addWidget(self.log_box)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def make_path_row(self, line_edit, function, is_dir=False):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(function)
        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        return row

    def choose_project_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择项目文件夹")
        if path:
            self.project_dir.setText(path)
            if not self.output_dir.text().strip():
                self.output_dir.setText(str(Path(path) / "outputs"))
            if not self.project_name.text().strip():
                self.project_name.setText(Path(path).name)

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir.setText(path)

    def choose_cross_section_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择河道断面文件", "", "Data Files (*.csv *.txt *.xlsx *.xls);;All Files (*)")
        if path:
            self.cross_section_file.setText(path)

    def choose_grain_distribution_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择泥沙级配文件", "", "Data Files (*.csv *.txt *.xlsx *.xls);;All Files (*)")
        if path:
            self.grain_distribution_file.setText(path)

    def _float_value(self, edit, default, name):
        text = edit.text().strip()
        if not text:
            return float(default)
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{name} 必须为数字。")

    def get_seismic_params_dict(self):
        return {
            "v0": self._float_value(self.v0_edit, 2206.0, "v0"),
            "z0": self._float_value(self.z0_edit, 1000.0, "z0"),
            "f0": self._float_value(self.f0_edit, 1.0, "f0"),
            "a": self._float_value(self.a_edit, 0.272, "a"),
            "Q0": self._float_value(self.Q0_edit, 20.0, "Q0"),
            "eta": self._float_value(self.eta_edit, 0.0, "eta"),
            "phi": self._float_value(self.phi_edit, 0.0, "phi"),
            "eb": self._float_value(self.eb_edit, 0.5, "eb"),
            "fx": self._float_value(self.fx_edit, 0.146, "fx"),
            "fy": self._float_value(self.fy_edit, 0.146, "fy"),
            "fz": self._float_value(self.fz_edit, 0.539, "fz"),
            "Nzz": self._float_value(self.Nzz_edit, 0.352, "Nzz"),
        }

    def get_project_data(self):
        slope = self._float_value(self.slope_edit, 0.01, "河床坡降 S")
        r0 = self._float_value(self.r0_edit, 17.0, "台站-河道距离 r₀")
        return {
            "project": {
                "project_name": self.project_name.text().strip(),
                "project_dir": self.project_dir.text().strip(),
                "output_dir": self.output_dir.text().strip(),
                "description": self.description.toPlainText().strip(),
            },
            "station": {
                "station_id": self.station_id.text().strip(),
                "river_name": self.river_name.text().strip(),
                "station_name": self.station_name.text().strip(),
                "cross_section_file": self.cross_section_file.text().strip(),
                "grain_distribution_file": self.grain_distribution_file.text().strip(),
                "slope": slope,
                "r0_m": r0,
            },
            "seismic_params": self.get_seismic_params_dict(),
            "paths": {
                "project_dir": self.project_dir.text().strip(),
                "output_dir": self.output_dir.text().strip(),
            },
        }

    def set_project_data(self, data):
        data = data or {}
        project = data.get("project", data) if isinstance(data, dict) else {}
        station = data.get("station", {}) if isinstance(data.get("station", {}), dict) else {}
        seismic = data.get("seismic_params", {}) if isinstance(data.get("seismic_params", {}), dict) else {}
        paths = data.get("paths", {}) if isinstance(data.get("paths", {}), dict) else {}

        self.project_name.setText(str(project.get("project_name", data.get("project_name", ""))))
        self.project_dir.setText(str(project.get("project_dir", paths.get("project_dir", data.get("project_dir", "")))))
        self.output_dir.setText(str(project.get("output_dir", paths.get("output_dir", data.get("output_dir", "")))))
        self.description.setText(str(project.get("description", data.get("description", ""))))

        self.station_id.setText(str(station.get("station_id", data.get("station_id", ""))))
        self.river_name.setText(str(station.get("river_name", data.get("river_name", ""))))
        self.station_name.setText(str(station.get("station_name", data.get("station_name", ""))))
        self.cross_section_file.setText(str(station.get("cross_section_file", data.get("cross_section_file", ""))))
        self.grain_distribution_file.setText(str(station.get("grain_distribution_file", data.get("grain_distribution_file", ""))))
        self.slope_edit.setText(str(station.get("slope", data.get("slope", 0.01))))
        self.r0_edit.setText(str(station.get("r0_m", station.get("r0", data.get("r0_m", 17.0)))))

        for key, edit in [
            ("v0", self.v0_edit), ("z0", self.z0_edit), ("f0", self.f0_edit), ("a", self.a_edit),
            ("Q0", self.Q0_edit), ("eta", self.eta_edit), ("phi", self.phi_edit), ("eb", self.eb_edit),
            ("fx", self.fx_edit), ("fy", self.fy_edit), ("fz", self.fz_edit), ("Nzz", self.Nzz_edit),
        ]:
            if key in seismic:
                edit.setText(str(seismic.get(key)))

    def save_project(self):
        if not self.project_dir.text().strip():
            QMessageBox.warning(self, "提示", "请先选择项目文件夹。")
            return
        try:
            data = self.get_project_data()
            project_path = Path(data["project"]["project_dir"])
            project_path.mkdir(parents=True, exist_ok=True)
            output_dir = Path(data["project"].get("output_dir") or project_path / "outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.setText(str(output_dir))
            data["project"]["output_dir"] = str(output_dir)
            data["paths"]["output_dir"] = str(output_dir)
            config_path = project_path / "project_config.json"
            old_config = load_json_safely(config_path)
            new_config = deep_update_dict(old_config, data)
            # 根层兼容旧代码检索。
            new_config.update({
                "project_name": data["project"]["project_name"],
                "project_dir": data["project"]["project_dir"],
                "output_dir": data["project"]["output_dir"],
                "river_name": data["station"]["river_name"],
                "station_name": data["station"]["station_name"],
            })
            save_json_safely(config_path, new_config)
            self.log_box.setPlainText(f"项目配置已保存：\n{config_path}")
            self.apply_project_config_to_modules(show_message=False)
            QMessageBox.information(self, "保存成功", f"项目配置已保存：\n{config_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存项目配置时出错：\n{e}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择项目配置文件", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            data = load_json_safely(path)
            self.set_project_data(data)
            self.apply_project_config_to_modules(show_message=False)
            self.log_box.setPlainText(f"项目配置已加载：\n{path}")
            QMessageBox.information(self, "加载成功", "项目配置已加载。")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载项目配置时出错：\n{e}")

    def get_station_profile(self):
        seismic_dict = self.get_seismic_params_dict()
        sp = BedloadSeismicParams(**seismic_dict)
        return StationProfile(
            station_id=self.station_id.text().strip(),
            river_name=self.river_name.text().strip(),
            station_name=self.station_name.text().strip(),
            cross_section_file=self.cross_section_file.text().strip(),
            grain_distribution_file=self.grain_distribution_file.text().strip(),
            slope=self._float_value(self.slope_edit, 0.01, "河床坡降 S"),
            r0_m=self._float_value(self.r0_edit, 17.0, "r0"),
            seismic_params=sp,
        )

    def apply_project_config_to_modules(self, show_message=True):
        lines = []
        try:
            profile = self.get_station_profile()
            out_dir = self.output_dir.text().strip()
            if self.sediment_page is not None:
                set_widget_text_if_nonempty(getattr(self.sediment_page, "grain_file", None), profile.grain_distribution_file)
                if out_dir and not widget_text(getattr(self.sediment_page, "output_dir", None)):
                    set_widget_text(getattr(self.sediment_page, "output_dir", None), str(Path(out_dir) / "sediment"))
                lines.append("泥沙粒径模块已继承级配文件。")
            if self.hydraulic_page is not None:
                set_widget_text_if_nonempty(getattr(self.hydraulic_page, "cross_file", None), profile.cross_section_file)
                set_widget_text(getattr(self.hydraulic_page, "slope_edit", None), profile.slope)
                if out_dir and not widget_text(getattr(self.hydraulic_page, "output_dir", None)):
                    set_widget_text(getattr(self.hydraulic_page, "output_dir", None), str(Path(out_dir) / "hydraulic"))
                lines.append("水力参数模块已继承断面文件和坡降 S。")
            if self.inversion_page is not None:
                set_widget_text_if_nonempty(getattr(self.inversion_page, "grain_file", None), profile.grain_distribution_file)
                set_widget_text(getattr(self.inversion_page, "r0_edit", None), profile.r0_m)
                if out_dir and not widget_text(getattr(self.inversion_page, "output_dir", None)):
                    set_widget_text(getattr(self.inversion_page, "output_dir", None), str(Path(out_dir) / "inversion"))
                if profile.seismic_params is not None:
                    for key in ["v0", "z0", "f0", "a", "Q0", "eta", "phi", "eb", "fx", "fy", "fz", "Nzz"]:
                        set_widget_text(getattr(self.inversion_page, f"{key}_edit", None), getattr(profile.seismic_params, key))
                lines.append("反演计算模块已继承级配文件、r₀ 和传播参数。")
            if not lines:
                lines.append("当前没有可应用的模块。")
            self.log_box.setPlainText("\n".join(lines))
            if show_message:
                QMessageBox.information(self, "应用完成", "项目配置已应用到相关模块。")
        except Exception as e:
            QMessageBox.critical(self, "应用失败", f"应用项目配置时出错：\n{e}")

    def clear_form(self):
        for edit in [self.project_name, self.project_dir, self.output_dir, self.station_id, self.river_name, self.station_name,
                     self.cross_section_file, self.grain_distribution_file, self.slope_edit, self.r0_edit,
                     self.v0_edit, self.z0_edit, self.f0_edit, self.a_edit, self.Q0_edit, self.eta_edit,
                     self.phi_edit, self.eb_edit, self.fx_edit, self.fy_edit, self.fz_edit, self.Nzz_edit]:
            edit.clear()
        self.slope_edit.setText("0.01")
        self.r0_edit.setText("17")
        self.v0_edit.setText("2206"); self.z0_edit.setText("1000"); self.f0_edit.setText("1"); self.a_edit.setText("0.272")
        self.Q0_edit.setText("20"); self.eta_edit.setText("0"); self.phi_edit.setText("0"); self.eb_edit.setText("0.5")
        self.fx_edit.setText("0.146"); self.fy_edit.setText("0.146"); self.fz_edit.setText("0.539"); self.Nzz_edit.setText("0.352")
        self.description.clear()
        self.log_box.clear()


class SeismicPreprocessPage(QWidget):
    """SAC 地震数据读取、去响应和保存处理后 SAC 文件。"""

    processed_sac_saved = Signal(str)

    def __init__(self):
        super().__init__()
        self.raw_trace = None
        self.processed_trace = None
        self.processed_unit = ""
        self.last_processed_paths = []
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("地震预处理")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        subtitle = QLabel("读取单个 SAC 或 SAC 文件夹；根据处理模式选择是否去仪器响应；输出处理后的 SAC 文件。")
        subtitle.setStyleSheet("color: gray; font-size: 14px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setSpacing(10)

        self.input_mode = QComboBox()
        self.input_mode.addItem("单个 SAC 文件", "FILE")
        self.input_mode.addItem("SAC 文件夹", "DIR")
        self.input_mode.currentIndexChanged.connect(self.update_input_mode_ui)

        self.sac_file = QLineEdit()
        self.response_file = QLineEdit()
        self.output_dir = QLineEdit()

        form.addRow("输入方式：", self.input_mode)
        form.addRow("SAC 文件/文件夹：", self.make_path_row(self.sac_file, self.choose_sac_file, button_attr="sac_browse_button"))

        response_row = QWidget()
        response_layout = QHBoxLayout(response_row)
        response_layout.setContentsMargins(0, 0, 0, 0)

        self.response_browse_button = QPushButton("浏览")
        self.response_browse_button.clicked.connect(self.choose_response_file)

        response_layout.addWidget(self.response_file)
        response_layout.addWidget(self.response_browse_button)

        self.response_mode = QComboBox()
        self.response_mode.addItem("不去仪器响应，直接读取 SAC 数据", "RAW")
        self.response_mode.addItem("使用内置 STS-2 PAZ 去响应，输出速度 m/s", "STS2_PAZ_VEL")
        self.response_mode.addItem("使用外部响应文件去响应为速度 VEL，单位 m/s", "VEL")
        self.response_mode.setCurrentIndex(1)
        self.response_mode.currentIndexChanged.connect(self.update_response_ui_state)

        form.addRow("处理模式：", self.response_mode)
        form.addRow("响应文件：", response_row)
        form.addRow("输出文件夹：", self.make_path_row(self.output_dir, self.choose_output_dir))

        self.convert_to_beijing = QCheckBox("保存处理后 SAC 时间为北京时间 UTC+8")
        self.convert_to_beijing.setChecked(True)
        form.addRow("时间设置：", self.convert_to_beijing)

        self.pre_filt_1 = QLineEdit("0.01")
        self.pre_filt_2 = QLineEdit("0.02")
        self.pre_filt_3 = QLineEdit("45")
        self.pre_filt_4 = QLineEdit("50")

        pre_filt_row = QWidget()
        pre_filt_layout = QHBoxLayout(pre_filt_row)
        pre_filt_layout.setContentsMargins(0, 0, 0, 0)
        pre_filt_layout.addWidget(self.pre_filt_1)
        pre_filt_layout.addWidget(self.pre_filt_2)
        pre_filt_layout.addWidget(self.pre_filt_3)
        pre_filt_layout.addWidget(self.pre_filt_4)

        form.addRow("外部响应 pre_filt：", pre_filt_row)

        layout.addLayout(form)

        button_layout = QHBoxLayout()

        self.read_button = QPushButton("读取/扫描 SAC 信息")
        self.process_button = QPushButton("处理并保存 SAC")
        self.clear_button = QPushButton("清空")

        self.read_button.clicked.connect(self.read_sac)
        self.process_button.clicked.connect(self.process_and_save_sac)
        self.clear_button.clicked.connect(self.clear_page)

        button_layout.addWidget(self.read_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setPlaceholderText("点击“读取/扫描 SAC 信息”后，SAC 文件信息会显示在这里；该页面不绘制波形。")

        layout.addWidget(self.info_box, stretch=1)

        self.update_response_ui_state()
        self.update_input_mode_ui()

    def make_path_row(self, line_edit, function, button_attr=None):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(function)
        if button_attr:
            setattr(self, button_attr, browse_button)

        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)

        return row

    def is_dir_mode(self):
        return hasattr(self, "input_mode") and self.input_mode.currentData() == "DIR"

    def update_input_mode_ui(self):
        if self.is_dir_mode():
            self.sac_file.setPlaceholderText("请选择包含 SAC 文件的文件夹")
            if hasattr(self, "sac_browse_button"):
                self.sac_browse_button.setText("浏览文件夹")
        else:
            self.sac_file.setPlaceholderText("请选择单个 SAC 文件")
            if hasattr(self, "sac_browse_button"):
                self.sac_browse_button.setText("浏览文件")
        self.raw_trace = None
        self.processed_trace = None
        self.processed_unit = ""
        self.last_processed_paths = []
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)

    def update_response_ui_state(self):
        mode = self.response_mode.currentData()
        use_external_response = mode == "VEL"

        self.response_file.setEnabled(use_external_response)
        self.response_browse_button.setEnabled(use_external_response)

        self.pre_filt_1.setEnabled(use_external_response)
        self.pre_filt_2.setEnabled(use_external_response)
        self.pre_filt_3.setEnabled(use_external_response)
        self.pre_filt_4.setEnabled(use_external_response)

        if mode == "RAW":
            self.response_file.setPlaceholderText("不去仪器响应，无需响应文件")
            self.response_file.clear()
        elif mode == "STS2_PAZ_VEL":
            self.response_file.setPlaceholderText("使用内置 STS-2 PAZ 参数，无需响应文件")
            self.response_file.clear()
        elif mode == "VEL":
            self.response_file.setPlaceholderText("请选择 StationXML / RESP / SAC_PZ 响应文件")

    def choose_sac_file(self):
        if self.is_dir_mode():
            path = QFileDialog.getExistingDirectory(self, "选择 SAC 文件夹")
            if path:
                self.sac_file.setText(path)
                if not self.output_dir.text().strip():
                    self.output_dir.setText(str(Path(path) / "outputs"))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择 SAC 文件",
                "",
                "SAC Files (*.sac *.SAC);;All Files (*)",
            )
            if path:
                self.sac_file.setText(path)
                if not self.output_dir.text().strip():
                    self.output_dir.setText(str(Path(path).parent / "outputs"))

        self.raw_trace = None
        self.processed_trace = None
        self.processed_unit = ""
        self.last_processed_paths = []
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)

    def choose_response_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择仪器响应文件",
            "",
            "Response Files (*.xml *.XML *.resp *.RESP *.pz *.PZ *.sacpz *.SACPZ);;All Files (*)",
        )
        if path:
            self.response_file.setText(path)

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir.setText(path)

    def scan_sac_files_in_dir(self, folder):
        folder = Path(folder)
        if not folder.exists() or not folder.is_dir():
            return []

        files = []
        seen = set()
        for f in folder.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() != ".sac":
                continue
            key = str(f.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            files.append(f)
        return sorted(files, key=lambda p: p.name.lower())

    def get_selected_sac_files(self):
        input_text = self.sac_file.text().strip()
        if not input_text:
            raise ValueError("请先选择 SAC 文件或 SAC 文件夹。")

        path = Path(input_text)
        if self.is_dir_mode():
            if not path.exists() or not path.is_dir():
                raise ValueError("SAC 文件夹不存在。")
            files = self.scan_sac_files_in_dir(path)
            if not files:
                raise ValueError("该文件夹下没有 SAC 文件。")
            return files

        if not path.exists() or not path.is_file():
            raise ValueError("SAC 文件不存在。")
        return [path]

    def load_trace_from_file(self, path):
        try:
            from obspy import read
        except ImportError:
            raise ImportError("当前环境未安装 ObsPy，请先安装 obspy。")

        st = read(str(path))
        if len(st) == 0:
            raise ValueError("SAC 文件为空。")
        st.merge(method=1, fill_value="interpolate")
        return st[0].copy()

    def read_sac(self):
        try:
            files = self.get_selected_sac_files()
        except Exception as e:
            QMessageBox.warning(self, "提示", str(e))
            return

        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)

        if self.is_dir_mode():
            lines = []
            lines.append("SAC 文件夹扫描结果")
            lines.append("-" * 60)
            lines.append(f"文件夹：{self.sac_file.text().strip()}")
            lines.append(f"发现 SAC 文件数：{len(files)}")
            lines.append("")
            for f in files:
                lines.append(f"  - {f.name}")

            try:
                tr = self.load_trace_from_file(files[0])
                self.raw_trace = tr
                self.processed_trace = None
                self.processed_unit = ""
                lines.append("")
                lines.append("首个 SAC 文件信息：")
                lines.append(self.make_trace_info_text(tr, title=files[0].name))
            except Exception as e:
                lines.append("")
                lines.append(f"首个 SAC 文件读取失败：{e}")

            self.info_box.setPlainText("\n".join(lines))
            return

        try:
            tr = self.load_trace_from_file(files[0])
            self.raw_trace = tr
            self.processed_trace = None
            self.processed_unit = ""
            self.show_trace_info(tr, title="原始 SAC 信息")
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"读取 SAC 文件时出错：\n{e}")

    def process_trace(self, tr):
        tr = tr.copy()

        try:
            tr.detrend("demean")
            tr.detrend("linear")
        except Exception:
            pass

        mode = self.response_mode.currentData()

        if mode == "RAW":
            self.processed_unit = "amplitude"
            return tr

        if mode == "STS2_PAZ_VEL":
            paz_sts2, paz_5hz = self.get_builtin_sts2_paz()
            tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_5hz, taper=False)
            self.processed_unit = "m/s"
            return tr

        if mode == "VEL":
            response_path = self.response_file.text().strip()
            if not response_path:
                raise ValueError("当前选择了外部响应文件去响应模式，请先选择响应文件。")
            if not Path(response_path).exists():
                raise ValueError("响应文件不存在。")

            from obspy import read_inventory
            from obspy import Stream

            pre_filt = self.get_pre_filt()
            inv = read_inventory(response_path)
            st = Stream(traces=[tr])
            st.remove_response(inventory=inv, output="VEL", pre_filt=pre_filt, water_level=60)
            tr = st[0]
            self.processed_unit = "m/s"
            return tr

        raise ValueError("未知处理模式。")

    def sac_header_is_bjt(self, tr):
        try:
            return str(tr.stats.sac.kuser0).strip().upper() == "BJT"
        except Exception:
            return False

    def apply_beijing_time_if_needed(self, tr):
        tr = tr.copy()
        if self.convert_to_beijing.isChecked() and not self.sac_header_is_bjt(tr):
            tr.stats.starttime = tr.stats.starttime + 8 * 3600
            try:
                tr.stats.sac.kuser0 = "BJT"
            except Exception:
                pass
        return tr

    def save_processed_trace_for_file(self, tr, input_sac_path, output_dir):
        input_sac_path = Path(input_sac_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tr = self.apply_beijing_time_if_needed(tr)
        output_path = output_dir / f"{input_sac_path.stem}_processed.sac"
        tr.write(str(output_path), format="SAC")
        return output_path, tr

    def process_and_save_sac(self):
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)
            QApplication.processEvents()

        try:
            files = self.get_selected_sac_files()
        except Exception as e:
            QMessageBox.warning(self, "提示", str(e))
            return

        if self.response_mode.currentData() == "VEL":
            response_path = self.response_file.text().strip()
            if not response_path:
                QMessageBox.warning(self, "提示", "当前选择了外部响应文件去响应模式，请先选择响应文件。")
                return
            if not Path(response_path).exists():
                QMessageBox.warning(self, "提示", "响应文件不存在。")
                return

        output_dir = self.get_output_dir()
        total = len(files)
        ok_paths = []
        failed = []

        for idx, sac_path in enumerate(files):
            try:
                base_progress = int(idx * 100 / max(total, 1))
                if hasattr(self, "progress_bar"):
                    self.progress_bar.setValue(base_progress)
                    QApplication.processEvents()

                tr = self.load_trace_from_file(sac_path)
                self.raw_trace = tr.copy()

                if hasattr(self, "progress_bar"):
                    self.progress_bar.setValue(min(95, base_progress + int(35 / max(total, 1))))
                    QApplication.processEvents()

                tr_processed = self.process_trace(tr)

                if hasattr(self, "progress_bar"):
                    self.progress_bar.setValue(min(98, base_progress + int(70 / max(total, 1))))
                    QApplication.processEvents()

                output_path, saved_trace = self.save_processed_trace_for_file(tr_processed, sac_path, output_dir)
                ok_paths.append(output_path)
                self.processed_trace = saved_trace.copy()

            except Exception as e:
                failed.append((sac_path, str(e)))

            if hasattr(self, "progress_bar"):
                self.progress_bar.setValue(int((idx + 1) * 100 / max(total, 1)))
                QApplication.processEvents()

        self.last_processed_paths = ok_paths

        if ok_paths:
            # 只发射最后一个文件路径即可；时频分析页会自动接收其所在文件夹。
            self.processed_sac_saved.emit(str(ok_paths[-1]))
            self.show_trace_info(self.processed_trace, title="处理后 SAC 信息")

        lines = []
        lines.append("SAC 预处理完成")
        lines.append("-" * 60)
        lines.append(f"输入数量：{total}")
        lines.append(f"成功处理：{len(ok_paths)}")
        lines.append(f"处理失败：{len(failed)}")
        lines.append(f"输出文件夹：{output_dir}")
        if ok_paths:
            lines.append("")
            lines.append("输出文件：")
            for p in ok_paths[:50]:
                lines.append(f"  - {Path(p).name}")
            if len(ok_paths) > 50:
                lines.append(f"  ... 其余 {len(ok_paths) - 50} 个文件未在此处展开显示")
        if failed:
            lines.append("")
            lines.append("失败文件：")
            for f, err in failed[:20]:
                lines.append(f"  - {Path(f).name}: {err}")
            if len(failed) > 20:
                lines.append(f"  ... 其余 {len(failed) - 20} 个失败文件未在此处展开显示")

        if ok_paths:
            lines.append("")
            lines.append("最后一个处理后 SAC 文件信息：")
            lines.append(self.make_trace_info_text(self.processed_trace, title=Path(ok_paths[-1]).name))

        self.info_box.setPlainText("\n".join(lines))

        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(100 if ok_paths else 0)

        if ok_paths:
            QMessageBox.information(self, "处理完成", f"SAC 预处理完成。成功处理 {len(ok_paths)} / {total} 个文件。")
        else:
            QMessageBox.warning(self, "处理失败", "没有成功处理任何 SAC 文件，请查看日志信息。")

    def get_builtin_sts2_paz(self):
        try:
            from obspy.signal.invsim import corn_freq_2_paz
        except ImportError:
            raise ImportError("当前环境未安装 ObsPy，无法调用 corn_freq_2_paz。")

        paz_sts2 = {
            "poles": [-22.211059 + 22.217768j, -22.211059 - 22.217768j],
            "zeros": [0j, 0j],
            "gain": 8000,
            "sensitivity": 7.68e1,
        }

        paz_5hz = corn_freq_2_paz(5.0, damp=0.707)
        paz_5hz["sensitivity"] = 7.68e1

        return paz_sts2, paz_5hz

    def get_pre_filt(self):
        try:
            f1 = float(self.pre_filt_1.text())
            f2 = float(self.pre_filt_2.text())
            f3 = float(self.pre_filt_3.text())
            f4 = float(self.pre_filt_4.text())
        except ValueError:
            raise ValueError("pre_filt 必须为四个数字。")

        if not (0 < f1 < f2 < f3 < f4):
            raise ValueError("pre_filt 必须满足 0 < f1 < f2 < f3 < f4。")

        return (f1, f2, f3, f4)

    def get_display_time(self, trace_time, header_is_bjt=False):
        dt = trace_time.datetime
        if self.convert_to_beijing.isChecked() and not header_is_bjt:
            dt = dt + timedelta(hours=8)
        return dt

    def make_trace_info_text(self, tr, title):
        start_time = tr.stats.starttime
        end_time = tr.stats.endtime
        data = np.asarray(tr.data, dtype=float)
        header_is_bjt = self.sac_header_is_bjt(tr)

        if header_is_bjt:
            display_label = "文件头时间（北京时间 UTC+8）"
            start_display = start_time.datetime
            end_display = end_time.datetime
        else:
            display_label = "北京时间 UTC+8" if self.convert_to_beijing.isChecked() else "UTC 时间"
            start_display = self.get_display_time(start_time, header_is_bjt=False)
            end_display = self.get_display_time(end_time, header_is_bjt=False)

        info = []
        info.append(title)
        info.append("-" * 60)
        info.append(f"台站：{getattr(tr.stats, 'station', '')}")
        info.append(f"通道：{getattr(tr.stats, 'channel', '')}")
        info.append(f"网络：{getattr(tr.stats, 'network', '')}")
        info.append(f"采样率：{tr.stats.sampling_rate} Hz")
        info.append(f"采样点数：{tr.stats.npts}")
        if header_is_bjt:
            info.append(f"{display_label}开始：{start_display}")
            info.append(f"{display_label}结束：{end_display}")
        else:
            info.append(f"UTC 开始时间：{start_time}")
            info.append(f"UTC 结束时间：{end_time}")
            info.append(f"{display_label}开始：{start_display}")
            info.append(f"{display_label}结束：{end_display}")
        info.append(f"数据单位：{self.processed_unit if self.processed_unit else '原始 amplitude'}")
        info.append(f"均值：{np.nanmean(data):.6e}")
        info.append(f"标准差：{np.nanstd(data):.6e}")
        info.append(f"最小值：{np.nanmin(data):.6e}")
        info.append(f"最大值：{np.nanmax(data):.6e}")
        return "\n".join(info)

    def show_trace_info(self, tr, title):
        self.info_box.setPlainText(self.make_trace_info_text(tr, title))

    def get_output_dir(self):
        output_text = self.output_dir.text().strip()
        if output_text:
            output_dir = Path(output_text)
        else:
            input_text = self.sac_file.text().strip()
            if input_text:
                input_path = Path(input_text)
                if self.is_dir_mode():
                    output_dir = input_path / "outputs"
                else:
                    output_dir = input_path.parent / "outputs"
            else:
                output_dir = Path.cwd() / "outputs"
            self.output_dir.setText(str(output_dir))

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_processed_sac(self):
        """兼容旧的单文件保存接口。"""
        if self.processed_trace is None:
            QMessageBox.warning(self, "提示", "请先读取并处理 SAC 文件。")
            return None

        input_text = self.sac_file.text().strip()
        if not input_text:
            QMessageBox.warning(self, "提示", "请先选择 SAC 文件。")
            return None

        try:
            output_path, saved_trace = self.save_processed_trace_for_file(
                self.processed_trace,
                Path(input_text),
                self.get_output_dir(),
            )
            self.processed_trace = saved_trace.copy()
            QMessageBox.information(self, "保存成功", f"处理后 SAC 已保存：\n{output_path}")
            return output_path
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存 SAC 时出错：\n{e}")
            return None

    def clear_page(self):
        self.sac_file.clear()
        self.response_file.clear()
        self.output_dir.clear()
        self.info_box.clear()
        self.raw_trace = None
        self.processed_trace = None
        self.processed_unit = ""
        self.last_processed_paths = []
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)


class TimeFrequencyPage(QWidget):
    """处理后 SAC 文件的时频分析页面：逐个 SAC 文件计算 PSD，不拼接前后天。"""

    energy_csv_saved = Signal(str)

    def __init__(self):
        super().__init__()
        self.discovered_files = []
        self.build_ui()

    def build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("时频分析")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        subtitle = QLabel("自动读取处理后的 SAC 文件，逐个文件计算 Welch PSD，并输出 PSD 矩阵、时频图和 30–80 Hz 线性能量时间序列。")
        subtitle.setStyleSheet("color: gray; font-size: 14px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setSpacing(10)

        self.sac_input_mode_combo = QComboBox()
        self.sac_input_mode_combo.addItems([
            "单个处理后 SAC 文件",
            "处理后 SAC 文件夹",
        ])
        self.sac_input_mode_combo.setCurrentIndex(1)

        self.sac_dir = QLineEdit()
        self.output_dir = QLineEdit()

        form.addRow("输入方式：", self.sac_input_mode_combo)
        form.addRow("处理后 SAC 输入：", self.make_path_row(self.sac_dir, self.choose_sac_dir))
        form.addRow("PSD 输出文件夹：", self.make_path_row(self.output_dir, self.choose_output_dir))

        self.fs_edit = QLineEdit("250")
        self.win_sec_edit = QLineEdit("120")
        self.step_sec_edit = QLineEdit("60")
        self.max_freq_edit = QLineEdit("100")
        self.target_df_edit = QLineEdit("1")

        self.energy_fmin_edit = QLineEdit("30")
        self.energy_fmax_edit = QLineEdit("80")

        self.vmin_edit = QLineEdit("-170")
        self.vmax_edit = QLineEdit("-110")

        form.addRow("STFT 窗长 WIN_SEC：", self.win_sec_edit)
        form.addRow("STFT 步长 STEP_SEC：", self.step_sec_edit)
        form.addRow("最大频率 MAX_FREQ：", self.max_freq_edit)
        form.addRow("目标频率分辨率 TARGET_DF：", self.target_df_edit)

        energy_band_row = QWidget()
        energy_band_layout = QHBoxLayout(energy_band_row)
        energy_band_layout.setContentsMargins(0, 0, 0, 0)
        energy_band_layout.addWidget(self.energy_fmin_edit)
        energy_band_layout.addWidget(QLabel("至"))
        energy_band_layout.addWidget(self.energy_fmax_edit)
        energy_band_layout.addWidget(QLabel("Hz"))

        form.addRow("能量频带：", energy_band_row)
        form.addRow("PSD 图色标 vmin：", self.vmin_edit)
        form.addRow("PSD 图色标 vmax：", self.vmax_edit)

        layout.addLayout(form)

        button_layout = QHBoxLayout()

        self.scan_button = QPushButton("扫描处理后 SAC")
        self.run_button = QPushButton("计算 PSD 与能量")
        self.clear_button = QPushButton("清空日志")

        self.scan_button.clicked.connect(self.scan_files)
        self.run_button.clicked.connect(self.run_psd_files)
        self.clear_button.clicked.connect(self.clear_log)

        button_layout.addWidget(self.scan_button)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("文件扫描、PSD 计算和能量提取日志会显示在这里。")
        self.log_box.setMinimumHeight(110)
        self.log_box.setMaximumHeight(170)
        layout.addWidget(self.log_box)

        preview_title = QLabel("PSD 时频图预览")
        preview_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(preview_title)

        self.figure = Figure(figsize=(12.5, 4.8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(430)
        layout.addWidget(self.canvas)

        note = QLabel("提示：如果预览图显示不全，可以向下滚动页面；完整高清图会保存到 PSD 输出文件夹的 figures 子文件夹中。")
        note.setStyleSheet("color: gray; font-size: 13px;")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()


        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)


    def make_path_row(self, line_edit, function):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(function)
        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        return row

    def is_single_sac_input_mode(self):
        return hasattr(self, "sac_input_mode_combo") and self.sac_input_mode_combo.currentIndex() == 0

    def set_input_from_processed_sac(self, sac_path):
        sac_path = Path(sac_path)
        if sac_path.exists():
            # 预处理模块传入单个 processed SAC 时，时频分析默认处理其所在文件夹，
            # 这样单文件和批量预处理后的流程都能直接衔接。
            if hasattr(self, "sac_input_mode_combo"):
                self.sac_input_mode_combo.setCurrentIndex(1)
            self.sac_dir.setText(str(sac_path.parent))
            if not self.output_dir.text().strip():
                self.output_dir.setText(str(sac_path.parent / "PSD_outputs"))
            self.append_log(f"已自动接收处理后 SAC 文件：{sac_path}")
            self.append_log(f"处理后 SAC 输入设置为文件夹：{sac_path.parent}")

    def choose_sac_dir(self):
        if self.is_single_sac_input_mode():
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择处理后 SAC 文件",
                "",
                "SAC Files (*.sac *.SAC);;All Files (*)",
            )
            if path:
                self.sac_dir.setText(path)
                if not self.output_dir.text().strip():
                    self.output_dir.setText(str(Path(path).parent / "PSD_outputs"))
        else:
            path = QFileDialog.getExistingDirectory(self, "选择处理后 SAC 文件夹")
            if path:
                self.sac_dir.setText(path)
                if not self.output_dir.text().strip():
                    self.output_dir.setText(str(Path(path) / "PSD_outputs"))

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择 PSD 输出文件夹")
        if path:
            self.output_dir.setText(path)

    def append_log(self, message):
        self.log_box.append(message)

    def clear_log(self):
        self.log_box.clear()
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)

    def sac_header_is_bjt(self, tr):
        """判断 SAC 文件头是否已由预处理模块标记为北京时间。"""
        try:
            return str(tr.stats.sac.kuser0).strip().upper() == "BJT"
        except Exception:
            return False

    def get_params(self):
        try:
            params = {
                "win_sec": float(self.win_sec_edit.text()),
                "step_sec": float(self.step_sec_edit.text()),
                "max_freq": float(self.max_freq_edit.text()),
                "target_df": float(self.target_df_edit.text()),
                "energy_fmin": float(self.energy_fmin_edit.text()),
                "energy_fmax": float(self.energy_fmax_edit.text()),
                "vmin": float(self.vmin_edit.text()),
                "vmax": float(self.vmax_edit.text()),
            }
        except ValueError:
            raise ValueError("时频分析参数必须为数字。")

        if params["win_sec"] <= 0 or params["step_sec"] <= 0:
            raise ValueError("窗长和步长必须大于 0。")

        if params["max_freq"] <= 0 or params["target_df"] <= 0:
            raise ValueError("最大频率和目标频率分辨率必须大于 0。")

        if params["energy_fmin"] < 0 or params["energy_fmax"] <= params["energy_fmin"]:
            raise ValueError("能量频带必须满足 0 <= fmin < fmax。")

        if params["energy_fmax"] > params["max_freq"]:
            raise ValueError("能量频带上限不能大于 MAX_FREQ。")

        return params

    def scan_files(self):
        sac_input_text = self.sac_dir.text().strip()
        if not sac_input_text:
            QMessageBox.warning(self, "提示", "请先选择处理后 SAC 文件或文件夹。")
            return

        sac_input = Path(sac_input_text)
        if not sac_input.exists():
            QMessageBox.warning(self, "提示", "处理后 SAC 文件或文件夹不存在。")
            return

        all_sac_files = []

        if sac_input.is_file():
            if sac_input.suffix.lower() != ".sac":
                QMessageBox.warning(self, "提示", "选择的文件不是 SAC 文件。")
                return
            all_sac_files = [sac_input]
            self.append_log("当前输入为单个处理后 SAC 文件。")
        else:
            # Windows 下 glob("*.sac") 和 glob("*.SAC") 可能重复匹配同一批文件。
            # 这里统一用 iterdir + suffix.lower()，避免 3 个文件被扫描成 6 个。
            seen = set()
            for f in sac_input.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() != ".sac":
                    continue
                try:
                    key = str(f.resolve()).lower()
                except Exception:
                    key = str(f).lower()
                if key in seen:
                    continue
                seen.add(key)
                all_sac_files.append(f)
            self.append_log("当前输入为处理后 SAC 文件夹。")

        all_sac_files = sorted(all_sac_files, key=lambda p: p.name.lower())
        processed_files = [
            f for f in all_sac_files
            if "_processed" in f.stem.lower() or "_clean" in f.stem.lower()
        ]

        if processed_files:
            sac_files = processed_files
            self.append_log("优先使用文件名中包含 _processed 或 _clean 的 SAC 文件。")
        else:
            sac_files = all_sac_files
            self.append_log("未发现 _processed 或 _clean 文件，使用当前输入中的全部 SAC 文件。")

        if not sac_files:
            QMessageBox.warning(self, "提示", "当前输入中没有 SAC 文件。")
            return

        self.discovered_files = sac_files
        self.append_log(f"共发现可处理 SAC 文件 {len(self.discovered_files)} 个。")
        for file_path in self.discovered_files:
            self.append_log(f"  - {file_path.name}")

    def run_psd_files(self):
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)
            QApplication.processEvents()

        if not self.discovered_files:
            self.scan_files()
        if not self.discovered_files:
            return

        try:
            params = self.get_params()
        except Exception as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return

        sac_input = Path(self.sac_dir.text().strip())
        output_dir_text = self.output_dir.text().strip()
        if output_dir_text:
            output_dir = Path(output_dir_text)
        else:
            base_dir = sac_input.parent if sac_input.is_file() else sac_input
            output_dir = base_dir / "PSD_outputs"
            self.output_dir.setText(str(output_dir))

        npz_dir = output_dir / "npz"
        img_dir = output_dir / "figures"
        energy_dir = output_dir / "energy_csv"
        npz_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        energy_dir.mkdir(parents=True, exist_ok=True)

        ok_count = 0
        total_files = len(self.discovered_files)

        def update_progress(file_index, inner_ratio):
            if not hasattr(self, "progress_bar"):
                return
            inner_ratio = max(0.0, min(1.0, float(inner_ratio)))
            progress = int(((file_index + inner_ratio) / max(total_files, 1)) * 100)
            self.progress_bar.setValue(progress)
            QApplication.processEvents()

        for file_index, sac_file in enumerate(self.discovered_files):
            try:
                self.append_log(f"正在处理：{sac_file.name}")
                stem = sac_file.stem
                fmin_str = f"{params['energy_fmin']:g}".replace(".", "p")
                fmax_str = f"{params['energy_fmax']:g}".replace(".", "p")
                npz_path = npz_dir / f"{stem}_PSD.npz"
                png_path = img_dir / f"{stem}_PSD.png"
                energy_csv_path = energy_dir / f"{stem}_energy_{fmin_str}_{fmax_str}Hz.csv"

                result = self.process_one_sac_file(
                    sac_path=sac_file,
                    npz_path=npz_path,
                    png_path=png_path,
                    energy_csv_path=energy_csv_path,
                    params=params,
                    progress_callback=lambda ratio, i=file_index: update_progress(i, ratio),
                )

                if result:
                    ok_count += 1
                    self.append_log(f"  完成：{sac_file.name}")
                    self.append_log(f"  PSD NPZ：{npz_path}")
                    self.append_log(f"  PSD PNG：{png_path}")
                    self.append_log(f"  能量 CSV：{energy_csv_path}")
                    self.energy_csv_saved.emit(str(energy_csv_path))
                else:
                    self.append_log(f"  无有效 PSD 结果：{sac_file.name}")

                update_progress(file_index + 1, 0.0)

            except Exception as e:
                self.append_log(f"  处理失败：{sac_file.name} | {e}")
                update_progress(file_index + 1, 0.0)

        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(100)
        QMessageBox.information(self, "处理完成", f"时频分析完成。成功处理 {ok_count} 个 SAC 文件。")

    def process_one_sac_file(self, sac_path, npz_path, png_path, energy_csv_path, params, progress_callback=None):
        try:
            import obspy
            from scipy.signal import welch
        except ImportError as e:
            QMessageBox.critical(self, "缺少依赖", f"缺少必要依赖：{e}")
            return False

        win_sec = params["win_sec"]
        step_sec = params["step_sec"]
        max_freq = params["max_freq"]
        target_df = params["target_df"]
        energy_fmin = params["energy_fmin"]
        energy_fmax = params["energy_fmax"]

        try:
            st = obspy.read(str(sac_path))
        except Exception as e:
            self.append_log(f"  SAC 读取失败：{e}")
            return False

        if len(st) == 0:
            return False

        st.merge(method=1, fill_value="interpolate")
        tr = st[0]

        # 时间安全检查：本软件要求时频分析读取的 processed SAC 已在预处理阶段写成北京时间。
        # 判断依据为预处理阶段写入的 SAC 头字段 kuser0 = "BJT"。
        # 如果没有该标记，通常说明文件是旧版 processed SAC 或未经过当前预处理流程；
        # 此时继续计算会导致 energy CSV 时间仍为 UTC，容易与北京时间水位文件错配。
        if not self.sac_header_is_bjt(tr):
            self.append_log(
                "  时间检查失败：该 SAC 文件头没有 BJT 标记。"
                "请先回到‘地震预处理’页面重新处理原始 SAC，"
                "生成带 BJT 标记的 processed SAC 后再计算 PSD。"
            )
            return False

        data = np.asarray(tr.data, dtype=float)
        fs = float(tr.stats.sampling_rate)

        npts = len(data)
        win_samples = int(round(win_sec * fs))
        step_samples = int(round(step_sec * fs))

        if win_samples <= 0 or step_samples <= 0:
            return False

        if npts < win_samples:
            self.append_log("  数据长度短于一个窗口，跳过。")
            return False

        psd_columns = []
        psd_times_num = []
        psd_times_str = []
        f_new = np.arange(0, max_freq + target_df, target_df)

        start_indices = list(range(0, npts - win_samples + 1, step_samples))
        total_windows = len(start_indices)

        for win_i, start_idx in enumerate(start_indices):
            seg = data[start_idx: start_idx + win_samples]
            center_time = tr.stats.starttime + (start_idx + win_samples / 2.0) / fs
            f, p = welch(seg, fs=fs, nperseg=win_samples)

            p_smoothed = []
            for f_low in f_new:
                mask = (f >= f_low - target_df / 2.0) & (f < f_low + target_df / 2.0)
                if np.any(mask):
                    p_smoothed.append(np.median(p[mask]))
                else:
                    p_smoothed.append(1e-20)

            psd_db = 10.0 * np.log10(np.array(p_smoothed) + 1e-20)
            psd_columns.append(psd_db)

            # processed SAC 的文件头时间已经在预处理阶段写成北京时间；这里不再额外 +8。
            center_dt = center_time.datetime
            psd_times_num.append(mdates.date2num(center_dt))
            psd_times_str.append(center_dt.strftime("%Y-%m-%d %H:%M:%S"))

            if progress_callback is not None:
                if win_i % 5 == 0 or win_i == total_windows - 1:
                    progress_callback((win_i + 1) / max(total_windows, 1))

        if not psd_columns:
            return False

        matrix = np.array(psd_columns).T.astype(np.float32)
        time_array = np.array(psd_times_num, dtype=np.float64)
        freq_array = f_new.astype(np.float32)
        freq_mask = (freq_array >= energy_fmin) & (freq_array <= energy_fmax)

        if not np.any(freq_mask):
            self.append_log(f"  {energy_fmin}-{energy_fmax} Hz 频带内没有可用频率点。")
            return False

        energy_1d_db = np.median(matrix[freq_mask, :], axis=0)
        energy_1d_linear = 10 ** (energy_1d_db / 10)

        np.savez_compressed(
            npz_path,
            data=matrix,
            freqs=freq_array,
            times=time_array,
            time_strings=np.array(psd_times_str),
            energy_linear=energy_1d_linear.astype(np.float64),
            energy_db_median=energy_1d_db.astype(np.float64),
            energy_band=np.array([energy_fmin, energy_fmax], dtype=np.float32),
            source_sac=str(sac_path),
            fs=fs,
            win_sec=win_sec,
            step_sec=step_sec,
            max_freq=max_freq,
            target_df=target_df,
            psd_unit="dB",
            energy_unit="linear_from_dB",
        )

        self.save_energy_csv(energy_csv_path=energy_csv_path, time_strings=psd_times_str, energy_linear=energy_1d_linear)
        self.plot_psd_matrix(matrix=matrix, time_array=time_array, freq_array=freq_array, sac_name=sac_path.name, png_path=png_path, params=params)

        if progress_callback is not None:
            progress_callback(1.0)

        return True

    def save_energy_csv(self, energy_csv_path, time_strings, energy_linear):
        try:
            import pandas as pd
        except ImportError:
            QMessageBox.critical(self, "缺少依赖", "当前环境未安装 pandas，请先安装 pandas。")
            return

        df = pd.DataFrame({"time": time_strings, "energy_linear": energy_linear})
        df.to_csv(energy_csv_path, index=False, encoding="utf-8-sig")

    def plot_psd_matrix(self, matrix, time_array, freq_array, sac_name, png_path, params):
        from datetime import datetime, time as datetime_time

        self.figure.clear()
        self.figure.set_size_inches(12.5, 4.8)
        ax = self.figure.add_subplot(111)

        time_datetimes = mdates.num2date(time_array)
        first_dt = time_datetimes[0].replace(tzinfo=None)
        day_start = datetime.combine(first_dt.date(), datetime_time(0, 0, 0))
        day_end = day_start + timedelta(days=1)
        x_min = mdates.date2num(day_start)
        x_max = mdates.date2num(day_end)

        img = ax.imshow(
            matrix,
            extent=[time_array[0], time_array[-1], freq_array[0], freq_array[-1]],
            aspect="auto",
            origin="lower",
            cmap="jet",
            interpolation="nearest",
            vmin=params["vmin"],
            vmax=params["vmax"],
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(freq_array[0], freq_array[-1])
        ax.set_title(f"{sac_name} PSD spectrogram", fontsize=14, pad=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.set_xlabel("Time", fontsize=12)

        tick_hours = list(range(0, 25, 2))
        tick_times = [day_start + timedelta(hours=h) for h in tick_hours]
        tick_nums = [mdates.date2num(t) for t in tick_times]
        tick_labels = [f"{h:02d}:00" for h in tick_hours]

        ax.set_xticks(tick_nums)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(False)

        cbar = self.figure.colorbar(img, ax=ax, orientation="vertical", pad=0.015, fraction=0.035)
        cbar.set_label("PSD (dB)", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        self.figure.subplots_adjust(left=0.07, right=0.93, bottom=0.16, top=0.88)
        self.figure.savefig(png_path, dpi=300, bbox_inches="tight")
        self.canvas.draw()



class SedimentGrainPage(QWidget):
    """模块 4：泥沙粒径模块。"""

    sediment_saved = Signal(str)

    def __init__(self):
        super().__init__()

        self.grain_df_original = None
        self.grain_df_standard = None
        self.source_mode = None

        self.build_ui()

    def build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("泥沙粒径")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        subtitle = QLabel(
            "导入已有粒径分布文件，或输入 D<sub>16</sub>、D<sub>50</sub>、D<sub>84</sub> 生成近似级配曲线。"
            "已有级配文件只输出级配图；代表粒径输入会输出近似级配 CSV 和级配图。"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 14px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        file_group = QGroupBox("输入与输出")
        file_layout = QFormLayout(file_group)
        file_layout.setSpacing(10)

        self.sediment_input_mode_combo = QComboBox()
        self.sediment_input_mode_combo.addItems([
            "导入粒径分布文件",
            "输入代表粒径 D₁₆ / D₅₀ / D₈₄",
        ])
        self.sediment_input_mode_combo.currentIndexChanged.connect(self.update_sediment_input_mode_ui)

        self.grain_file = QLineEdit()
        self.output_dir = QLineEdit()

        file_layout.addRow("输入方式：", self.sediment_input_mode_combo)
        file_layout.addRow("粒径分布文件：", self.make_path_row(self.grain_file, self.choose_grain_file, button_attr="grain_browse_button"))
        file_layout.addRow("输出文件夹：", self.make_path_row(self.output_dir, self.choose_output_dir))

        layout.addWidget(file_group)

        self.percentile_group = QGroupBox("代表粒径输入")
        percentile_layout = QFormLayout(self.percentile_group)
        percentile_layout.setSpacing(10)

        self.d16_edit = QLineEdit("")
        self.d50_edit = QLineEdit("")
        self.d84_edit = QLineEdit("")

        self.d16_edit.setPlaceholderText("例如 2，单位 mm")
        self.d50_edit.setPlaceholderText("例如 10，单位 mm")
        self.d84_edit.setPlaceholderText("例如 40，单位 mm")

        percentile_layout.addRow("D<sub>16</sub>，mm：", self.d16_edit)
        percentile_layout.addRow("D<sub>50</sub>，mm：", self.d50_edit)
        percentile_layout.addRow("D<sub>84</sub>，mm：", self.d84_edit)

        layout.addWidget(self.percentile_group)

        parameter_group = QGroupBox("泥沙基本参数")
        parameter_layout = QFormLayout(parameter_group)
        parameter_layout.setSpacing(10)

        self.rho_s_edit = QLineEdit("2650")
        self.rho_s_edit.setPlaceholderText("颗粒密度，kg/m³")
        parameter_layout.addRow("颗粒密度 ρ<sub>s</sub>，kg/m³：", self.rho_s_edit)

        layout.addWidget(parameter_group)

        button_layout = QHBoxLayout()

        self.read_button = QPushButton("导入粒径分布")
        self.generate_button = QPushButton("生成级配图")
        self.save_button = QPushButton("保存输出")
        self.clear_button = QPushButton("清空")

        self.read_button.clicked.connect(self.read_grain_file)
        self.generate_button.clicked.connect(self.generate_grading_curve)
        self.save_button.clicked.connect(self.save_outputs)
        self.clear_button.clicked.connect(self.clear_page)

        button_layout.addWidget(self.read_button)
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(170)
        self.preview_box.setMaximumHeight(240)
        self.preview_box.setPlaceholderText("粒径信息、识别结果和输出说明会显示在这里。")

        layout.addWidget(self.preview_box)

        plot_title = QLabel("级配图预览")
        plot_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(plot_title)

        self.figure = Figure(figsize=(12.5, 4.8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(430)
        layout.addWidget(self.canvas)

        note = QLabel(
            "说明：已有粒径分布文件默认前两列分别为粒径和比例，若列名包含 D、粒径、fraction、percent 等会自动优先识别。"
            "如果未导入文件，则使用 D<sub>16</sub>、D<sub>50</sub>、D<sub>84</sub> 生成对数正态近似级配。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 13px;")

        layout.addWidget(note)
        layout.addStretch()

        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

        # 初始化泥沙粒径输入方式的互斥状态。
        self.update_sediment_input_mode_ui()

    def make_path_row(self, line_edit, function, button_attr=None):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(function)
        if button_attr:
            setattr(self, button_attr, browse_button)

        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)

        return row

    def is_representative_grain_mode(self):
        return hasattr(self, "sediment_input_mode_combo") and self.sediment_input_mode_combo.currentIndex() == 1

    def update_sediment_input_mode_ui(self):
        representative_mode = self.is_representative_grain_mode()

        # 两种输入方式互斥：
        # 导入级配文件时，不需要填写代表粒径；
        # 输入代表粒径时，不需要选择级配文件。
        if hasattr(self, "grain_file"):
            self.grain_file.setEnabled(not representative_mode)
        if hasattr(self, "grain_browse_button"):
            self.grain_browse_button.setEnabled(not representative_mode)
        if hasattr(self, "percentile_group"):
            self.percentile_group.setEnabled(representative_mode)
        if hasattr(self, "read_button"):
            self.read_button.setEnabled(not representative_mode)

        self.source_mode = "percentiles" if representative_mode else "file"
        self.grain_df_standard = None

        if hasattr(self, "preview_box"):
            if representative_mode:
                self.preview_box.setPlaceholderText(
                    "代表粒径模式：输入 D₁₆、D₅₀、D₈₄ 后点击“生成级配图”，程序会自动生成近似级配。"
                )
            else:
                self.preview_box.setPlaceholderText(
                    "导入模式：选择 D_mm + fraction/percent 格式的粒径分布文件，点击“导入粒径分布”或“生成级配图”。"
                )

    def choose_grain_file(self):
        if self.is_representative_grain_mode():
            QMessageBox.information(self, "提示", "当前为代表粒径输入模式，不需要选择粒径分布文件。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择粒径分布文件",
            "",
            "Data Files (*.csv *.xlsx *.xls *.txt);;All Files (*)",
        )

        if path:
            self.grain_file.setText(path)
            if not self.output_dir.text().strip():
                self.output_dir.setText(str(Path(path).parent / "sediment_outputs"))

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir.setText(path)

    def read_table_data(self, path):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("当前环境未安装 pandas，请先安装 pandas。")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            for encoding in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    return pd.read_csv(path, encoding=encoding)
                except Exception:
                    pass
            return pd.read_csv(path)

        if suffix == ".txt":
            for encoding in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
                except Exception:
                    pass
            return pd.read_csv(path, sep=None, engine="python")

        if suffix == ".xlsx":
            try:
                return pd.read_excel(path, engine="openpyxl")
            except ImportError:
                raise ImportError("读取 .xlsx 文件需要安装 openpyxl。")

        if suffix == ".xls":
            try:
                return pd.read_excel(path, engine="xlrd")
            except ImportError:
                raise ImportError("读取 .xls 文件需要安装 xlrd。")

        raise ValueError(f"暂不支持该文件格式：{suffix}")

    def detect_distribution_columns(self, df):
        columns = list(df.columns)
        lower_map = {str(col).strip().lower(): col for col in columns}

        d_candidates = [
            "d_mm", "d", "diameter", "grain_size", "grain size", "size",
            "particle_size", "粒径", "粒径(mm)", "粒径（mm）", "粒径/mm"
        ]
        fraction_candidates = [
            "fraction", "percent", "percentage", "p", "%", "mass_fraction",
            "volume_fraction", "含量", "百分比", "质量百分比", "体积百分比", "频率"
        ]

        d_col = None
        f_col = None

        for candidate in d_candidates:
            if candidate.lower() in lower_map:
                d_col = lower_map[candidate.lower()]
                break

        for candidate in fraction_candidates:
            if candidate.lower() in lower_map:
                f_col = lower_map[candidate.lower()]
                break

        if d_col is None:
            for col in columns:
                if any(candidate.lower() in str(col).lower() for candidate in d_candidates):
                    d_col = col
                    break

        if f_col is None:
            for col in columns:
                if any(candidate.lower() in str(col).lower() for candidate in fraction_candidates):
                    f_col = col
                    break

        if d_col is None or f_col is None:
            if len(columns) >= 2:
                d_col = columns[0]
                f_col = columns[1]
            else:
                raise ValueError("粒径分布文件至少需要两列：粒径列和比例列。")

        return d_col, f_col

    def read_grain_file(self):
        if self.is_representative_grain_mode():
            QMessageBox.information(self, "提示", "当前为代表粒径输入模式，请直接填写 D₁₆、D₅₀、D₈₄ 并点击“生成级配图”。")
            return

        file_path = self.grain_file.text().strip()
        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择粒径分布文件。")
            return

        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "提示", "粒径分布文件不存在。")
            return

        try:
            df = self.read_table_data(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"读取粒径分布文件时出错：\n{e}")
            return

        if df.empty:
            QMessageBox.warning(self, "提示", "粒径分布文件为空。")
            return

        df.columns = [str(col).strip() for col in df.columns]
        self.grain_df_original = df
        self.grain_df_standard = None
        self.source_mode = "file"

        try:
            self.grain_df_standard = self.build_standard_distribution_dataframe_from_file()
            self.preview_grain_data()
        except Exception as e:
            QMessageBox.warning(self, "数据检查失败", str(e))
            self.preview_box.setPlainText(f"粒径分布文件读取成功，但标准化失败：\n{e}")

    def build_standard_distribution_dataframe_from_file(self):
        if self.grain_df_original is None:
            raise ValueError("请先导入粒径分布文件。")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("当前环境未安装 pandas，请先安装 pandas。")

        d_col, f_col = self.detect_distribution_columns(self.grain_df_original)
        df = self.grain_df_original.copy()

        out = pd.DataFrame()
        out["D_mm"] = pd.to_numeric(df[d_col], errors="coerce")
        out["fraction_raw"] = pd.to_numeric(df[f_col], errors="coerce")
        out = out.dropna(subset=["D_mm", "fraction_raw"])
        out = out[(out["D_mm"] > 0) & (out["fraction_raw"] >= 0)]
        out = out.sort_values("D_mm").reset_index(drop=True)

        if out.empty:
            raise ValueError("标准化后粒径分布为空，请检查粒径列和比例列。")

        total = out["fraction_raw"].sum()
        if total <= 0:
            raise ValueError("粒径比例总和必须大于 0。")

        out["fraction"] = out["fraction_raw"] / total
        out["fraction_percent"] = out["fraction"] * 100.0
        out["cum_fraction"] = out["fraction"].cumsum()
        out["cum_percent"] = out["cum_fraction"] * 100.0

        return out[["D_mm", "fraction", "fraction_percent", "cum_fraction", "cum_percent"]]

    def get_required_float(self, line_edit, name):
        text = line_edit.text().strip()
        if text == "":
            raise ValueError(f"{name} 不能为空。")
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{name} 必须是数字。")
        return value

    def get_optional_float(self, line_edit):
        text = line_edit.text().strip()
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def build_distribution_from_percentiles(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("当前环境未安装 pandas，请先安装 pandas。")

        d16 = self.get_required_float(self.d16_edit, "D16")
        d50 = self.get_required_float(self.d50_edit, "D50")
        d84 = self.get_required_float(self.d84_edit, "D84")

        if not (d16 > 0 and d50 > 0 and d84 > 0):
            raise ValueError("D<sub>16</sub>、D<sub>50</sub>、D<sub>84</sub> 必须大于 0。")
        if not (d16 < d50 < d84):
            raise ValueError("必须满足 D16 < D50 < D84。")

        import math
        mu = math.log(d50)
        sigma = (math.log(d84) - math.log(d16)) / 2.0
        if sigma <= 0:
            raise ValueError("由 D16 和 D84 计算得到的分布宽度无效。")

        d_min = max(d16 / 6.0, 1e-4)
        d_max = d84 * 6.0
        edges = np.geomspace(d_min, d_max, 33)

        def lognormal_cdf(d):
            z = (np.log(d) - mu) / (sigma * np.sqrt(2.0))
            erf_vec = np.vectorize(math.erf)
            return 0.5 * (1.0 + erf_vec(z))

        cdf_edges = lognormal_cdf(edges)
        fractions = np.diff(cdf_edges)
        centers = np.sqrt(edges[:-1] * edges[1:])

        mask = fractions > 1e-6
        centers = centers[mask]
        fractions = fractions[mask]
        fractions = fractions / fractions.sum()

        out = pd.DataFrame({"D_mm": centers, "fraction": fractions})
        out["fraction_percent"] = out["fraction"] * 100.0
        out["cum_fraction"] = out["fraction"].cumsum()
        out["cum_percent"] = out["cum_fraction"] * 100.0

        return out[["D_mm", "fraction", "fraction_percent", "cum_fraction", "cum_percent"]]

    def get_current_distribution(self):
        if self.is_representative_grain_mode():
            self.source_mode = "percentiles"
            # 代表粒径模式每次都根据当前输入重新生成，避免沿用此前导入文件的缓存。
            self.grain_df_standard = self.build_distribution_from_percentiles()
            return self.grain_df_standard.copy()

        self.source_mode = "file"

        if self.grain_df_standard is not None and self.grain_df_original is not None:
            return self.grain_df_standard.copy()

        file_path = self.grain_file.text().strip()
        if not file_path:
            raise ValueError("当前为导入粒径分布模式，请先选择粒径分布文件；如果想用 D₁₆、D₅₀、D₈₄，请把输入方式切换为代表粒径输入。")

        path = Path(file_path)
        if not path.exists():
            raise ValueError("粒径分布文件不存在。")

        df = self.read_table_data(path)
        if df.empty:
            raise ValueError("粒径分布文件为空。")

        df.columns = [str(col).strip() for col in df.columns]
        self.grain_df_original = df
        self.grain_df_standard = self.build_standard_distribution_dataframe_from_file()
        return self.grain_df_standard.copy()

    def infer_percentile_from_distribution(self, distribution_df, p):
        df = distribution_df.sort_values("D_mm").copy()
        d = df["D_mm"].to_numpy(dtype=float)
        c = df["cum_fraction"].to_numpy(dtype=float)
        if len(d) == 0:
            return None
        if p <= c[0]:
            return float(d[0])
        if p >= c[-1]:
            return float(d[-1])
        return float(np.interp(p, c, d))

    def preview_grain_data(self):
        lines = []
        lines.append("泥沙粒径模块")
        lines.append("-" * 80)

        try:
            distribution_df = self.get_current_distribution()
        except Exception as e:
            self.preview_box.setPlainText(f"当前粒径数据检查失败：\n{e}")
            return

        if self.source_mode == "file":
            lines.append("输入方式：导入粒径分布文件")
            lines.append(f"文件路径：{self.grain_file.text().strip()}")
            d16 = self.infer_percentile_from_distribution(distribution_df, 0.16)
            d50 = self.infer_percentile_from_distribution(distribution_df, 0.50)
            d84 = self.infer_percentile_from_distribution(distribution_df, 0.84)
            lines.append(f"识别得到 D16 = {d16:.6g} mm")
            lines.append(f"识别得到 D50 = {d50:.6g} mm")
            lines.append(f"识别得到 D84 = {d84:.6g} mm")
            lines.append("保存输出：仅保存级配图 PNG")
        else:
            lines.append("输入方式：D<sub>16</sub>-D<sub>50</sub>-D<sub>84</sub> 生成近似级配")
            lines.append(f"D16 = {self.get_required_float(self.d16_edit, 'D16'):.6g} mm")
            lines.append(f"D50 = {self.get_required_float(self.d50_edit, 'D50'):.6g} mm")
            lines.append(f"D84 = {self.get_required_float(self.d84_edit, 'D84'):.6g} mm")
            lines.append("保存输出：保存近似级配 CSV + 级配图 PNG")

        rho_s = self.get_optional_float(self.rho_s_edit)
        if rho_s is not None:
            lines.append(f"颗粒密度 rho_s = {rho_s:.6g} kg/m³")

        lines.append("")
        lines.append("级配统计：")
        lines.append(f"粒径级数量：{len(distribution_df)}")
        lines.append(f"最小粒径：{distribution_df['D_mm'].min():.6g} mm")
        lines.append(f"最大粒径：{distribution_df['D_mm'].max():.6g} mm")
        lines.append(f"比例总和：{distribution_df['fraction'].sum():.6f}")
        lines.append("")
        lines.append("前 10 行预览：")
        lines.append(distribution_df.head(10).to_string(index=False))

        self.preview_box.setPlainText("\n".join(lines))

    def generate_grading_curve(self):
        try:
            distribution_df = self.get_current_distribution()
        except Exception as e:
            QMessageBox.warning(self, "数据错误", str(e))
            return

        self.preview_grain_data()
        self.plot_grain_distribution(distribution_df)

    def plot_grain_distribution(self, distribution_df=None, save_path=None):
        if distribution_df is None:
            distribution_df = self.get_current_distribution()

        self.figure.clear()
        self.figure.set_size_inches(12.5, 4.8)
        ax1 = self.figure.add_subplot(111)

        d = distribution_df["D_mm"].to_numpy(dtype=float)
        frac_percent = distribution_df["fraction_percent"].to_numpy(dtype=float)
        cum_percent = distribution_df["cum_percent"].to_numpy(dtype=float)

        width = d * 0.22
        ax1.bar(d, frac_percent, width=width, align="center", alpha=0.75, label="Fraction (%)")
        ax1.set_xscale("log")
        ax1.set_xlabel("Grain size D (mm)")
        ax1.set_ylabel("Fraction (%)")
        ax1.grid(True, alpha=0.3, which="both")

        ax2 = ax1.twinx()
        ax2.plot(d, cum_percent, marker="o", linewidth=1.2, markersize=3, label="Cumulative (%)")
        ax2.set_ylabel("Cumulative (%)")
        ax2.set_ylim(0, 105)

        try:
            if self.source_mode == "percentiles":
                d16 = self.get_required_float(self.d16_edit, "D16")
                d50 = self.get_required_float(self.d50_edit, "D50")
                d84 = self.get_required_float(self.d84_edit, "D84")
            else:
                d16 = self.infer_percentile_from_distribution(distribution_df, 0.16)
                d50 = self.infer_percentile_from_distribution(distribution_df, 0.50)
                d84 = self.infer_percentile_from_distribution(distribution_df, 0.84)

            for value, label in [(d16, "D16"), (d50, "D50"), (d84, "D84")]:
                if value is not None and value > 0:
                    ax1.axvline(value, linestyle="--", linewidth=1.0)
                    ax1.text(value, ax1.get_ylim()[1] * 0.92, label, rotation=90, ha="right", va="top", fontsize=9)
        except Exception:
            pass

        ax1.set_title("Grain-size distribution curve")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper left")
        self.figure.tight_layout()

        if save_path is not None:
            self.figure.savefig(save_path, dpi=300, bbox_inches="tight")

        self.canvas.draw()

    def save_outputs(self):
        try:
            distribution_df = self.get_current_distribution()
        except Exception as e:
            QMessageBox.warning(self, "数据错误", str(e))
            return

        output_text = self.output_dir.text().strip()
        if output_text:
            output_dir = Path(output_text)
        else:
            if self.grain_file.text().strip():
                output_dir = Path(self.grain_file.text().strip()).parent / "sediment_outputs"
            else:
                output_dir = Path.cwd() / "sediment_outputs"
            self.output_dir.setText(str(output_dir))

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "grain_size_distribution.png"

        try:
            self.plot_grain_distribution(distribution_df=distribution_df, save_path=plot_path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存级配图时出错：\n{e}")
            return

        try:
            if self.source_mode == "percentiles":
                csv_path = output_dir / "generated_grain_distribution.csv"
            else:
                csv_path = output_dir / "standard_grain_distribution.csv"

            distribution_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "保存成功", "已保存：\n" f"{csv_path}\n" f"{plot_path}")
            self.sediment_saved.emit(str(csv_path))
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存输出文件时出错：\n{e}")

    def clear_page(self):
        self.grain_file.clear()
        self.output_dir.clear()
        self.d16_edit.clear()
        self.d50_edit.clear()
        self.d84_edit.clear()
        self.rho_s_edit.setText("2650")
        self.preview_box.clear()
        self.figure.clear()
        self.canvas.draw()
        self.grain_df_original = None
        self.grain_df_standard = None
        self.source_mode = None
        if hasattr(self, "sediment_input_mode_combo"):
            self.sediment_input_mode_combo.setCurrentIndex(0)
        self.update_sediment_input_mode_ui()


class HydraulicGeometryPage(QWidget):
    """模块 5：水力参数模块。"""

    hydraulic_saved = Signal(str)

    def __init__(self):
        super().__init__()

        self.cross_df_original = None
        self.water_df_original = None
        self.cross_df_standard = None
        self.water_df_standard = None
        self.effective_df = None

        self.x_nodes = None
        self.z_nodes = None
        self.node_widths = None

        # 内置默认参数：不在界面显示。
        self.rho_s_default = 2650.0
        self.kappa_default = 0.4
        self.rouse_threshold_default = 2.5
        self.active_threshold_percent_default = 5.0
        self.transport_exponent_default = 1.5
        self.g_default = 9.81
        self.kinematic_viscosity_default = 1.0e-6

        # 加速用：沉降速度反算 D_min 的查找表缓存。
        # 原来每个时间、每个断面点都做 80 次二分迭代，会明显拖慢界面。
        self._settling_lookup_key = None
        self._settling_D_grid = None
        self._settling_ws_grid = None

        self.build_ui()

    def build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("水力参数")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        subtitle = QLabel(
            "导入河道断面和水位序列，计算有效水深 H<sub>eff</sub>(t)、有效宽度 W<sub>eff</sub>(t) "
            "以及由 H<sub>eff</sub> 代表的有效运动粒径范围。完整级配筛选留到后续反演模块完成。"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 14px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        file_group = QGroupBox("数据文件")
        file_layout = QFormLayout(file_group)
        file_layout.setSpacing(10)

        self.hydraulic_mode_combo = QComboBox()
        self.hydraulic_mode_combo.addItems([
            "有河道断面：水位 + 断面计算",
            "无河道断面：水深 + 默认宽度计算",
        ])
        self.hydraulic_mode_combo.currentIndexChanged.connect(self.update_hydraulic_mode_ui)

        self.cross_file = QLineEdit()
        self.water_file = QLineEdit()
        self.default_width_edit = QLineEdit("10")
        self.output_dir = QLineEdit()

        file_layout.addRow("计算方式：", self.hydraulic_mode_combo)
        self.cross_file_widget = self.make_path_row(self.cross_file, self.choose_cross_file, button_attr="cross_browse_button")
        self.water_file_widget = self.make_path_row(self.water_file, self.choose_water_file, button_attr="water_browse_button")
        self.output_dir_widget = self.make_path_row(self.output_dir, self.choose_output_dir, button_attr="output_browse_button")
        file_layout.addRow("河道断面文件：", self.cross_file_widget)
        file_layout.addRow("默认有效宽度 W<sub>eff</sub>，m：", self.default_width_edit)
        file_layout.addRow("水位/水深序列文件：", self.water_file_widget)
        file_layout.addRow("输出文件夹：", self.output_dir_widget)

        layout.addWidget(file_group)

        param_group = QGroupBox("水力与起动参数")
        param_layout = QFormLayout(param_group)
        param_layout.setSpacing(10)

        self.slope_edit = QLineEdit("0.01")
        self.rho_w_edit = QLineEdit("1000")
        self.theta_c_edit = QLineEdit("0.045")

        param_layout.addRow("河道坡降 S：", self.slope_edit)
        param_layout.addRow("水体密度 ρ<sub>w</sub>，kg/m³：", self.rho_w_edit)
        param_layout.addRow("临界 Shields 参数 θ<sub>c</sub>：", self.theta_c_edit)

        layout.addWidget(param_group)

        button_layout = QHBoxLayout()

        self.read_data_button = QPushButton("导入数据")
        self.plot_water_button = QPushButton("绘制水位过程")
        self.compute_button = QPushButton("计算水力参数")
        self.save_button = QPushButton("保存结果")
        self.clear_button = QPushButton("清空")

        self.read_data_button.clicked.connect(self.read_all_data)
        self.plot_water_button.clicked.connect(lambda checked=False: self.plot_water_level())
        self.compute_button.clicked.connect(self.compute_hydraulic_parameters)
        self.save_button.clicked.connect(self.save_results)
        self.clear_button.clicked.connect(self.clear_page)

        button_layout.addWidget(self.read_data_button)
        button_layout.addWidget(self.plot_water_button)
        button_layout.addWidget(self.compute_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(190)
        self.preview_box.setMaximumHeight(280)
        self.preview_box.setPlaceholderText("数据读取、水力参数和有效运动粒径范围会显示在这里。")
        layout.addWidget(self.preview_box)

        plot_title = QLabel("水位/水深过程图预览")
        plot_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 8px;")
        layout.addWidget(plot_title)

        self.figure = Figure(figsize=(12.5, 4.8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(430)
        layout.addWidget(self.canvas)

        note = QLabel(
            "输出文件：effective_depth_width.csv 和 water_level_timeseries.png。"
            "D_effective_min_mm 与 D_effective_max_mm 使用最终 H<sub>eff</sub>(t) 重新计算。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 13px;")
        layout.addWidget(note)
        layout.addStretch()

        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)
        self.update_hydraulic_mode_ui()

    def make_path_row(self, line_edit, function, button_attr=None):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(function)
        if button_attr:
            setattr(self, button_attr, browse_button)

        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)

        return row

    def is_depth_only_mode(self):
        return hasattr(self, "hydraulic_mode_combo") and self.hydraulic_mode_combo.currentIndex() == 1

    def update_hydraulic_mode_ui(self):
        depth_mode = self.is_depth_only_mode()

        # 两种模式互斥：
        # 有断面模式需要河道断面文件，不需要默认宽度；
        # 无断面模式不使用河道断面文件，需要默认有效宽度。
        self.cross_file.setEnabled(not depth_mode)
        if hasattr(self, "cross_browse_button"):
            self.cross_browse_button.setEnabled(not depth_mode)
        self.default_width_edit.setEnabled(depth_mode)

        if depth_mode:
            if self.cross_df_standard is not None:
                self.cross_df_standard = None
            self.preview_box.setPlaceholderText(
                "无断面模式：读取 time + water_depth/depth（水深）表，使用默认有效宽度 W_eff 计算 H_eff、W_eff 和有效粒径范围。"
            )
        else:
            self.preview_box.setPlaceholderText(
                "有断面模式：读取河道断面和水位序列，自动计算 H_eff、W_eff 和有效粒径范围；默认宽度输入框不参与计算。"
            )

    def choose_cross_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择河道断面文件", "", "Data Files (*.csv *.xlsx *.xls *.txt);;All Files (*)")
        if path:
            self.cross_file.setText(path)
            if not self.output_dir.text().strip():
                self.output_dir.setText(str(Path(path).parent / "hydraulic_outputs"))

    def choose_water_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择水位/水深序列文件", "", "Data Files (*.csv *.xlsx *.xls *.txt);;All Files (*)")
        if path:
            self.water_file.setText(path)
            if not self.output_dir.text().strip():
                self.output_dir.setText(str(Path(path).parent / "hydraulic_outputs"))

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir.setText(path)

    def append_preview(self, text):
        current = self.preview_box.toPlainText().strip()
        if current:
            self.preview_box.setPlainText(current + "\n" + text)
        else:
            self.preview_box.setPlainText(text)

    def read_table_data(self, path):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("当前环境未安装 pandas，请先安装 pandas。")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            for encoding in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    return pd.read_csv(path, encoding=encoding)
                except Exception:
                    pass
            return pd.read_csv(path)

        if suffix == ".txt":
            for encoding in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
                except Exception:
                    pass
            return pd.read_csv(path, sep=None, engine="python")

        if suffix == ".xlsx":
            try:
                return pd.read_excel(path, engine="openpyxl")
            except ImportError:
                raise ImportError("读取 .xlsx 文件需要安装 openpyxl。")

        if suffix == ".xls":
            try:
                return pd.read_excel(path, engine="xlrd")
            except ImportError:
                raise ImportError("读取 .xls 文件需要安装 xlrd。")

        raise ValueError(f"暂不支持该文件格式：{suffix}")

    def read_all_data(self):
        depth_mode = self.is_depth_only_mode()

        if not depth_mode and not self.cross_file.text().strip():
            QMessageBox.warning(self, "提示", "有断面模式下请先选择河道断面文件。")
            return
        if not self.water_file.text().strip():
            QMessageBox.warning(self, "提示", "请先选择水位/水深序列文件。")
            return

        if depth_mode:
            self.cross_df_standard = None
            ok_cross = True
        else:
            ok_cross = self.read_cross_file(show_message=False)
        ok_water = self.read_water_file(show_message=False)

        if ok_cross and ok_water:
            self.preview_loaded_data()
            if depth_mode:
                QMessageBox.information(self, "导入完成", "水深序列已导入，将使用默认宽度计算水力参数。")
            else:
                QMessageBox.information(self, "导入完成", "河道断面和水位序列已导入。")

    def detect_column(self, columns, candidates, fallback_index):
        lower_map = {str(col).strip().lower(): col for col in columns}
        for candidate in candidates:
            key = candidate.lower()
            if key in lower_map:
                return lower_map[key]
        for col in columns:
            col_lower = str(col).strip().lower()
            for candidate in candidates:
                if candidate.lower() in col_lower:
                    return col
        if len(columns) > fallback_index:
            return columns[fallback_index]
        raise ValueError("数据列数量不足。")

    def read_cross_file(self, show_message=True):
        path = Path(self.cross_file.text().strip())
        if not path.exists():
            QMessageBox.warning(self, "提示", "河道断面文件不存在。")
            return False

        try:
            df = self.read_table_data(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"读取河道断面时出错：\n{e}")
            return False

        if df.empty or len(df.columns) < 2:
            QMessageBox.warning(self, "提示", "河道断面文件为空或少于两列。")
            return False

        df.columns = [str(col).strip() for col in df.columns]
        self.cross_df_original = df

        try:
            self.cross_df_standard = self.build_standard_cross_dataframe()
            if show_message:
                self.preview_loaded_data()
        except Exception as e:
            QMessageBox.warning(self, "数据错误", str(e))
            return False

        return True

    def read_water_file(self, show_message=True):
        path = Path(self.water_file.text().strip())
        if not path.exists():
            QMessageBox.warning(self, "提示", "水位序列文件不存在。")
            return False

        try:
            df = self.read_table_data(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"读取水位序列时出错：\n{e}")
            return False

        if df.empty or len(df.columns) < 2:
            QMessageBox.warning(self, "提示", "水位序列文件为空或少于两列。")
            return False

        df.columns = [str(col).strip() for col in df.columns]
        self.water_df_original = df

        try:
            self.water_df_standard = self.build_standard_water_dataframe()
            if show_message:
                self.preview_loaded_data()
        except Exception as e:
            QMessageBox.warning(self, "数据错误", str(e))
            return False

        return True

    def clean_numeric_series(self, series):
        return series.astype(str).str.strip().str.replace(",", "", regex=False)

    def build_standard_cross_dataframe(self):
        if self.cross_df_original is None:
            raise ValueError("请先读取河道断面文件。")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("当前环境未安装 pandas，请先安装 pandas。")

        df = self.cross_df_original.copy()
        columns = list(df.columns)

        x_col = self.detect_column(columns, ["x", "distance", "station", "offset", "起点距", "横向距离", "距离", "断面距离"], 0)
        z_col = self.detect_column(columns, ["bed_elevation", "bed elevation", "elevation", "bed", "z", "河床高程", "高程", "河底高程", "床面高程"], 1)

        out = pd.DataFrame()
        out["x_m"] = pd.to_numeric(self.clean_numeric_series(df[x_col]), errors="coerce")
        out["bed_elevation_m"] = pd.to_numeric(self.clean_numeric_series(df[z_col]), errors="coerce")
        out = out.dropna(subset=["x_m", "bed_elevation_m"])

        # 如果文件没有表头，pandas 可能把第一行数据当作列名；这里自动把列名也作为首行数据尝试补回。
        if len(columns) >= 2:
            header_x = pd.to_numeric(self.clean_numeric_series(pd.Series([columns[0]])), errors="coerce").iloc[0]
            header_z = pd.to_numeric(self.clean_numeric_series(pd.Series([columns[1]])), errors="coerce").iloc[0]
            if pd.notna(header_x) and pd.notna(header_z):
                header_row = pd.DataFrame({"x_m": [header_x], "bed_elevation_m": [header_z]})
                out = pd.concat([header_row, out], ignore_index=True)

        out = out.sort_values("x_m").drop_duplicates(subset=["x_m"], keep="first").reset_index(drop=True)

        if len(out) < 2:
            raise ValueError("标准化后的河道断面点少于 2 个，无法计算。默认第一列为 x、第二列为河床高程。")

        return out

    def parse_time_series(self, values):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("当前环境未安装 pandas，请先安装 pandas。")

        # 先保留原始对象直接解析，兼容 Excel 读入的 Timestamp/datetime。
        parsed = pd.to_datetime(values, errors="coerce")

        # 再尝试字符串解析，兼容中文时间、带空格时间等。
        text_values = values.astype(str).str.strip()
        text_values = text_values.str.replace("年", "-", regex=False)
        text_values = text_values.str.replace("月", "-", regex=False)
        text_values = text_values.str.replace("日", " ", regex=False)
        text_values = text_values.str.replace("/", "-", regex=False)

        parsed_text = pd.to_datetime(text_values, errors="coerce")
        if parsed_text.notna().sum() > parsed.notna().sum():
            parsed = parsed_text

        # 最后尝试 Excel 日期序列号。
        numeric_values = pd.to_numeric(values, errors="coerce")
        parsed_excel = pd.to_datetime(numeric_values, unit="D", origin="1899-12-30", errors="coerce")
        if parsed_excel.notna().sum() > parsed.notna().sum():
            parsed = parsed_excel

        return parsed

    def build_standard_water_dataframe(self):
        if self.water_df_original is None:
            raise ValueError("请先读取水位序列文件。")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("当前环境未安装 pandas，请先安装 pandas。")

        df = self.water_df_original.copy()
        columns = list(df.columns)

        # 第一优先级：按列名识别；识别不到时自动回退为第一列时间、第二列数值。
        # 有断面模式：第二列解释为水位；无断面模式：第二列解释为水深。
        time_col = self.detect_column(columns, ["time", "datetime", "date", "timestamp", "时间", "日期", "观测时间"], 0)
        if self.is_depth_only_mode():
            value_candidates = [
                "water_depth", "water depth", "depth", "flowdepth", "flow depth",
                "h", "H", "水深", "流深", "平均水深", "有效水深"
            ]
        else:
            value_candidates = [
                "water_level", "water level", "stage", "level",
                "水位", "水位高程", "水位(m)", "水位（m）", "水位_m"
            ]
        water_col = self.detect_column(columns, value_candidates, 1)

        out = pd.DataFrame()
        out["time"] = self.parse_time_series(df[time_col])
        # 为了兼容后续反演模块，统一保留列名 water_level_m。
        # 无断面模式下，这一列表示输入水深。
        out["water_level_m"] = pd.to_numeric(self.clean_numeric_series(df[water_col]), errors="coerce")
        out = out.dropna(subset=["time", "water_level_m"])

        # 如果文件没有表头，pandas 可能把第一行数据当作列名；这里把列名也作为首行数据尝试补回。
        if len(columns) >= 2:
            header_df = pd.DataFrame({"raw_time": [columns[0]], "raw_water": [columns[1]]})
            header_time = self.parse_time_series(header_df["raw_time"])
            header_water = pd.to_numeric(self.clean_numeric_series(header_df["raw_water"]), errors="coerce")
            if header_time.notna().iloc[0] and header_water.notna().iloc[0]:
                header_row = pd.DataFrame({"time": [header_time.iloc[0]], "water_level_m": [header_water.iloc[0]]})
                out = pd.concat([header_row, out], ignore_index=True)

        # 如果按列名识别失败，强制使用第一列和第二列再试一次。
        if out.empty and len(columns) >= 2:
            first = columns[0]
            second = columns[1]
            fallback = pd.DataFrame()
            fallback["time"] = self.parse_time_series(df[first])
            fallback["water_level_m"] = pd.to_numeric(self.clean_numeric_series(df[second]), errors="coerce")
            fallback = fallback.dropna(subset=["time", "water_level_m"])
            out = fallback

        out = out.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

        if out.empty:
            preview = df.head(5).to_string(index=False)
            raise ValueError(
                "标准化后的水位/水深数据为空。程序已按‘第一列=时间、第二列=水位/水深’尝试解析，但仍失败。\n\n"
                "请检查：\n"
                "1. 第一列是否为可识别时间，例如 2025-07-01 00:00；\n"
                "2. 第二列是否为纯数字水位或水深；\n"
                "3. Excel 是否有合并单元格、说明行或空行。\n\n"
                f"文件前 5 行预览：\n{preview}"
            )

        return out

    def get_required_float(self, line_edit, name):
        text = line_edit.text().strip()
        if text == "":
            raise ValueError(f"{name} 不能为空。")
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{name} 必须是数字。")
        return value

    def get_hydraulic_settings(self):
        slope = self.get_required_float(self.slope_edit, "河道坡降 S")
        rho_w = self.get_required_float(self.rho_w_edit, "水体密度 rho_w")
        theta_c = self.get_required_float(self.theta_c_edit, "临界 Shields 参数 theta_c")
        default_width = self.get_required_float(self.default_width_edit, "默认有效宽度 W")

        if slope <= 0:
            raise ValueError("河道坡降 S 必须大于 0。")
        if default_width <= 0:
            raise ValueError("默认有效宽度 W 必须大于 0。")
        if rho_w <= 0:
            raise ValueError("水体密度 rho_w 必须大于 0。")
        if theta_c <= 0:
            raise ValueError("临界 Shields 参数 theta_c 必须大于 0。")
        if self.rho_s_default <= rho_w:
            raise ValueError("内置颗粒密度必须大于水体密度。")

        return {
            "slope": slope,
            "rho_s_kg_m3": self.rho_s_default,
            "rho_w_kg_m3": rho_w,
            "default_width_m": default_width,
            "relative_submerged_density_Rs": (self.rho_s_default - rho_w) / rho_w,
            "theta_c": theta_c,
            "kappa": self.kappa_default,
            "rouse_threshold": self.rouse_threshold_default,
            "active_threshold_percent": self.active_threshold_percent_default,
            "transport_exponent_beta": self.transport_exponent_default,
            "g": self.g_default,
            "kinematic_viscosity": self.kinematic_viscosity_default,
        }

    def preview_loaded_data(self):
        lines = []
        lines.append("水力参数模块数据检查")
        lines.append("-" * 80)

        depth_mode = self.is_depth_only_mode()
        lines.append("计算方式：" + ("无河道断面：水深 + 默认宽度" if depth_mode else "有河道断面：水位 + 河道断面"))
        lines.append("")

        if not depth_mode and self.cross_df_standard is not None:
            cross_df = self.cross_df_standard
            lines.append("河道断面：已读取")
            lines.append(f"断面点数：{len(cross_df)}")
            lines.append(f"x 范围：{cross_df['x_m'].min():.3f} - {cross_df['x_m'].max():.3f} m")
            lines.append(f"河床高程范围：{cross_df['bed_elevation_m'].min():.3f} - {cross_df['bed_elevation_m'].max():.3f} m")
        elif depth_mode:
            lines.append("河道断面：未使用")
            lines.append(f"默认有效宽度 W：{self.default_width_edit.text().strip() or '未设置'} m")
        else:
            lines.append("河道断面：未读取")

        lines.append("")

        if self.water_df_standard is not None:
            water_df = self.water_df_standard
            value_name = "水深" if depth_mode else "水位"
            lines.append(f"{value_name}序列：已读取")
            lines.append(f"{value_name}行数：{len(water_df)}")
            lines.append(f"开始时间：{water_df['time'].iloc[0]}")
            lines.append(f"结束时间：{water_df['time'].iloc[-1]}")
            lines.append(f"{value_name}范围：{water_df['water_level_m'].min():.3f} - {water_df['water_level_m'].max():.3f} m")
        else:
            lines.append("水位/水深序列：未读取")

        lines.append("")
        lines.append("说明：模块 5 不读取完整级配表；完整级配筛选将在后续反演模块用模块 4 级配表和本模块粒径范围完成。")
        if depth_mode:
            lines.append("无断面模式下，输出表中的 water_level_m 为输入水深值，用于兼容后续反演流程。")

        self.preview_box.setPlainText("\n".join(lines))

    def compute_node_widths(self, x):
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            return np.zeros_like(x)
        widths = np.zeros_like(x, dtype=float)
        widths[0] = 0.5 * (x[1] - x[0])
        widths[-1] = 0.5 * (x[-1] - x[-2])
        if len(x) > 2:
            widths[1:-1] = 0.5 * (x[2:] - x[:-2])
        widths[widths < 0] = 0.0
        return widths

    def compute_segment_geometry(self, x, z, water_level):
        wetted_width = 0.0
        wetted_area = 0.0
        wetted_perimeter = 0.0

        for i in range(len(x) - 1):
            x1 = float(x[i])
            x2 = float(x[i + 1])
            z1 = float(z[i])
            z2 = float(z[i + 1])
            dx = x2 - x1
            dz = z2 - z1

            if dx <= 0:
                continue

            h1 = water_level - z1
            h2 = water_level - z2

            if h1 <= 0 and h2 <= 0:
                continue

            segment_len = float(np.sqrt(dx * dx + dz * dz))

            if h1 > 0 and h2 > 0:
                wetted_width += dx
                wetted_area += 0.5 * (h1 + h2) * dx
                wetted_perimeter += segment_len
                continue

            if h1 > 0 and h2 <= 0:
                ratio = h1 / (h1 - h2)
                wet_dx = ratio * dx
                wet_dz = ratio * dz
                wetted_width += wet_dx
                wetted_area += 0.5 * h1 * wet_dx
                wetted_perimeter += float(np.sqrt(wet_dx * wet_dx + wet_dz * wet_dz))
                continue

            if h1 <= 0 and h2 > 0:
                ratio = (-h1) / (h2 - h1)
                wet_dx = (1.0 - ratio) * dx
                wet_dz = (1.0 - ratio) * dz
                wetted_width += wet_dx
                wetted_area += 0.5 * h2 * wet_dx
                wetted_perimeter += float(np.sqrt(wet_dx * wet_dx + wet_dz * wet_dz))
                continue

        hydraulic_radius = wetted_area / wetted_perimeter if wetted_perimeter > 0 else 0.0
        mean_depth = wetted_area / wetted_width if wetted_width > 0 else 0.0
        return wetted_width, wetted_area, wetted_perimeter, hydraulic_radius, mean_depth

    def settling_velocity(self, D_m, Rs, g, nu):
        D_m = float(D_m)
        if D_m <= 0:
            return 0.0
        C1 = 18.0
        C2 = 1.0
        denominator = C1 * nu + np.sqrt(0.75 * C2 * Rs * g * D_m ** 3)
        if denominator <= 0:
            return 0.0
        return float(Rs * g * D_m ** 2 / denominator)

    def settling_velocity_array(self, D_m, Rs, g, nu):
        D_m = np.asarray(D_m, dtype=float)
        D_safe = np.maximum(D_m, 1.0e-12)
        C1 = 18.0
        C2 = 1.0
        denominator = C1 * nu + np.sqrt(0.75 * C2 * Rs * g * D_safe ** 3)
        ws = Rs * g * D_safe ** 2 / denominator
        ws[D_m <= 0] = 0.0
        return ws

    def prepare_settling_lookup(self, settings):
        """预先建立 w_s(D) 查找表，用插值代替逐点 80 次二分迭代。"""
        Rs = float(settings["relative_submerged_density_Rs"])
        g = float(settings["g"])
        nu = float(settings["kinematic_viscosity"])

        key = (round(Rs, 12), round(g, 12), round(nu, 16))

        if self._settling_lookup_key == key and self._settling_D_grid is not None:
            return

        D_grid = np.geomspace(1.0e-7, 2.0, 5000)  # m，0.0001 mm 到 2000 mm
        ws_grid = self.settling_velocity_array(D_grid, Rs, g, nu)

        # np.interp 要求横坐标单调递增。这里理论上单调，为稳妥仍做一次清理。
        order = np.argsort(ws_grid)
        ws_grid = ws_grid[order]
        D_grid = D_grid[order]

        keep = np.concatenate([[True], np.diff(ws_grid) > 0])
        self._settling_ws_grid = ws_grid[keep]
        self._settling_D_grid = D_grid[keep]
        self._settling_lookup_key = key

    def inverse_settling_velocity(self, target_ws, settings):
        """由目标沉降速度反查粒径 D，支持标量和数组输入，单位：m。"""
        self.prepare_settling_lookup(settings)

        target = np.asarray(target_ws, dtype=float)
        target_safe = np.maximum(target, 0.0)

        D = np.interp(
            target_safe,
            self._settling_ws_grid,
            self._settling_D_grid,
            left=0.0,
            right=self._settling_D_grid[-1],
        )

        if np.isscalar(target_ws):
            return float(D)
        return D

    def solve_d_from_settling_velocity(self, target_ws, Rs, g, nu):
        """保留旧接口，内部改为查找表插值。"""
        settings = {
            "relative_submerged_density_Rs": Rs,
            "g": g,
            "kinematic_viscosity": nu,
        }
        return self.inverse_settling_velocity(target_ws, settings)

    def local_activity_weights_vectorized(self, h, settings):
        """
        向量化计算每个断面节点的活动权重 G_i。

        原逻辑是每个 h_i 单独调用 local_activity_weight()，并且每次都二分反解 D_min；
        现在改为数组一次性计算，速度会快很多。
        """
        h = np.asarray(h, dtype=float)
        G = np.zeros_like(h, dtype=float)
        D_min_mm = np.full_like(h, np.nan, dtype=float)
        D_max_mm = np.full_like(h, np.nan, dtype=float)

        slope = float(settings["slope"])
        Rs = float(settings["relative_submerged_density_Rs"])
        theta_c = float(settings["theta_c"])
        kappa = float(settings["kappa"])
        rouse_threshold = float(settings["rouse_threshold"])
        beta = float(settings["transport_exponent_beta"])
        g = float(settings["g"])

        wet = h > 0
        if not np.any(wet):
            return G, D_min_mm, D_max_mm

        h_wet = h[wet]

        D_max = h_wet * slope / (Rs * theta_c)
        u_star = np.sqrt(g * h_wet * slope)
        target_ws = rouse_threshold * kappa * u_star
        D_min = self.inverse_settling_velocity(target_ws, settings)

        D_min_mm[wet] = D_min * 1000.0
        D_max_mm[wet] = D_max * 1000.0

        valid = np.isfinite(D_min) & np.isfinite(D_max) & (D_min > 0) & (D_max > D_min)
        if not np.any(valid):
            return G, D_min_mm, D_max_mm

        h_v = h_wet[valid]
        D_min_v = D_min[valid]
        D_max_v = D_max[valid]

        n_samples = 24
        q = np.linspace(0.0, 1.0, n_samples)
        ratio = D_max_v / D_min_v
        D_samples = D_min_v[:, None] * ratio[:, None] ** q[None, :]

        theta_samples = h_v[:, None] * slope / (Rs * D_samples)
        excess = np.maximum(theta_samples - theta_c, 0.0)
        y = excess ** beta

        dlog = (np.log(D_max_v) - np.log(D_min_v)) / (n_samples - 1)
        weight_valid = (0.5 * y[:, 0] + np.sum(y[:, 1:-1], axis=1) + 0.5 * y[:, -1]) * dlog

        wet_indices = np.where(wet)[0]
        valid_indices = wet_indices[valid]
        G[valid_indices] = weight_valid

        return G, D_min_mm, D_max_mm

    def representative_effective_grain_range(self, H_eff, settings):
        if H_eff <= 0:
            return np.nan, np.nan

        slope = settings["slope"]
        Rs = settings["relative_submerged_density_Rs"]
        theta_c = settings["theta_c"]
        kappa = settings["kappa"]
        rouse_threshold = settings["rouse_threshold"]
        g = settings["g"]

        D_max_m = H_eff * slope / (Rs * theta_c)
        u_star_eff = np.sqrt(g * H_eff * slope)
        target_ws = rouse_threshold * kappa * u_star_eff
        D_min_m = self.inverse_settling_velocity(target_ws, settings)

        if D_max_m <= D_min_m:
            return np.nan, np.nan

        return D_min_m * 1000.0, D_max_m * 1000.0

    def compute_hydraulic_parameters(self):
        depth_mode = self.is_depth_only_mode()
        try:
            if depth_mode:
                self.cross_df_standard = None
                if self.water_df_standard is None:
                    if not self.read_water_file(show_message=False):
                        return
            else:
                if self.cross_df_standard is None:
                    if not self.read_cross_file(show_message=False):
                        return
                if self.water_df_standard is None:
                    if not self.read_water_file(show_message=False):
                        return
            settings = self.get_hydraulic_settings()
            self.prepare_settling_lookup(settings)
        except Exception as e:
            QMessageBox.warning(self, "数据错误", str(e))
            return

        water_df = self.water_df_standard
        rows = []

        self.preview_box.setPlainText("正在计算水力参数，请稍候...\n已启用向量化加速。")
        QApplication.processEvents()

        if depth_mode:
            default_width = float(settings["default_width_m"])
            for idx, row in water_df.iterrows():
                time_value = row["time"]
                water_depth = max(float(row["water_level_m"]), 0.0)
                H_eff = water_depth
                W_eff = default_width if H_eff > 0 else 0.0
                D_eff_min_mm, D_eff_max_mm = self.representative_effective_grain_range(H_eff, settings)
                rows.append(
                    {
                        "time": time_value,
                        "water_level_m": water_depth,
                        "H_eff_m": H_eff,
                        "W_eff_m": W_eff,
                        "D_effective_min_mm": D_eff_min_mm,
                        "D_effective_max_mm": D_eff_max_mm,
                    }
                )
                if idx % 500 == 0:
                    self.progress_bar.setValue(int(100 * (idx + 1) / max(len(water_df), 1)))
                    QApplication.processEvents()
        else:
            cross_df = self.cross_df_standard
            x = cross_df["x_m"].to_numpy(dtype=float)
            z = cross_df["bed_elevation_m"].to_numpy(dtype=float)
            node_widths = self.compute_node_widths(x)

            self.x_nodes = x
            self.z_nodes = z
            self.node_widths = node_widths

            active_threshold_ratio = settings["active_threshold_percent"] / 100.0

            for idx, row in water_df.iterrows():
                time_value = row["time"]
                water_level = float(row["water_level_m"])
                h = np.maximum(water_level - z, 0.0)

                G, _, _ = self.local_activity_weights_vectorized(h, settings)

                total_weight = float(np.sum(G * node_widths))
                if total_weight > 0:
                    H_eff = float(np.sum(h * G * node_widths) / total_weight)
                else:
                    H_eff = 0.0

                G_max = float(np.nanmax(G)) if len(G) else 0.0
                if G_max > 0:
                    active_mask = (G > active_threshold_ratio * G_max) & (h > 0)
                    W_eff = float(np.sum(node_widths[active_mask]))
                else:
                    W_eff = 0.0

                D_eff_min_mm, D_eff_max_mm = self.representative_effective_grain_range(H_eff, settings)

                rows.append(
                    {
                        "time": time_value,
                        "water_level_m": water_level,
                        "H_eff_m": H_eff,
                        "W_eff_m": W_eff,
                        "D_effective_min_mm": D_eff_min_mm,
                        "D_effective_max_mm": D_eff_max_mm,
                    }
                )

                if idx % 200 == 0:
                    self.progress_bar.setValue(int(100 * (idx + 1) / max(len(water_df), 1)))
                    QApplication.processEvents()

        try:
            import pandas as pd
        except ImportError:
            QMessageBox.critical(self, "缺少依赖", "当前环境未安装 pandas，请先安装 pandas。")
            return

        self.effective_df = pd.DataFrame(rows)
        self.progress_bar.setValue(100)

        lines = []
        lines.append("水力参数计算完成")
        lines.append("-" * 80)
        lines.append("计算方式：" + ("无河道断面：水深 + 默认宽度" if depth_mode else "有河道断面：水位 + 河道断面"))
        lines.append(f"计算时刻数：{len(self.effective_df)}")
        lines.append(f"H_eff 范围：{self.effective_df['H_eff_m'].min():.4f} - {self.effective_df['H_eff_m'].max():.4f} m")
        lines.append(f"W_eff 范围：{self.effective_df['W_eff_m'].min():.4f} - {self.effective_df['W_eff_m'].max():.4f} m")
        valid_min = self.effective_df['D_effective_min_mm'].dropna()
        valid_max = self.effective_df['D_effective_max_mm'].dropna()
        if len(valid_min) and len(valid_max):
            lines.append(f"D_effective_min 范围：{valid_min.min():.4g} - {valid_min.max():.4g} mm")
            lines.append(f"D_effective_max 范围：{valid_max.min():.4g} - {valid_max.max():.4g} mm")
        if depth_mode:
            lines.append("")
            lines.append("说明：无断面模式中 H_eff = 输入水深，W_eff = 默认有效宽度。")
            lines.append("输出列名 water_level_m 保持不变，用于兼容后续反演模块；此时该列表示输入水深。")
        lines.append("")
        lines.append("前 10 行结果：")
        lines.append(self.effective_df.head(10).to_string(index=False))
        self.preview_box.setPlainText("\n".join(lines))

    def plot_water_level(self, save_path=None):
        # QPushButton.clicked 会自动传入一个 bool checked 参数。
        # 如果直接 connect(self.plot_water_level)，这个 bool 会被当成 save_path，导致 savefig 报错：
        # AttributeError: 'bool' object has no attribute 'write'
        if isinstance(save_path, bool):
            save_path = None

        try:
            if self.water_df_standard is None:
                if not self.read_water_file(show_message=False):
                    return
            water_df = self.water_df_standard
        except Exception as e:
            QMessageBox.warning(self, "数据错误", str(e))
            return

        self.figure.clear()
        self.figure.set_size_inches(12.5, 4.8)
        ax = self.figure.add_subplot(111)
        ax.plot(water_df["time"], water_df["water_level_m"], linewidth=1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Water depth (m)" if self.is_depth_only_mode() else "Water level (m)")
        ax.set_title("Water depth time series" if self.is_depth_only_mode() else "Water level time series")
        ax.grid(True, alpha=0.3)
        self.figure.autofmt_xdate()
        self.figure.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(str(save_path), dpi=300, bbox_inches="tight")

        self.canvas.draw()

        lines = []
        lines.append("水位过程曲线已生成")
        lines.append("-" * 80)
        lines.append(f"有效水位行数：{len(water_df)}")
        lines.append(f"开始时间：{water_df['time'].iloc[0]}")
        lines.append(f"结束时间：{water_df['time'].iloc[-1]}")
        lines.append(f"水位范围：{water_df['water_level_m'].min():.4f} - {water_df['water_level_m'].max():.4f} m")
        self.preview_box.setPlainText("\n".join(lines))

    def save_results(self):
        try:
            if self.effective_df is None:
                self.compute_hydraulic_parameters()
            if self.effective_df is None:
                return

            output_text = self.output_dir.text().strip()
            if output_text:
                output_dir = Path(output_text)
            else:
                if self.water_file.text().strip():
                    output_dir = Path(self.water_file.text().strip()).parent / "hydraulic_outputs"
                elif self.cross_file.text().strip():
                    output_dir = Path(self.cross_file.text().strip()).parent / "hydraulic_outputs"
                else:
                    output_dir = Path.cwd() / "hydraulic_outputs"
                self.output_dir.setText(str(output_dir))

            output_dir.mkdir(parents=True, exist_ok=True)
            effective_path = output_dir / "effective_depth_width.csv"
            water_plot_path = output_dir / "water_level_timeseries.png"

            save_cols = ["time", "water_level_m", "H_eff_m", "W_eff_m", "D_effective_min_mm", "D_effective_max_mm"]
            self.effective_df[save_cols].to_csv(effective_path, index=False, encoding="utf-8-sig")
            self.plot_water_level(save_path=water_plot_path)

            QMessageBox.information(self, "保存成功", "已保存：\n" f"{effective_path}\n" f"{water_plot_path}")
            self.hydraulic_saved.emit(str(effective_path))
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存结果时出错：\n{e}")

    def clear_page(self):
        self.cross_file.clear()
        self.water_file.clear()
        self.output_dir.clear()
        if hasattr(self, "hydraulic_mode_combo"):
            self.hydraulic_mode_combo.setCurrentIndex(0)
        self.slope_edit.setText("0.01")
        self.default_width_edit.setText("10")
        self.rho_w_edit.setText("1000")
        self.theta_c_edit.setText("0.045")
        self.preview_box.clear()
        self.figure.clear()
        self.canvas.draw()
        self.cross_df_original = None
        self.water_df_original = None
        self.cross_df_standard = None
        self.water_df_standard = None
        self.effective_df = None
        self.x_nodes = None
        self.z_nodes = None
        self.node_widths = None





class BedloadInversionPage(QWidget):
    """反演计算页面：调用模块 3、4、5 输出，反演推移质通量。"""

    def __init__(self):
        super().__init__()
        self.project_page = None
        self.time_frequency_page = None
        self.sediment_page = None
        self.hydraulic_page = None

        self.energy_df = None
        self.hydraulic_df = None
        self.grain_df = None
        self.unit_energy_df = None
        self.unit_energy_cache_key = None
        self.unit_energy_stats = {}
        self.result_df = None

        self.build_ui()

    def build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("反演计算")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        subtitle = QLabel(
            "自动调用时频分析、水力参数和泥沙级配结果；计算单位通量理论能量，并反演单位宽和断面推移质通量。"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        file_group = QGroupBox("文件输入")
        file_layout = QFormLayout(file_group)
        file_layout.setSpacing(10)

        self.energy_input_mode_combo = QComboBox()
        self.energy_input_mode_combo.addItems([
            "单个地震能量 CSV 文件",
            "地震能量 CSV 文件夹",
        ])
        self.energy_input_mode_combo.setCurrentIndex(0)

        self.energy_file = QLineEdit()
        self.hydraulic_file = QLineEdit()
        self.grain_file = QLineEdit()
        self.output_dir = QLineEdit()

        file_layout.addRow("能量输入方式：", self.energy_input_mode_combo)
        file_layout.addRow("地震能量输入：", self.make_path_row(self.energy_file, self.choose_energy_file))
        file_layout.addRow("水力参数 CSV：", self.make_path_row(self.hydraulic_file, self.choose_hydraulic_file))
        file_layout.addRow("泥沙级配 CSV：", self.make_path_row(self.grain_file, self.choose_grain_file))
        file_layout.addRow("输出文件夹：", self.make_path_row(self.output_dir, self.choose_output_dir, is_dir=True))
        layout.addWidget(file_group)

        param_group = QGroupBox("反演参数")
        param_layout = QFormLayout(param_group)
        param_layout.setSpacing(10)
        self.r0_edit = QLineEdit("17")
        self.r0_edit.setPlaceholderText("台站到河道源区距离 r₀，m")
        param_layout.addRow("台站-河道距离 r<sub>0</sub>，m：", self.r0_edit)
        layout.addWidget(param_group)

        outlier_group = QGroupBox("结果去异常值")
        outlier_layout = QFormLayout(outlier_group)
        outlier_layout.setSpacing(8)
        self.outlier_enable = QCheckBox("启用单位宽通量 qᵦ 异常值剔除与插值补点")
        self.outlier_enable.setChecked(True)
        self.outlier_window_min_edit = QLineEdit("60")
        self.outlier_window_min_edit.setPlaceholderText("滚动窗口长度，分钟，例如 60")
        self.outlier_factor_edit = QLineEdit("5")
        self.outlier_factor_edit.setPlaceholderText("超过局部中值的几倍判为异常，例如 5")
        outlier_layout.addRow("是否启用：", self.outlier_enable)
        outlier_layout.addRow("滚动中值窗口，min：", self.outlier_window_min_edit)
        outlier_layout.addRow("异常阈值倍数：", self.outlier_factor_edit)
        layout.addWidget(outlier_group)

        seismic_group = QGroupBox("地震传播参数（高级，可按场地修改）")
        seismic_layout = QFormLayout(seismic_group)
        seismic_layout.setSpacing(8)

        self.v0_edit = QLineEdit("2206")
        self.z0_edit = QLineEdit("1000")
        self.f0_edit = QLineEdit("1")
        self.a_edit = QLineEdit("0.272")
        self.Q0_edit = QLineEdit("20")
        self.eta_edit = QLineEdit("0")
        self.phi_edit = QLineEdit("0")
        self.eb_edit = QLineEdit("0.5")
        self.fx_edit = QLineEdit("0.146")
        self.fy_edit = QLineEdit("0.146")
        self.fz_edit = QLineEdit("0.539")
        self.Nzz_edit = QLineEdit("0.352")

        seismic_layout.addRow("v<sub>0</sub>（参考相速度，m/s）：", self.v0_edit)
        seismic_layout.addRow("z<sub>0</sub>（参考深度，m）：", self.z0_edit)
        seismic_layout.addRow("f<sub>0</sub>（参考频率，Hz）：", self.f0_edit)
        seismic_layout.addRow("a（频散指数）：", self.a_edit)
        seismic_layout.addRow("Q<sub>0</sub>（品质因子）：", self.Q0_edit)
        seismic_layout.addRow("η（Q 的频率指数）：", self.eta_edit)
        seismic_layout.addRow("φ（源–台站方位角）：", self.phi_edit)
        seismic_layout.addRow("e<sub>b</sub>（反弹系数）：", self.eb_edit)
        seismic_layout.addRow("f<sub>x</sub>（x 向冲量系数）：", self.fx_edit)
        seismic_layout.addRow("f<sub>y</sub>（y 向冲量系数）：", self.fy_edit)
        seismic_layout.addRow("f<sub>z</sub>（z 向冲量系数）：", self.fz_edit)
        seismic_layout.addRow("N<sub>zz</sub>（垂直辐射项）：", self.Nzz_edit)
        layout.addWidget(seismic_group)

        button_row = QHBoxLayout()
        self.auto_find_button = QPushButton("自动查找文件")
        self.import_button = QPushButton("导入反演数据")
        self.unit_energy_button = QPushButton("计算单位能量表")
        self.compute_button = QPushButton("计算推移质通量")
        self.save_button = QPushButton("保存结果")
        self.clear_button = QPushButton("清空")

        self.auto_find_button.clicked.connect(lambda checked=False: self.run_with_error_dialog("自动查找文件", lambda: self.auto_find_files(show_message=True)))
        self.import_button.clicked.connect(lambda checked=False: self.run_with_error_dialog("导入反演数据", self.import_inversion_data))
        self.unit_energy_button.clicked.connect(lambda checked=False: self.run_with_error_dialog("计算单位能量表", lambda: self.compute_unit_energy_lookup(show_message=True, force=True)))
        self.compute_button.clicked.connect(lambda checked=False: self.run_with_error_dialog("计算推移质通量", self.compute_bedload_flux))
        self.save_button.clicked.connect(lambda checked=False: self.run_with_error_dialog("保存反演结果", self.save_results))
        self.clear_button.clicked.connect(lambda checked=False: self.run_with_error_dialog("清空反演页面", self.clear_page))

        for btn in [self.auto_find_button, self.import_button, self.unit_energy_button, self.compute_button, self.save_button, self.clear_button]:
            button_row.addWidget(btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(230)
        layout.addWidget(self.preview_box)

        self.figure = Figure(figsize=(8.8, 4.8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(430)
        layout.addWidget(self.canvas)

        layout.addStretch()
        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

    def run_with_error_dialog(self, action_name, func):
        """执行按钮动作；如果报错，直接在软件界面显示原因和完整 traceback。"""
        try:
            return func()
        except Exception as e:
            user_message = self.explain_exception(e)
            detail = traceback.format_exc()
            text = []
            text.append(f"{action_name}失败")
            text.append("-" * 80)
            text.append(user_message)
            text.append("")
            text.append("完整错误信息：")
            text.append(detail)
            if hasattr(self, "preview_box"):
                self.preview_box.setPlainText("\n".join(text))
            QMessageBox.critical(
                self,
                f"{action_name}失败",
                user_message + "\n\n详细错误已经显示在下方日志框中。"
            )
            return None

    def explain_exception(self, error):
        """把常见 Python/依赖错误转换成用户可理解的说明。"""
        msg = str(error)
        etype = type(error).__name__

        if isinstance(error, AttributeError) and "trapz" in msg and "numpy" in msg:
            return (
                "错误原因：当前 Python 环境中的 NumPy 版本不再支持 np.trapz。\n"
                "修复方式：本版本已改为兼容 NumPy 1.x/2.x 的 safe_trapz/np.trapezoid 写法。\n"
                "请使用当前修正版 main.py 重新运行。"
            )

        if isinstance(error, FileNotFoundError):
            return f"错误原因：找不到文件。\n{msg}\n请检查文件路径是否存在，或重新选择文件。"

        if isinstance(error, KeyError):
            return (
                f"错误原因：输入表格缺少必要字段：{msg}\n"
                "请检查 CSV 表头是否包含程序需要的 time、energy_linear、H_eff_m、W_eff_m、D_effective_min_mm、D_effective_max_mm、D_mm、fraction 等字段。"
            )

        if isinstance(error, ValueError):
            return (
                f"错误原因：参数或数据格式不正确。\n{msg}\n"
                "请检查输入文件是否为空、数字列是否能转换为数值，以及参数输入框是否为有效数字。"
            )

        if "No module named" in msg:
            return (
                f"错误原因：当前 Python 环境缺少必要依赖库。\n{msg}\n"
                "请在 bedload_app 环境中安装对应库后再运行。"
            )

        return f"错误类型：{etype}\n错误原因：{msg}"

    def set_context_pages(self, project_page=None, time_frequency_page=None, sediment_page=None, hydraulic_page=None):
        self.project_page = project_page
        self.time_frequency_page = time_frequency_page
        self.sediment_page = sediment_page
        self.hydraulic_page = hydraulic_page
        self.auto_find_files(show_message=False)

    def make_path_row(self, line_edit, callback, is_dir=False):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(callback)
        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        return row

    def is_energy_folder_mode(self):
        return hasattr(self, "energy_input_mode_combo") and self.energy_input_mode_combo.currentIndex() == 1

    def choose_energy_file(self):
        if self.is_energy_folder_mode():
            path = QFileDialog.getExistingDirectory(self, "选择地震能量 CSV 文件夹")
            if path:
                self.energy_file.setText(path)
                if not self.output_dir.text().strip():
                    self.output_dir.setText(str(Path(path).parent / "inversion_outputs"))
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择地震能量 CSV", "", "CSV Files (*.csv);;All Files (*)")
            if path:
                self.energy_file.setText(path)
                if not self.output_dir.text().strip():
                    self.output_dir.setText(str(Path(path).parent.parent / "inversion_outputs"))

    def choose_hydraulic_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择水力参数 CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.hydraulic_file.setText(path)
            if not self.output_dir.text().strip():
                self.output_dir.setText(str(Path(path).parent / "inversion_outputs"))

    def choose_grain_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择泥沙级配 CSV", "", "CSV Files (*.csv *.txt *.xlsx *.xls);;All Files (*)")
        if path:
            self.grain_file.setText(path)

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir.setText(path)

    def set_energy_file(self, path):
        if path and Path(path).exists():
            p = Path(path)
            if hasattr(self, "energy_input_mode_combo"):
                self.energy_input_mode_combo.setCurrentIndex(1 if p.is_dir() else 0)
            self.energy_file.setText(str(p))
            if not self.output_dir.text().strip():
                if p.is_dir():
                    self.output_dir.setText(str(p.parent / "inversion_outputs"))
                else:
                    self.output_dir.setText(str(p.parent.parent / "inversion_outputs"))

    def set_hydraulic_file(self, path):
        if path and Path(path).exists():
            self.hydraulic_file.setText(str(path))
            if not self.output_dir.text().strip():
                self.output_dir.setText(str(Path(path).parent / "inversion_outputs"))

    def set_grain_file(self, path):
        if path and Path(path).exists():
            p = Path(path)
            if p.suffix.lower() == ".csv":
                self.grain_file.setText(str(path))

    def get_project_search_dirs(self):
        dirs = []
        for page_attr in ["time_frequency_page", "sediment_page", "hydraulic_page", "project_page"]:
            page = getattr(self, page_attr, None)
            if page is None:
                continue
            for attr in ["output_dir", "project_dir", "sac_dir", "water_file", "grain_file"]:
                widget = getattr(page, attr, None)
                if widget is None or not hasattr(widget, "text"):
                    continue
                value = widget.text().strip()
                if not value:
                    continue
                p = Path(value)
                if p.is_file():
                    p = p.parent
                if p.exists():
                    dirs.append(p)
        unique = []
        seen = set()
        for d in dirs:
            try:
                key = str(d.resolve())
            except Exception:
                key = str(d)
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    def latest_file(self, paths):
        paths = [Path(p) for p in paths if Path(p).exists() and Path(p).is_file()]
        if not paths:
            return None
        return max(paths, key=lambda p: p.stat().st_mtime)

    def search_file(self, directories, names=None, patterns=None):
        candidates = []
        names = names or []
        patterns = patterns or []
        for d in directories:
            if not d.exists():
                continue
            for name in names:
                candidates.extend(d.rglob(name))
            for pattern in patterns:
                candidates.extend(d.rglob(pattern))
        return self.latest_file(candidates)

    def auto_find_files(self, show_message=True):
        dirs = self.get_project_search_dirs()

        if not self.energy_file.text().strip():
            energy_candidates = []
            energy_dir_candidate = None
            if self.time_frequency_page is not None and hasattr(self.time_frequency_page, "output_dir"):
                out = self.time_frequency_page.output_dir.text().strip()
                if out:
                    energy_dir = Path(out) / "energy_csv"
                    if energy_dir.exists():
                        energy_dir_candidate = energy_dir
                        energy_candidates.extend(energy_dir.glob("*.csv"))
            if energy_dir_candidate is not None and len(energy_candidates) > 1:
                if hasattr(self, "energy_input_mode_combo"):
                    self.energy_input_mode_combo.setCurrentIndex(1)
                self.energy_file.setText(str(energy_dir_candidate))
            else:
                energy_path = self.latest_file(energy_candidates)
                if energy_path is None:
                    energy_path = self.search_file(dirs, patterns=["*_energy_*Hz.csv", "*energy*.csv"])
                if energy_path is not None:
                    if hasattr(self, "energy_input_mode_combo"):
                        self.energy_input_mode_combo.setCurrentIndex(0)
                    self.energy_file.setText(str(energy_path))

        if not self.hydraulic_file.text().strip():
            hydraulic_path = None
            if self.hydraulic_page is not None and hasattr(self.hydraulic_page, "output_dir"):
                out = self.hydraulic_page.output_dir.text().strip()
                if out:
                    p = Path(out) / "effective_depth_width.csv"
                    if p.exists():
                        hydraulic_path = p
            if hydraulic_path is None:
                hydraulic_path = self.search_file(dirs, names=["effective_depth_width.csv"])
            if hydraulic_path is not None:
                self.hydraulic_file.setText(str(hydraulic_path))

        if not self.grain_file.text().strip():
            grain_path = None
            if self.sediment_page is not None and hasattr(self.sediment_page, "output_dir"):
                out = self.sediment_page.output_dir.text().strip()
                if out:
                    for name in ["standard_grain_distribution.csv", "generated_grain_distribution.csv"]:
                        p = Path(out) / name
                        if p.exists():
                            grain_path = p
                            break
            if grain_path is None:
                grain_path = self.search_file(
                    dirs,
                    names=["standard_grain_distribution.csv", "generated_grain_distribution.csv"],
                    patterns=["*grain*distribution*.csv"],
                )
            if grain_path is not None:
                self.grain_file.setText(str(grain_path))

        if not self.output_dir.text().strip():
            output_dir = None
            if self.project_page is not None and hasattr(self.project_page, "output_dir"):
                value = self.project_page.output_dir.text().strip()
                if value:
                    output_dir = Path(value) / "inversion_outputs"
            if output_dir is None and self.hydraulic_file.text().strip():
                output_dir = Path(self.hydraulic_file.text().strip()).parent / "inversion_outputs"
            if output_dir is None and self.energy_file.text().strip():
                energy_p = Path(self.energy_file.text().strip())
                if energy_p.is_dir():
                    output_dir = energy_p.parent / "inversion_outputs"
                else:
                    output_dir = energy_p.parent.parent / "inversion_outputs"
            if output_dir is None:
                output_dir = Path.cwd() / "inversion_outputs"
            self.output_dir.setText(str(output_dir))

        if show_message:
            lines = ["自动查找完成", "-" * 80]
            lines.append(f"地震能量输入：{self.energy_file.text().strip() or '未找到'}")
            lines.append(f"水力参数 CSV：{self.hydraulic_file.text().strip() or '未找到'}")
            lines.append(f"泥沙级配 CSV：{self.grain_file.text().strip() or '未找到'}")
            lines.append(f"输出文件夹：{self.output_dir.text().strip() or '未设置'}")
            self.preview_box.setPlainText("\n".join(lines))

    def read_table_data(self, path):
        import pandas as pd
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                return pd.read_csv(path, encoding="utf-8-sig")
            except Exception:
                return pd.read_csv(path)
        if suffix in [".txt", ".dat"]:
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
            except Exception:
                return pd.read_csv(path, sep=None, engine="python")
        if suffix == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")
        if suffix == ".xls":
            return pd.read_excel(path, engine="xlrd")
        return pd.read_csv(path, sep=None, engine="python")

    def parse_time_series(self, series):
        import pandas as pd
        dt = pd.to_datetime(series, errors="coerce")
        if dt.notna().sum() == 0:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() > 0:
                dt = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
        return dt

    def to_numeric_series(self, series):
        import pandas as pd
        s = series.astype(str).str.strip().str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    def find_column(self, df, candidates):
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        for candidate in candidates:
            key = candidate.lower()
            if key in lower_map:
                return lower_map[key]
        for c in df.columns:
            c_lower = str(c).strip().lower()
            for candidate in candidates:
                if candidate.lower() in c_lower:
                    return c
        return None

    def collect_energy_csv_files(self, folder):
        folder = Path(folder)
        if not folder.exists() or not folder.is_dir():
            return []
        candidates = list(folder.glob("*_energy_*Hz.csv"))
        if not candidates:
            candidates = [p for p in folder.glob("*.csv") if "energy" in p.name.lower()]
        if not candidates:
            candidates = list(folder.glob("*.csv"))
        return sorted(candidates, key=lambda p: p.name.lower())

    def read_energy_input(self, path_text):
        import pandas as pd
        p = Path(path_text)
        if p.is_file():
            return self.standardize_energy_df(self.read_table_data(p))
        if p.is_dir():
            csv_files = self.collect_energy_csv_files(p)
            if not csv_files:
                raise ValueError("地震能量 CSV 文件夹中没有可用 CSV 文件。")
            frames = []
            bad_files = []
            for csv_path in csv_files:
                try:
                    frames.append(self.standardize_energy_df(self.read_table_data(csv_path)))
                except Exception as e:
                    bad_files.append(f"{csv_path.name}: {e}")
            if not frames:
                raise ValueError("地震能量 CSV 文件夹中的文件均无法读取。" + ("\n" + "\n".join(bad_files[:5]) if bad_files else ""))
            out = pd.concat(frames, ignore_index=True)
            out = out.sort_values("time").drop_duplicates("time", keep="first").reset_index(drop=True)
            return out
        raise FileNotFoundError(f"地震能量输入不存在：{path_text}")

    def standardize_energy_df(self, df):
        time_col = self.find_column(df, ["time", "datetime", "date", "timestamp", "时间", "日期"])
        energy_col = self.find_column(df, ["energy_linear", "linear_energy", "energy", "psd_energy", "线性能量"])
        if time_col is None and len(df.columns) >= 1:
            time_col = df.columns[0]
        if energy_col is None and len(df.columns) >= 2:
            energy_col = df.columns[1]
        if time_col is None or energy_col is None:
            raise ValueError("地震能量 CSV 至少需要 time 和 energy_linear 两列。")
        out = df[[time_col, energy_col]].copy()
        out.columns = ["time", "energy_linear"]
        out["time"] = self.parse_time_series(out["time"])
        out["energy_linear"] = self.to_numeric_series(out["energy_linear"])
        out = out.dropna(subset=["time", "energy_linear"])
        out = out[np.isfinite(out["energy_linear"]) & (out["energy_linear"] >= 0)]
        out = out.sort_values("time").drop_duplicates("time")
        if out.empty:
            raise ValueError("地震能量 CSV 没有有效数据。")
        return out.reset_index(drop=True)

    def get_slope_from_context(self):
        """从项目管理或水力模块读取固定坡降 S。"""
        try:
            if self.project_page is not None and hasattr(self.project_page, "slope_edit"):
                value = float(self.project_page.slope_edit.text())
                if value > 0:
                    return value
        except Exception:
            pass
        try:
            if self.hydraulic_page is not None and hasattr(self.hydraulic_page, "slope_edit"):
                value = float(self.hydraulic_page.slope_edit.text())
                if value > 0:
                    return value
        except Exception:
            pass
        return 0.01

    def standardize_hydraulic_df(self, df):
        time_col = self.find_column(df, ["time", "datetime", "date", "timestamp", "时间", "日期"])
        wl_col = self.find_column(df, ["water_level_m", "water_level", "stage", "level", "水位"])
        h_col = self.find_column(df, ["H_eff_m", "H_eff", "effective_depth", "有效水深"])
        w_col = self.find_column(df, ["W_eff_m", "W_eff", "effective_width", "有效宽度"])
        dmin_col = self.find_column(df, ["D_effective_min_mm", "D_min_mm", "effective_min", "有效最小粒径"])
        dmax_col = self.find_column(df, ["D_effective_max_mm", "D_max_mm", "effective_max", "有效最大粒径"])
        slope_col = self.find_column(df, ["slope", "S", "坡降"])

        required = [time_col, wl_col, h_col, w_col, dmin_col, dmax_col]
        if any(c is None for c in required):
            raise ValueError("水力参数 CSV 需要包含 time, water_level_m, H_eff_m, W_eff_m, D_effective_min_mm, D_effective_max_mm。坡降 S 从项目管理读取。")

        if slope_col is not None:
            out = df[[time_col, wl_col, h_col, w_col, dmin_col, dmax_col, slope_col]].copy()
            out.columns = ["time", "water_level_m", "H_eff_m", "W_eff_m", "D_effective_min_mm", "D_effective_max_mm", "slope"]
        else:
            out = df[[time_col, wl_col, h_col, w_col, dmin_col, dmax_col]].copy()
            out.columns = ["time", "water_level_m", "H_eff_m", "W_eff_m", "D_effective_min_mm", "D_effective_max_mm"]
            out["slope"] = self.get_slope_from_context()

        out["time"] = self.parse_time_series(out["time"])
        for col in ["water_level_m", "H_eff_m", "W_eff_m", "D_effective_min_mm", "D_effective_max_mm", "slope"]:
            out[col] = self.to_numeric_series(out[col])
        out = out.dropna(subset=["time"])
        out = out.sort_values("time").drop_duplicates("time")
        if out.empty:
            raise ValueError("水力参数 CSV 没有有效数据。")
        return out.reset_index(drop=True)

    def standardize_grain_df(self, df):
        d_col = self.find_column(df, ["D_mm", "d_mm", "diameter_mm", "grain_size_mm", "grain size", "粒径", "粒径mm"])
        f_col = self.find_column(df, ["fraction", "fraction_percent", "percentage", "percent", "weight_percent", "cum_percent", "cum_fraction", "含量", "百分比", "累计"])
        if d_col is None and len(df.columns) >= 1:
            d_col = df.columns[0]
        if f_col is None and len(df.columns) >= 2:
            f_col = df.columns[1]
        if d_col is None or f_col is None:
            raise ValueError("泥沙级配 CSV 至少需要 D_mm 和 fraction 两列。")

        out = df[[d_col, f_col]].copy()
        out.columns = ["D_mm", "fraction_raw"]
        out["D_mm"] = self.to_numeric_series(out["D_mm"])
        out["fraction_raw"] = self.to_numeric_series(out["fraction_raw"])
        out = out.dropna(subset=["D_mm", "fraction_raw"])
        out = out[(out["D_mm"] > 0) & np.isfinite(out["D_mm"]) & np.isfinite(out["fraction_raw"])]
        out = out.sort_values("D_mm")
        if out.empty:
            raise ValueError("泥沙级配文件没有有效粒径数据。")

        f_name = str(f_col).lower()
        values = out["fraction_raw"].to_numpy(dtype=float)
        if "cum" in f_name or "累计" in f_name:
            cum = values.copy()
            if np.nanmax(cum) > 1.5:
                cum = cum / 100.0
            cum = np.clip(cum, 0.0, 1.0)
            frac = np.diff(np.r_[0.0, cum])
            frac = np.where(frac > 0, frac, 0.0)
        else:
            frac = values.copy()
            if np.nanmax(frac) > 1.5 or "percent" in f_name or "百分" in f_name:
                frac = frac / 100.0
            frac = np.where(frac > 0, frac, 0.0)

        total = np.nansum(frac)
        if total <= 0:
            raise ValueError("泥沙级配 fraction 总和必须大于 0。")
        out["fraction"] = frac / total
        return out[["D_mm", "fraction"]].reset_index(drop=True)

    def import_inversion_data(self):
        self.auto_find_files(show_message=False)
        energy_path = self.energy_file.text().strip()
        hydraulic_path = self.hydraulic_file.text().strip()
        grain_path = self.grain_file.text().strip()

        for label, path in [("地震能量输入", energy_path), ("水力参数 CSV", hydraulic_path), ("泥沙级配 CSV", grain_path)]:
            if not path:
                QMessageBox.warning(self, "提示", f"请先选择或自动查找{label}。")
                return
            if not Path(path).exists():
                QMessageBox.warning(self, "提示", f"{label}不存在：\n{path}")
                return

        try:
            self.energy_df = self.read_energy_input(energy_path)
            self.hydraulic_df = self.standardize_hydraulic_df(self.read_table_data(hydraulic_path))
            self.grain_df = self.standardize_grain_df(self.read_table_data(grain_path))
            self.unit_energy_df = None
            self.unit_energy_cache_key = None
            self.unit_energy_stats = {}
            self.result_df = None
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入反演数据时出错：\n{e}")
            return

        lines = []
        lines.append("反演数据导入完成")
        lines.append("-" * 80)
        lines.append(f"地震能量：{len(self.energy_df)} 行，{self.energy_df['time'].min()} 至 {self.energy_df['time'].max()}")
        lines.append(f"水力参数：{len(self.hydraulic_df)} 行，{self.hydraulic_df['time'].min()} 至 {self.hydraulic_df['time'].max()}")
        lines.append(f"泥沙级配：{len(self.grain_df)} 个粒径点，D = {self.grain_df['D_mm'].min():.4g} - {self.grain_df['D_mm'].max():.4g} mm")
        lines.append(f"地震能量范围：{self.energy_df['energy_linear'].min():.4e} - {self.energy_df['energy_linear'].max():.4e}")
        self.preview_box.setPlainText("\n".join(lines))

    def get_float_from_edit(self, edit, default, name):
        text = edit.text().strip()
        if not text:
            return default
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{name} 必须为数字。")
        return value

    def get_rho_s(self):
        if self.sediment_page is not None and hasattr(self.sediment_page, "rho_s_edit"):
            try:
                value = float(self.sediment_page.rho_s_edit.text())
                if value > 0:
                    return value
            except Exception:
                pass
        return 2700.0

    def get_rho_w(self):
        if self.hydraulic_page is not None and hasattr(self.hydraulic_page, "rho_w_edit"):
            try:
                value = float(self.hydraulic_page.rho_w_edit.text())
                if value > 0:
                    return value
            except Exception:
                pass
        return 1000.0

    def get_tau_c50(self):
        if self.hydraulic_page is not None and hasattr(self.hydraulic_page, "theta_c_edit"):
            try:
                value = float(self.hydraulic_page.theta_c_edit.text())
                if value > 0:
                    return value
            except Exception:
                pass
        return 0.045

    def get_frequency_array(self):
        fmin, fmax, df = 30.0, 80.0, 1.0
        if self.time_frequency_page is not None:
            try:
                fmin = float(self.time_frequency_page.energy_fmin_edit.text())
                fmax = float(self.time_frequency_page.energy_fmax_edit.text())
                df = float(self.time_frequency_page.target_df_edit.text())
            except Exception:
                fmin, fmax, df = 30.0, 80.0, 1.0
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            fmin, fmax = 30.0, 80.0
        if not np.isfinite(df) or df <= 0:
            df = 1.0
        n = int(round((fmax - fmin) / df)) + 1
        n = max(n, 2)
        return np.linspace(fmin, fmax, n)

    def get_seismic_params(self):
        return BedloadSeismicParams(
            v0=self.get_float_from_edit(self.v0_edit, 2206.0, "v0"),
            z0=self.get_float_from_edit(self.z0_edit, 1000.0, "z0"),
            f0=self.get_float_from_edit(self.f0_edit, 1.0, "f0"),
            a=self.get_float_from_edit(self.a_edit, 0.272, "a"),
            Q0=self.get_float_from_edit(self.Q0_edit, 20.0, "Q0"),
            eta=self.get_float_from_edit(self.eta_edit, 0.0, "eta"),
            phi=self.get_float_from_edit(self.phi_edit, 0.0, "phi"),
            eb=self.get_float_from_edit(self.eb_edit, 0.5, "eb"),
            fx=self.get_float_from_edit(self.fx_edit, 0.146, "fx"),
            fy=self.get_float_from_edit(self.fy_edit, 0.146, "fy"),
            fz=self.get_float_from_edit(self.fz_edit, 0.539, "fz"),
            Nzz=self.get_float_from_edit(self.Nzz_edit, 0.352, "Nzz"),
        )

    def interpolate_hydraulic_to_energy_time(self):
        import pandas as pd
        e = self.energy_df.copy().sort_values("time")
        h = self.hydraulic_df.copy().sort_values("time")
        e_time = e["time"].astype("int64").to_numpy(dtype=float) / 1e9
        h_time = h["time"].astype("int64").to_numpy(dtype=float) / 1e9

        interp = e[["time", "energy_linear"]].copy()
        for col in ["water_level_m", "H_eff_m", "W_eff_m", "D_effective_min_mm", "D_effective_max_mm", "slope"]:
            y = h[col].to_numpy(dtype=float)
            valid = np.isfinite(h_time) & np.isfinite(y)
            if valid.sum() < 2:
                if valid.sum() == 1:
                    interp[col] = y[valid][0]
                else:
                    interp[col] = np.nan
            else:
                interp[col] = np.interp(e_time, h_time[valid], y[valid], left=np.nan, right=np.nan)
        return interp

    def weighted_median_D50_m(self, D_m, weights):
        D_m = np.asarray(D_m, dtype=float)
        weights = np.asarray(weights, dtype=float)
        order = np.argsort(D_m)
        D_sorted = D_m[order]
        w_sorted = weights[order]
        total = np.sum(w_sorted)
        if total <= 0:
            return np.nanmedian(D_sorted)
        cum = np.cumsum(w_sorted) / total
        idx = np.searchsorted(cum, 0.5)
        idx = min(max(idx, 0), len(D_sorted) - 1)
        return D_sorted[idx]

    def make_unit_energy_cache_key(self, r0, rho_s, rho_w, tau_c50, f, seismic_params):
        """生成单位能量表的内存缓存键。参数或输入文件改变后自动重算。"""
        def file_stamp(path_text):
            path_text = str(path_text).strip()
            if not path_text:
                return ("", None, None)
            p = Path(path_text)
            if not p.exists():
                return (str(p), None, None)
            try:
                return (str(p.resolve()), p.stat().st_mtime, p.stat().st_size)
            except Exception:
                return (str(p), None, None)

        seismic_tuple = (
            float(seismic_params.v0), float(seismic_params.z0), float(seismic_params.f0),
            float(seismic_params.a), float(seismic_params.Q0), float(seismic_params.eta),
            float(seismic_params.phi), float(seismic_params.eb), float(seismic_params.fx),
            float(seismic_params.fy), float(seismic_params.fz), float(seismic_params.Nzz),
        )
        f_tuple = tuple(np.round(np.asarray(f, dtype=float), 6))
        return (
            file_stamp(self.energy_file.text()),
            file_stamp(self.hydraulic_file.text()),
            file_stamp(self.grain_file.text()),
            round(float(r0), 8), round(float(rho_s), 8), round(float(rho_w), 8),
            round(float(tau_c50), 8), f_tuple, seismic_tuple,
        )

    def prepare_inversion_parameters(self):
        """读取反演所需参数；优先调用前面模块输入，找不到才使用默认值。"""
        r0 = self.get_float_from_edit(self.r0_edit, 17.0, "r0")
        if r0 <= 0:
            raise ValueError("r₀ 必须大于 0。")
        rho_s = self.get_rho_s()
        rho_w = self.get_rho_w()
        tau_c50 = self.get_tau_c50()
        f = self.get_frequency_array()
        seismic_params = self.get_seismic_params()
        cache_key = self.make_unit_energy_cache_key(r0, rho_s, rho_w, tau_c50, f, seismic_params)
        return r0, rho_s, rho_w, tau_c50, f, seismic_params, cache_key

    def compute_unit_energy_lookup(self, show_message=True, force=False):
        """
        在内存中计算单位通量理论能量表，不保存为 CSV。

        unit_energy_1m_kg_m_s 表示：
        1 m 有效宽度内，单位宽质量通量 1 kg m^-1 s^-1 在频带内产生的理论线性能量中值。
        后续反演使用：
            q_b_kg_m_s = energy_linear / (unit_energy_1m_kg_m_s * W_eff_m)
            Q_b_kg_s   = q_b_kg_m_s * W_eff_m
        """
        if self.energy_df is None or self.hydraulic_df is None or self.grain_df is None:
            self.import_inversion_data()
        if self.energy_df is None or self.hydraulic_df is None or self.grain_df is None:
            return None

        try:
            r0, rho_s, rho_w, tau_c50, f, seismic_params, cache_key = self.prepare_inversion_parameters()
        except Exception as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return None

        if (not force) and self.unit_energy_df is not None and self.unit_energy_cache_key == cache_key:
            if show_message:
                stats = self.unit_energy_stats or {}
                lines = []
                lines.append("单位能量表已存在于内存中，本次直接复用")
                lines.append("-" * 80)
                lines.append(f"表格行数：{len(self.unit_energy_df)}")
                lines.append(f"有效粒径为空：{stats.get('zero_empty_count', 0)} 个时刻")
                lines.append(f"水力参数无效：{stats.get('zero_invalid_count', 0)} 个时刻")
                lines.append("说明：单位能量表仅保存在内存中，不输出 unit_bedload_energy_lookup.csv。")
                self.preview_box.setPlainText("\n".join(lines))
            return self.unit_energy_df

        sediment_params = BedloadSedimentParams(rho_s=rho_s, rho_f=rho_w)
        model = SaltationBedloadModel(sediment_params=sediment_params, seismic_params=seismic_params)

        data = self.interpolate_hydraulic_to_energy_time()
        grain_D_mm_all = self.grain_df["D_mm"].to_numpy(dtype=float)
        grain_fraction_all = self.grain_df["fraction"].to_numpy(dtype=float)
        grain_D_m_all = grain_D_mm_all / 1000.0
        D50_all_m = self.weighted_median_D50_m(grain_D_m_all, grain_fraction_all)

        qb_unit_vol = 1.0 / rho_s
        unit_energy_values = []
        effective_fraction_values = []
        zero_empty_count = 0
        zero_invalid_count = 0
        positive_unit_count = 0

        for idx, row in data.iterrows():
            H = float(row["H_eff_m"]) if np.isfinite(row["H_eff_m"]) else np.nan
            W = float(row["W_eff_m"]) if np.isfinite(row["W_eff_m"]) else np.nan
            Dmin = float(row["D_effective_min_mm"]) if np.isfinite(row["D_effective_min_mm"]) else np.nan
            Dmax = float(row["D_effective_max_mm"]) if np.isfinite(row["D_effective_max_mm"]) else np.nan
            slope = float(row["slope"]) if np.isfinite(row["slope"]) else np.nan

            if (
                not np.isfinite(H) or H <= 0
                or not np.isfinite(W) or W <= 0
                or not np.isfinite(Dmin) or not np.isfinite(Dmax) or Dmax <= Dmin
                or not np.isfinite(slope) or slope <= 0
            ):
                unit_energy_values.append(0.0)
                effective_fraction_values.append(0.0)
                zero_invalid_count += 1
                continue

            mask = (
                (grain_D_mm_all >= Dmin) & (grain_D_mm_all <= Dmax)
                & np.isfinite(grain_D_mm_all) & np.isfinite(grain_fraction_all)
            )
            if not np.any(mask):
                unit_energy_values.append(0.0)
                effective_fraction_values.append(0.0)
                zero_empty_count += 1
                continue

            D_sel_m = grain_D_m_all[mask]
            w_sel = grain_fraction_all[mask]
            valid = np.isfinite(D_sel_m) & (D_sel_m > 0) & np.isfinite(w_sel) & (w_sel > 0)
            D_sel_m = D_sel_m[valid]
            w_sel = w_sel[valid]
            if D_sel_m.size == 0 or np.sum(w_sel) <= 0:
                unit_energy_values.append(0.0)
                effective_fraction_values.append(0.0)
                zero_empty_count += 1
                continue

            effective_fraction = float(np.sum(w_sel))
            order = np.argsort(D_sel_m)
            D_sel_m = D_sel_m[order]
            w_sel = w_sel[order]
            w_sel = w_sel / np.sum(w_sel)
            D50_eff_m = self.weighted_median_D50_m(D_sel_m, w_sel)

            if D_sel_m.size >= 2:
                area = safe_trapz(w_sel, x=D_sel_m)
                pdf_density = w_sel / area if area > 0 else None
            else:
                pdf_density = None

            try:
                PSD_unit_1m = model.forward_psd(
                    f=f,
                    D=D_sel_m,
                    H=H,
                    W=1.0,
                    theta=slope,
                    r0=r0,
                    qb=qb_unit_vol,
                    D50=D50_eff_m if np.isfinite(D50_eff_m) else D50_all_m,
                    tau_c50=tau_c50,
                    pdf=pdf_density,
                )
                PSD_unit_1m = np.asarray(PSD_unit_1m, dtype=float)
                finite_positive = PSD_unit_1m[np.isfinite(PSD_unit_1m) & (PSD_unit_1m > 0)]
                E_unit_1m = float(np.median(finite_positive)) if finite_positive.size > 0 else 0.0
            except Exception:
                E_unit_1m = 0.0

            if np.isfinite(E_unit_1m) and E_unit_1m > 0:
                positive_unit_count += 1
            else:
                E_unit_1m = 0.0

            unit_energy_values.append(E_unit_1m)
            effective_fraction_values.append(effective_fraction)

            if idx % 20 == 0:
                if hasattr(self, "progress_bar") and len(data) > 0:
                    self.progress_bar.setValue(int((idx + 1) * 100 / len(data)))
                QApplication.processEvents()

        lookup = data[[
            "time", "water_level_m", "H_eff_m", "W_eff_m",
            "D_effective_min_mm", "D_effective_max_mm", "slope"
        ]].copy()
        lookup["effective_grain_fraction"] = effective_fraction_values
        lookup["unit_energy_1m_kg_m_s"] = unit_energy_values

        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(100)
        self.unit_energy_df = lookup
        self.unit_energy_cache_key = cache_key
        self.unit_energy_stats = {
            "zero_empty_count": zero_empty_count,
            "zero_invalid_count": zero_invalid_count,
            "positive_unit_count": positive_unit_count,
            "rho_s": rho_s,
            "rho_w": rho_w,
            "tau_c50": tau_c50,
            "r0": r0,
            "fmin": float(f[0]),
            "fmax": float(f[-1]),
            "nfreq": len(f),
        }

        if show_message:
            positive = lookup["unit_energy_1m_kg_m_s"].to_numpy(dtype=float)
            positive = positive[np.isfinite(positive) & (positive > 0)]
            lines = []
            lines.append("单位能量表已在内存中计算完成")
            lines.append("-" * 80)
            lines.append("说明：该表不保存为 CSV，仅供本次反演直接查表使用。")
            lines.append(f"计算时刻数：{len(lookup)}")
            lines.append(f"频率数组：{f[0]:.3g} - {f[-1]:.3g} Hz，共 {len(f)} 点")
            lines.append(f"rho_s = {rho_s:.3g} kg/m³；rho_w = {rho_w:.3g} kg/m³；tau_c50 = {tau_c50:.4g}")
            lines.append(f"r0 = {r0:.3g} m")
            lines.append(f"有效粒径为空：{zero_empty_count} 个时刻")
            lines.append(f"水力参数无效：{zero_invalid_count} 个时刻")
            lines.append(f"单位能量有效：{positive_unit_count} 个时刻")
            if positive.size > 0:
                lines.append(f"单位能量范围：{positive.min():.4e} - {positive.max():.4e}")
            lines.append("\n前 10 行内存单位能量表：")
            lines.append(lookup.head(10).to_string(index=False))
            self.preview_box.setPlainText("\n".join(lines))

        return lookup

    def get_outlier_filter_parameters(self):
        """读取反演结果去异常值参数。"""
        enabled = bool(self.outlier_enable.isChecked()) if hasattr(self, "outlier_enable") else True
        try:
            window_min = float(self.outlier_window_min_edit.text().strip())
        except Exception:
            window_min = 60.0
        try:
            factor = float(self.outlier_factor_edit.text().strip())
        except Exception:
            factor = 5.0

        if not np.isfinite(window_min) or window_min <= 0:
            window_min = 60.0
        if not np.isfinite(factor) or factor <= 1:
            factor = 5.0
        return enabled, window_min, factor

    def remove_flux_outliers_and_interpolate(self, time_values, qb_values, Qb_values, W_values):
        """
        用时间滚动中值识别单位宽推移质通量 qᵦ 的尖峰，并用时间插值补点。

        正确处理顺序：
        1. 先由观测能量直接反演得到 qᵦ_raw。
        2. 对 qᵦ_raw 做局部中值倍数去异常值。
        3. 对被剔除的 q_b 点按时间插值补齐。
        4. 再用 Qᵦ_clean = qᵦ_clean * W_eff 计算断面总通量。

        这样不会先处理 Qᵦ 再反推 qᵦ，避免有效宽度 W_eff 的变化影响异常值判断。
        """
        enabled, window_min, factor = self.get_outlier_filter_parameters()

        # 显式 copy=True，避免某些 NumPy / pandas 版本返回只读视图导致写入失败。
        qb_values = np.array(qb_values, dtype=float, copy=True)
        Qb_values = np.array(Qb_values, dtype=float, copy=True)
        W_values = np.array(W_values, dtype=float, copy=True)

        if (not enabled) or len(qb_values) < 3:
            W_valid = np.where(np.isfinite(W_values) & (W_values > 0), W_values, 0.0)
            Qb_direct = qb_values * W_valid
            Qb_direct[~np.isfinite(Qb_direct) | (Qb_direct < 0)] = 0.0
            qb_values[~np.isfinite(qb_values) | (qb_values < 0)] = 0.0
            return qb_values, Qb_direct, 0, enabled, window_min, factor

        df = pd.DataFrame({
            "time": pd.to_datetime(time_values, errors="coerce"),
            "qb": qb_values,
            "W": W_values,
        })
        df = df.dropna(subset=["time"]).copy()
        if df.empty or len(df) < 3:
            W_valid = np.where(np.isfinite(W_values) & (W_values > 0), W_values, 0.0)
            Qb_direct = qb_values * W_valid
            Qb_direct[~np.isfinite(Qb_direct) | (Qb_direct < 0)] = 0.0
            qb_values[~np.isfinite(qb_values) | (qb_values < 0)] = 0.0
            return qb_values, Qb_direct, 0, enabled, window_min, factor

        # 保留原始顺序，内部按时间排序做时间滚动窗口。
        df["_original_order"] = np.arange(len(df))
        df = df.sort_values("time")

        s = pd.Series(np.array(df["qb"].to_numpy(dtype=float), dtype=float, copy=True), index=df["time"])
        s = s.replace([np.inf, -np.inf], np.nan)
        s[s < 0] = 0.0

        window = f"{window_min:g}min"
        local_median = s.rolling(window=window, center=True, min_periods=3).median()
        local_median_fallback = s.rolling(window=window, center=True, min_periods=1).median()
        local_median = local_median.fillna(local_median_fallback)

        values = np.array(s.to_numpy(dtype=float), dtype=float, copy=True)
        med = np.array(local_median.to_numpy(dtype=float), dtype=float, copy=True)

        positive_values = values[np.isfinite(values) & (values > 0)]
        if positive_values.size > 0:
            median_floor = max(float(np.nanmedian(positive_values)) * 1e-6, 1e-30)
        else:
            median_floor = 1e-30

        threshold = factor * np.maximum(med, median_floor)
        outlier_mask = np.isfinite(values) & (values > 0) & np.isfinite(threshold) & (values > threshold)

        # 只剔除 q_b 尖峰，然后对 q_b 本身按时间插值补点。
        cleaned_qb = s.copy()
        cleaned_qb.iloc[np.where(outlier_mask)[0]] = np.nan
        cleaned_qb = cleaned_qb.interpolate(method="time", limit_direction="both")
        cleaned_qb = cleaned_qb.fillna(0.0)

        qb_clean_sorted = np.array(cleaned_qb.to_numpy(dtype=float), dtype=float, copy=True)
        qb_clean_sorted[~np.isfinite(qb_clean_sorted) | (qb_clean_sorted < 0)] = 0.0

        W_sorted = np.array(df["W"].to_numpy(dtype=float), dtype=float, copy=True)
        W_sorted = np.where(np.isfinite(W_sorted) & (W_sorted > 0), W_sorted, 0.0)
        Qb_clean_sorted = qb_clean_sorted * W_sorted
        Qb_clean_sorted[~np.isfinite(Qb_clean_sorted) | (Qb_clean_sorted < 0)] = 0.0

        df["qb_clean"] = qb_clean_sorted
        df["Qb_clean"] = Qb_clean_sorted
        df = df.sort_values("_original_order")

        qb_clean = np.array(df["qb_clean"].to_numpy(dtype=float), dtype=float, copy=True)
        Qb_clean = np.array(df["Qb_clean"].to_numpy(dtype=float), dtype=float, copy=True)
        outlier_count = int(np.sum(outlier_mask))
        return qb_clean, Qb_clean, outlier_count, enabled, window_min, factor

    def compute_bedload_flux(self):
        if self.energy_df is None or self.hydraulic_df is None or self.grain_df is None:
            self.import_inversion_data()
        if self.energy_df is None or self.hydraulic_df is None or self.grain_df is None:
            return

        try:
            r0, rho_s, rho_w, tau_c50, f, seismic_params, cache_key = self.prepare_inversion_parameters()
        except Exception as e:
            QMessageBox.warning(self, "参数错误", str(e))
            return

        lookup = self.compute_unit_energy_lookup(show_message=False, force=False)
        if lookup is None or lookup.empty:
            return

        data = self.interpolate_hydraulic_to_energy_time()
        if len(data) != len(lookup):
            lookup = self.compute_unit_energy_lookup(show_message=False, force=True)
            if lookup is None or lookup.empty:
                return

        E_obs = data["energy_linear"].to_numpy(dtype=float)
        W = lookup["W_eff_m"].to_numpy(dtype=float)
        E_unit_1m = lookup["unit_energy_1m_kg_m_s"].to_numpy(dtype=float)

        valid = (
            np.isfinite(E_obs) & (E_obs > 0)
            & np.isfinite(W) & (W > 0)
            & np.isfinite(E_unit_1m) & (E_unit_1m > 0)
        )

        qb_raw = np.zeros_like(E_obs, dtype=float)
        Qb_raw = np.zeros_like(E_obs, dtype=float)
        qb_raw[valid] = E_obs[valid] / (E_unit_1m[valid] * W[valid])
        qb_raw[~np.isfinite(qb_raw) | (qb_raw < 0)] = 0.0
        Qb_raw = qb_raw * np.where(np.isfinite(W) & (W > 0), W, 0.0)
        Qb_raw[~np.isfinite(Qb_raw) | (Qb_raw < 0)] = 0.0

        qb, Qb, outlier_count, outlier_enabled, outlier_window_min, outlier_factor = self.remove_flux_outliers_and_interpolate(
            lookup["time"], qb_raw, Qb_raw, W
        )

        result = lookup[["time", "water_level_m"]].copy()
        result["q_b_kg_m_s"] = qb
        result["Q_b_kg_s"] = Qb
        self.result_df = result

        self.plot_result()

        stats = self.unit_energy_stats or {}
        zero_by_unit = int(np.sum(~valid))
        lines = []
        lines.append("推移质通量反演完成")
        lines.append("-" * 80)
        lines.append("单位能量表为内存临时表，未保存为 CSV。")
        lines.append(f"反演时刻数：{len(result)}")
        lines.append(f"频率数组：{f[0]:.3g} - {f[-1]:.3g} Hz，共 {len(f)} 点")
        lines.append(f"rho_s = {rho_s:.3g} kg/m³；rho_w = {rho_w:.3g} kg/m³；tau_c50 = {tau_c50:.4g}")
        lines.append(f"r0 = {r0:.3g} m")
        lines.append(f"有效粒径为空而置零：{stats.get('zero_empty_count', 0)} 个时刻")
        lines.append(f"水力参数无效而置零：{stats.get('zero_invalid_count', 0)} 个时刻")
        lines.append(f"单位能量或观测能量无效而置零：{zero_by_unit} 个时刻")
        if outlier_enabled:
            lines.append(f"q_b 去异常值：启用，窗口 {outlier_window_min:g} min，阈值 {outlier_factor:g} × 局部中值")
            lines.append(f"已剔除并插值补点：{outlier_count} 个尖峰点")
        else:
            lines.append("qᵦ 去异常值：未启用")
        lines.append(f"单位宽通量 q_b 范围：{result['q_b_kg_m_s'].min():.4e} - {result['q_b_kg_m_s'].max():.4e} kg m⁻¹ s⁻¹")
        lines.append(f"断面通量 Q_b 范围：{result['Q_b_kg_s'].min():.4e} - {result['Q_b_kg_s'].max():.4e} kg s⁻¹")
        lines.append("\n前 10 行结果：")
        lines.append(result.head(10).to_string(index=False))
        self.preview_box.setPlainText("\n".join(lines))

    def plot_result(self, save_path=None):
        if self.result_df is None or self.result_df.empty:
            return

        df = self.result_df.copy()
        self.figure.clear()
        ax1 = self.figure.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(df["time"], df["water_level_m"], linewidth=1.4, label="Water level")
        ax2.plot(df["time"], df["Q_b_kg_s"], color="red", linewidth=1.4, label="Section bedload flux")

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Water level (m)")
        ax2.set_ylabel(r"Section bedload flux $Q_b$ (kg s$^{-1}$)")
        ax1.set_title("Water level and inverted bedload flux")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
        self.figure.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(str(save_path), dpi=300, bbox_inches="tight")

        self.canvas.draw()

    def save_results(self):
        try:
            if self.result_df is None:
                self.compute_bedload_flux()
            if self.result_df is None:
                return

            output_text = self.output_dir.text().strip()
            if output_text:
                output_dir = Path(output_text)
            else:
                output_dir = Path.cwd() / "inversion_outputs"
                self.output_dir.setText(str(output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = output_dir / "bedload_inversion_timeseries.csv"
            fig_path = output_dir / "bedload_inversion_timeseries.png"
            save_cols = ["time", "water_level_m", "q_b_kg_m_s", "Q_b_kg_s"]
            self.result_df[save_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
            self.plot_result(save_path=fig_path)

            QMessageBox.information(self, "保存成功", "已保存：\n" f"{csv_path}\n" f"{fig_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存反演结果时出错：\n{e}")

    def clear_page(self):
        if hasattr(self, "energy_input_mode_combo"):
            self.energy_input_mode_combo.setCurrentIndex(0)
        self.energy_file.clear()
        self.hydraulic_file.clear()
        self.grain_file.clear()
        self.output_dir.clear()
        self.r0_edit.setText("17")
        if hasattr(self, "outlier_enable"):
            self.outlier_enable.setChecked(True)
        if hasattr(self, "outlier_window_min_edit"):
            self.outlier_window_min_edit.setText("60")
        if hasattr(self, "outlier_factor_edit"):
            self.outlier_factor_edit.setText("5")
        self.preview_box.clear()
        self.figure.clear()
        self.canvas.draw()
        self.energy_df = None
        self.hydraulic_df = None
        self.grain_df = None
        self.unit_energy_df = None
        self.unit_energy_cache_key = None
        self.unit_energy_stats = {}
        self.result_df = None


class StationConfigPage(QWidget):
    """台站配置页面：只保存长期固定的台站属性。"""

    def __init__(self):
        super().__init__()
        self.project_page = None
        self.time_frequency_page = None
        self.sediment_page = None
        self.hydraulic_page = None
        self.inversion_page = None
        self.build_ui()

    def build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("台站配置")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        subtitle = QLabel(
            "只保存一个台站长期固定的信息：台站编号、河流名称、站点名称、河道断面、泥沙级配、台站-河道距离和地震传播参数。"
            "原始地震数据、水位序列、输出目录属于本次任务输入，应在相应模块或自动运行页面选择。"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        basic_group = QGroupBox("固定台站信息")
        basic_form = QFormLayout(basic_group)
        self.station_id_edit = QLineEdit()
        self.river_name_edit = QLineEdit()
        self.station_name_edit = QLineEdit()
        self.cross_section_file_edit = QLineEdit()
        self.grain_distribution_file_edit = QLineEdit()
        self.r0_edit = QLineEdit("17")
        basic_form.addRow("台站编号：", self.station_id_edit)
        basic_form.addRow("河流名称：", self.river_name_edit)
        basic_form.addRow("站点名称：", self.station_name_edit)
        basic_form.addRow("河道断面文件：", self.make_path_row(self.cross_section_file_edit, self.choose_cross_file))
        basic_form.addRow("泥沙级配文件：", self.make_path_row(self.grain_distribution_file_edit, self.choose_grain_file))
        basic_form.addRow("台站-河道距离 r<sub>0</sub>，m：", self.r0_edit)
        layout.addWidget(basic_group)

        seismic_group = QGroupBox("地震传播参数")
        seismic_form = QFormLayout(seismic_group)
        self.v0_edit = QLineEdit("2206")
        self.z0_edit = QLineEdit("1000")
        self.f0_edit = QLineEdit("1")
        self.a_edit = QLineEdit("0.272")
        self.Q0_edit = QLineEdit("20")
        self.eta_edit = QLineEdit("0")
        self.phi_edit = QLineEdit("0")
        self.eb_edit = QLineEdit("0.5")
        self.fx_edit = QLineEdit("0.146")
        self.fy_edit = QLineEdit("0.146")
        self.fz_edit = QLineEdit("0.539")
        self.Nzz_edit = QLineEdit("0.352")
        for label, edit in [
            ("v<sub>0</sub>（参考相速度，m/s）：", self.v0_edit), ("z<sub>0</sub>（参考深度，m）：", self.z0_edit), ("f<sub>0</sub>（参考频率，Hz）：", self.f0_edit),
            ("a（频散指数）：", self.a_edit), ("Q<sub>0</sub>（品质因子）：", self.Q0_edit), ("η（Q 的频率指数）：", self.eta_edit),
            ("φ（源–台站方位角）：", self.phi_edit), ("e<sub>b</sub>（反弹系数）：", self.eb_edit), ("f<sub>x</sub>（x 向冲量系数）：", self.fx_edit),
            ("f<sub>y</sub>（y 向冲量系数）：", self.fy_edit), ("f<sub>z</sub>（z 向冲量系数）：", self.fz_edit), ("N<sub>zz</sub>（垂直辐射项）：", self.Nzz_edit),
        ]:
            seismic_form.addRow(label, edit)
        layout.addWidget(seismic_group)

        note = QLabel(
            "说明：点击“应用到各模块”后，断面会自动填入水力参数模块，级配会自动填入泥沙粒径和反演模块，"
            "r<sub>0</sub> 与地震传播参数会自动填入反演模块；但本次地震数据、水位文件和输出目录仍由各任务页面选择。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #4b5563;")
        layout.addWidget(note)

        button_row = QHBoxLayout()
        self.fill_button = QPushButton("从当前模块填充固定信息")
        self.apply_button = QPushButton("应用到各模块")
        self.save_button = QPushButton("保存台站配置")
        self.load_button = QPushButton("加载台站配置")
        self.clear_button = QPushButton("清空")
        self.fill_button.clicked.connect(self.fill_from_current_pages)
        self.apply_button.clicked.connect(self.apply_to_pages)
        self.save_button.clicked.connect(self.save_station_config)
        self.load_button.clicked.connect(self.load_station_config)
        self.clear_button.clicked.connect(self.clear_page)
        for btn in [self.fill_button, self.apply_button, self.save_button, self.load_button, self.clear_button]:
            button_row.addWidget(btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        layout.addWidget(self.log_box)
        layout.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

    def set_context_pages(self, project_page=None, time_frequency_page=None, sediment_page=None, hydraulic_page=None, inversion_page=None):
        self.project_page = project_page
        self.time_frequency_page = time_frequency_page
        self.sediment_page = sediment_page
        self.hydraulic_page = hydraulic_page
        self.inversion_page = inversion_page

    def make_path_row(self, line_edit, callback):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("浏览")
        btn.clicked.connect(callback)
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        return row

    def choose_file(self, title, edit, file_filter="All Files (*)"):
        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if path:
            edit.setText(path)

    def choose_cross_file(self):
        self.choose_file("选择河道断面文件", self.cross_section_file_edit, "Data Files (*.csv *.txt *.xlsx *.xls);;All Files (*)")

    def choose_grain_file(self):
        self.choose_file("选择泥沙级配文件", self.grain_distribution_file_edit, "Data Files (*.csv *.txt *.xlsx *.xls);;All Files (*)")

    def get_station_config_path(self):
        project_dir = project_dir_from_page(self.project_page)
        if project_dir is not None:
            return project_dir / "station_config.json"
        return None

    def _get_float(self, edit, default):
        try:
            text = edit.text().strip()
            if not text:
                return float(default)
            return float(text)
        except Exception:
            return float(default)

    def collect_profile(self):
        seismic_params = BedloadSeismicParams(
            v0=self._get_float(self.v0_edit, 2206.0),
            z0=self._get_float(self.z0_edit, 1000.0),
            f0=self._get_float(self.f0_edit, 1.0),
            a=self._get_float(self.a_edit, 0.272),
            Q0=self._get_float(self.Q0_edit, 20.0),
            eta=self._get_float(self.eta_edit, 0.0),
            phi=self._get_float(self.phi_edit, 0.0),
            eb=self._get_float(self.eb_edit, 0.5),
            fx=self._get_float(self.fx_edit, 0.146),
            fy=self._get_float(self.fy_edit, 0.146),
            fz=self._get_float(self.fz_edit, 0.539),
            Nzz=self._get_float(self.Nzz_edit, 0.352),
        )
        return StationProfile(
            station_id=self.station_id_edit.text().strip(),
            river_name=self.river_name_edit.text().strip(),
            station_name=self.station_name_edit.text().strip(),
            cross_section_file=self.cross_section_file_edit.text().strip(),
            grain_distribution_file=self.grain_distribution_file_edit.text().strip(),
            r0_m=self._get_float(self.r0_edit, 17.0),
            seismic_params=seismic_params,
        )

    def collect_config(self):
        return self.collect_profile().to_dict()

    def set_config(self, cfg):
        profile = StationProfile.from_dict(cfg or {})
        self.station_id_edit.setText(profile.station_id)
        self.river_name_edit.setText(profile.river_name)
        self.station_name_edit.setText(profile.station_name)
        self.cross_section_file_edit.setText(profile.cross_section_file)
        self.grain_distribution_file_edit.setText(profile.grain_distribution_file)
        self.r0_edit.setText(str(profile.r0_m))
        sp = profile.seismic_params if profile.seismic_params is not None else BedloadSeismicParams()
        for key, edit in [
            ("v0", self.v0_edit), ("z0", self.z0_edit), ("f0", self.f0_edit), ("a", self.a_edit),
            ("Q0", self.Q0_edit), ("eta", self.eta_edit), ("phi", self.phi_edit), ("eb", self.eb_edit),
            ("fx", self.fx_edit), ("fy", self.fy_edit), ("fz", self.fz_edit), ("Nzz", self.Nzz_edit),
        ]:
            edit.setText(str(getattr(sp, key)))

    def fill_from_current_pages(self):
        """从已有模块读取固定信息，但绝不让空字段覆盖台站配置页已有内容。"""
        lines = []

        if self.project_page is not None:
            river = widget_text(getattr(self.project_page, "river_name", None), "")
            station = widget_text(getattr(self.project_page, "station_name", None), "")
            if river:
                self.river_name_edit.setText(river)
                lines.append(f"河流名称 ← 项目管理：{river}")
            else:
                lines.append("河流名称：项目管理为空，保留当前输入。")
            if station:
                self.station_name_edit.setText(station)
                lines.append(f"站点名称 ← 项目管理：{station}")
            else:
                lines.append("站点名称：项目管理为空，保留当前输入。")

        if self.hydraulic_page is not None:
            cross = widget_text(getattr(self.hydraulic_page, "cross_file", None), "")
            if cross:
                self.cross_section_file_edit.setText(cross)
                lines.append(f"河道断面文件 ← 水力参数模块：{cross}")
            else:
                lines.append("河道断面文件：水力参数模块为空，保留当前输入。")

        if self.sediment_page is not None:
            candidate = widget_text(getattr(self.sediment_page, "grain_file", None), "")
            out = widget_text(getattr(self.sediment_page, "output_dir", None), "")
            if out:
                for name in ["standard_grain_distribution.csv", "generated_grain_distribution.csv"]:
                    p = Path(out) / name
                    if p.exists():
                        candidate = str(p)
                        break
            if candidate:
                self.grain_distribution_file_edit.setText(candidate)
                lines.append(f"泥沙级配文件 ← 泥沙粒径模块：{candidate}")
            else:
                lines.append("泥沙级配文件：泥沙粒径模块为空，保留当前输入。")

        if self.inversion_page is not None:
            r0 = widget_text(getattr(self.inversion_page, "r0_edit", None), "")
            if r0:
                self.r0_edit.setText(r0)
                lines.append(f"r0 ← 反演模块：{r0}")
            else:
                lines.append("r0：反演模块为空，保留当前输入。")

            copied_params = []
            for key in ["v0", "z0", "f0", "a", "Q0", "eta", "phi", "eb", "fx", "fy", "fz", "Nzz"]:
                src = getattr(self.inversion_page, f"{key}_edit", None)
                dst = getattr(self, f"{key}_edit", None)
                value = widget_text(src, "")
                if value:
                    set_widget_text(dst, value)
                    copied_params.append(key)
            if copied_params:
                lines.append("传播参数 ← 反演模块：" + ", ".join(copied_params))
            else:
                lines.append("传播参数：反演模块为空，保留当前输入。")

        self.log_box.setPlainText("已从当前模块填充固定台站信息。空字段不会覆盖已有输入。\n\n" + "\n".join(lines))

    def apply_to_pages(self):
        """把固定台站信息应用到模块；空路径不覆盖模块已有路径。"""
        profile = self.collect_profile()
        lines = []
        if self.sediment_page is not None:
            if profile.grain_distribution_file:
                set_widget_text_if_nonempty(getattr(self.sediment_page, "grain_file", None), profile.grain_distribution_file)
                lines.append("级配文件已应用到泥沙粒径模块。")
            else:
                lines.append("级配文件为空，未覆盖泥沙粒径模块。")
        if self.hydraulic_page is not None:
            if profile.cross_section_file:
                set_widget_text_if_nonempty(getattr(self.hydraulic_page, "cross_file", None), profile.cross_section_file)
                lines.append("断面文件已应用到水力参数模块。")
            else:
                lines.append("断面文件为空，未覆盖水力参数模块。")
        if self.inversion_page is not None:
            if profile.grain_distribution_file:
                set_widget_text_if_nonempty(getattr(self.inversion_page, "grain_file", None), profile.grain_distribution_file)
                lines.append("级配文件已应用到反演模块。")
            set_widget_text(getattr(self.inversion_page, "r0_edit", None), str(profile.r0_m))
            sp = profile.seismic_params if profile.seismic_params is not None else BedloadSeismicParams()
            for key in ["v0", "z0", "f0", "a", "Q0", "eta", "phi", "eb", "fx", "fy", "fz", "Nzz"]:
                dst = getattr(self.inversion_page, f"{key}_edit", None)
                set_widget_text(dst, str(getattr(sp, key)))
            lines.append("r₀ 和传播参数已应用到反演模块。")
        self.log_box.setPlainText("固定台站信息已应用。\n\n" + "\n".join(lines))

    def save_station_config(self):
        cfg = self.collect_config()
        path = self.get_station_config_path()
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "保存台站配置", "station_config.json", "JSON Files (*.json);;All Files (*)")
            if not path:
                return
            path = Path(path)
        save_json_safely(path, cfg)
        try:
            save_project_config_from_page(self.project_page, {"station_config_file": str(path), "station_profile": cfg})
        except Exception:
            pass
        QMessageBox.information(self, "保存成功", f"台站配置已保存：\n{path}")

    def load_station_config(self):
        default_path = self.get_station_config_path()
        if default_path is not None and default_path.exists():
            path = str(default_path)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "加载台站配置", "", "JSON Files (*.json);;All Files (*)")
            if not path:
                return
        cfg = load_json_safely(path)
        self.set_config(cfg)
        self.apply_to_pages()
        QMessageBox.information(self, "加载成功", "台站配置已加载，并已把固定信息应用到当前模块。")

    def clear_page(self):
        for edit in [
            self.station_id_edit, self.river_name_edit, self.station_name_edit,
            self.cross_section_file_edit, self.grain_distribution_file_edit,
        ]:
            edit.clear()
        self.r0_edit.setText("17")
        for key, value in {
            "v0": "2206", "z0": "1000", "f0": "1", "a": "0.272", "Q0": "20", "eta": "0",
            "phi": "0", "eb": "0.5", "fx": "0.146", "fy": "0.146", "fz": "0.539", "Nzz": "0.352",
        }.items():
            getattr(self, f"{key}_edit").setText(value)
        self.log_box.clear()


class AutoRunPage(QWidget):
    """批量反演页面：基于项目配置自动检查并执行批量反演。"""

    def __init__(self):
        super().__init__()
        self.project_page = None
        self.station_page = None
        self.seismic_page = None
        self.time_frequency_page = None
        self.sediment_page = None
        self.hydraulic_page = None
        self.inversion_page = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.scan_once)
        self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title = QLabel("批量反演")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        subtitle = QLabel("用于长时序或全年数据计算。输入原始 SAC 目录和水位表后，程序自动完成去响应、UTC+8 时间、30–80 Hz 能量、水力参数和通量反演；页面不显示图片，结果和每日图件直接保存到输出文件夹。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        option_group = QGroupBox("运行选项")
        form = QFormLayout(option_group)
        self.skip_existing_check = QCheckBox("跳过已经存在的中间结果")
        self.skip_existing_check.setChecked(True)
        self.save_after_run_check = QCheckBox("计算完成后自动保存结果")
        self.save_after_run_check.setChecked(True)
        form.addRow("断点续算：", self.skip_existing_check)
        form.addRow("自动保存：", self.save_after_run_check)
        layout.addWidget(option_group)

        batch_group = QGroupBox("批量输入")
        batch_form = QFormLayout(batch_group)
        self.batch_energy_dir = QLineEdit()
        self.batch_water_file = QLineEdit()
        self.batch_start_time = QLineEdit()
        self.batch_end_time = QLineEdit()
        self.batch_output_dir = QLineEdit()
        batch_form.addRow("原始 SAC 数据目录：", self.make_path_row(self.batch_energy_dir, self.choose_batch_energy_dir, is_dir=True))
        batch_form.addRow("全年/批量水位文件：", self.make_path_row(self.batch_water_file, self.choose_batch_water_file))
        batch_form.addRow("批量输出文件夹：", self.make_path_row(self.batch_output_dir, self.choose_batch_output_dir, is_dir=True))
        batch_form.addRow("开始时间，可空：", self.batch_start_time)
        batch_form.addRow("结束时间，可空：", self.batch_end_time)
        layout.addWidget(batch_group)

        button_row = QHBoxLayout()
        self.scan_button = QPushButton("检查输入 / 扫描数据")
        self.batch_button = QPushButton("开始批量反演")
        self.stop_button = QPushButton("停止")
        self.scan_button.clicked.connect(self.scan_once)
        self.batch_button.clicked.connect(self.run_batch_inversion)
        self.stop_button.clicked.connect(self.stop_timer)
        for btn in [self.scan_button, self.batch_button, self.stop_button]:
            button_row.addWidget(btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(260)
        layout.addWidget(self.status_box)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def set_context_pages(self, project_page=None, station_page=None, seismic_page=None, time_frequency_page=None, sediment_page=None, hydraulic_page=None, inversion_page=None):
        self.project_page = project_page
        self.station_page = station_page
        self.seismic_page = seismic_page
        self.time_frequency_page = time_frequency_page
        self.sediment_page = sediment_page
        self.hydraulic_page = hydraulic_page
        self.inversion_page = inversion_page

    def make_path_row(self, edit, callback, is_dir=False):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("浏览")
        btn.clicked.connect(callback)
        layout.addWidget(edit)
        layout.addWidget(btn)
        return row

    def choose_batch_energy_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择原始 SAC 数据目录")
        if path:
            self.batch_energy_dir.setText(path)

    def choose_batch_water_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择批量水位文件", "", "Data Files (*.csv *.txt *.xlsx *.xls);;All Files (*)")
        if path:
            self.batch_water_file.setText(path)

    def choose_batch_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择批量输出文件夹")
        if path:
            self.batch_output_dir.setText(path)

    def append_status(self, msg):
        self.status_box.append(msg)

    def load_and_apply_station(self):
        """批量反演页面不再依赖单独的台站配置页，而是直接应用项目管理中的固定配置。"""
        if self.project_page is None:
            QMessageBox.warning(self, "提示", "没有找到项目管理页面。")
            return
        try:
            self.project_page.apply_project_config_to_modules(show_message=False)
            cfg = self.project_page.get_project_data()
            project = cfg.get("project", {})
            station = cfg.get("station", {})
            out_dir = str(project.get("output_dir", "")).strip()
            if out_dir and not self.batch_output_dir.text().strip():
                self.batch_output_dir.setText(str(Path(out_dir) / "batch_inversion"))
            lines = [
                "项目配置已应用到相关模块。",
                "- 批量反演将直接使用项目管理中的断面、级配、坡降、r₀ 和传播参数。",
                f"- 台站编号：{station.get('station_id', '') or '未填写'}",
                f"- 河流名称：{station.get('river_name', '') or '未填写'}",
                f"- 站点名称：{station.get('station_name', '') or '未填写'}",
                f"- 河道断面：{station.get('cross_section_file', '') or '未填写'}",
                f"- 泥沙级配：{station.get('grain_distribution_file', '') or '未填写'}",
                f"- 河床坡降 S：{station.get('slope', '')}",
                f"- 台站-河道距离 r0：{station.get('r0_m', '')}",
            ]
            self.status_box.setPlainText("\n".join(lines))
        except Exception:
            detail = traceback.format_exc()
            self.status_box.setPlainText("应用项目配置失败\n" + "-" * 60 + "\n" + detail)
            QMessageBox.critical(self, "应用项目配置失败", "详细错误已显示在日志框中。")
    def scan_once(self):
        try:
            if self.project_page is not None:
                self.project_page.apply_project_config_to_modules(show_message=False)
            cfg = self.project_page.get_project_data() if self.project_page is not None else {}
            station = cfg.get("station", {}) if isinstance(cfg, dict) else {}
            raw_dir_text = self.batch_energy_dir.text().strip()
            raw_dir = Path(raw_dir_text) if raw_dir_text else None
            water_text = self.batch_water_file.text().strip()
            water_file = Path(water_text) if water_text else None
            output_dir = self.batch_output_dir.text().strip() or "未设置"
            sac_count = 0
            if raw_dir is not None and raw_dir.exists():
                sac_count = len(list(raw_dir.rglob("*.sac"))) + len(list(raw_dir.rglob("*.SAC")))
            lines = ["检查/扫描完成", "-" * 60]
            lines.append("本页面输入的是原始 SAC 目录和原始水位表，不需要预先提供地震能量 CSV 或水力参数 CSV。")
            lines.append("计算时将自动执行：原始 SAC → 去仪器响应 → UTC+8 时间 → 30–80 Hz 能量 → 水力参数 → 推移质通量")
            lines.append("")
            lines.append(f"原始 SAC 数据目录：{raw_dir if raw_dir else '未选择'}")
            lines.append(f"发现 SAC 文件数：{sac_count}")
            lines.append(f"全年/批量水位文件：{water_file if water_file else '未选择'}")
            lines.append(f"批量输出文件夹：{output_dir}")
            lines.append("")
            lines.append(f"河道断面：{station.get('cross_section_file', '') or '未填写'}")
            lines.append(f"泥沙级配：{station.get('grain_distribution_file', '') or '未填写'}")
            lines.append(f"河床坡降 S：{station.get('slope', '')}")
            lines.append(f"台站-河道距离 r₀：{station.get('r0_m', '')}")
            self.status_box.setPlainText("\n".join(lines))
        except Exception as e:
            self.status_box.setPlainText("扫描失败\n" + "-" * 60 + "\n" + traceback.format_exc())
            QMessageBox.critical(self, "扫描失败", str(e))

    def run_current_inversion(self):
        """兼容旧版本按钮名称：批量页面的计算统一走原始 SAC → 水位 → 水力参数 → 反演流水线。"""
        self.run_batch_inversion()

    def start_timer(self):
        QMessageBox.information(self, "提示", "当前版本批量反演页面已简化为：检查输入/扫描数据、开始批量反演、停止。")

    def stop_timer(self):
        self.timer.stop()
        self.append_status("自动扫描已停止。")

    def check_batch_ready(self):
        missing = []
        if self.project_page is None:
            missing.append("项目管理页面未初始化")
            return missing, {}
        cfg = {}
        try:
            cfg = self.project_page.get_project_data()
        except Exception:
            cfg = {}
        station = cfg.get("station", {})
        project = cfg.get("project", {})
        seismic = cfg.get("seismic_params", {})
        for label, key in [("台站编号", "station_id"), ("河流名称", "river_name"), ("站点名称", "station_name")]:
            if not str(station.get(key, "")).strip():
                missing.append(f"{label}未设置")
        cross = str(station.get("cross_section_file", "")).strip()
        grain = str(station.get("grain_distribution_file", "")).strip()
        if not cross or not Path(cross).exists():
            missing.append("河道断面文件不存在或未设置")
        if not grain or not Path(grain).exists():
            missing.append("泥沙级配文件不存在或未设置")
        try:
            if float(station.get("slope", 0)) <= 0:
                missing.append("河床坡降 S 必须大于 0")
        except Exception:
            missing.append("河床坡降 S 必须为数字")
        try:
            if float(station.get("r0_m", 0)) <= 0:
                missing.append("台站-河道距离 r₀ 必须大于 0")
        except Exception:
            missing.append("台站-河道距离 r₀ 必须为数字")
        energy_dir = self.batch_energy_dir.text().strip()
        water_file = self.batch_water_file.text().strip()
        output_dir = self.batch_output_dir.text().strip()
        if not energy_dir or not Path(energy_dir).exists():
            missing.append("原始 SAC 数据目录不存在或未选择")
        if not water_file or not Path(water_file).exists():
            missing.append("全年/批量水位文件不存在或未选择")
        if not output_dir:
            default_out = project.get("output_dir") or str(Path.cwd() / "batch_outputs")
            output_dir = str(Path(default_out) / "batch_inversion")
            self.batch_output_dir.setText(output_dir)
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            missing.append("批量输出文件夹无法创建")
        return missing, {"station": station, "project": project, "seismic_params": seismic, "raw_sac_dir": energy_dir, "water_file": water_file, "output_dir": output_dir}

    def save_daily_figures(self, result_df, output_dir):
        daily_dir = Path(output_dir) / "inversion" / "daily_figures"
        daily_dir.mkdir(parents=True, exist_ok=True)
        df = result_df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        for date_value, group in df.groupby(df["time"].dt.date):
            if group.empty:
                continue
            fig = Figure(figsize=(9, 4.2), dpi=120)
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.plot(group["time"], group["water_level_m"], linewidth=1.2)
            ax2.plot(group["time"], group["Q_b_kg_s"], color="red", linewidth=1.2)
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Water level (m)")
            ax2.set_ylabel(r"$Q_b$ (kg s$^{-1}$)")
            ax1.set_title(str(date_value))
            ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(str(daily_dir / f"{date_value}_bedload_inversion.png"), dpi=300, bbox_inches="tight")
        return daily_dir

    def get_batch_sac_files(self, raw_dir):
        raw_dir = Path(raw_dir)
        files = []
        seen = set()
        for f in raw_dir.rglob("*"):
            if not f.is_file() or f.suffix.lower() != ".sac":
                continue
            if "_processed" in f.stem.lower() or "_clean" in f.stem.lower():
                continue
            try:
                key = str(f.resolve()).lower()
            except Exception:
                key = str(f).lower()
            if key in seen:
                continue
            seen.add(key)
            files.append(f)
        return sorted(files, key=lambda p: str(p).lower())

    def batch_process_sac_files(self, raw_dir, processed_dir):
        """批量预处理原始 SAC 文件，输出处理后的 SAC 文件列表。"""
        raw_dir = Path(raw_dir)
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        sac_files = self.get_batch_sac_files(raw_dir)
        if not sac_files:
            raise ValueError("原始 SAC 数据目录下没有 SAC 文件。")

        try:
            from obspy import read, Stream, read_inventory
        except Exception as e:
            raise ImportError(f"当前环境缺少 ObsPy，无法批量预处理 SAC：{e}")

        mode = "STS2_PAZ_VEL"
        response_path = ""
        pre_filt = (0.01, 0.02, 45.0, 50.0)
        if self.seismic_page is not None:
            mode = self.seismic_page.response_mode.currentData() or "STS2_PAZ_VEL"
            response_path = self.seismic_page.response_file.text().strip()
            try:
                pre_filt = self.seismic_page.get_pre_filt()
            except Exception:
                pre_filt = (0.01, 0.02, 45.0, 50.0)

        if mode == "VEL" and (not response_path or not Path(response_path).exists()):
            raise ValueError("批量预处理选择了外部响应文件模式，但响应文件未设置或不存在。")
        inv = read_inventory(response_path) if mode == "VEL" else None
        paz_sts2 = paz_5hz = None
        if mode == "STS2_PAZ_VEL":
            if self.seismic_page is not None:
                paz_sts2, paz_5hz = self.seismic_page.get_builtin_sts2_paz()
            else:
                temp_page = SeismicPreprocessPage()
                paz_sts2, paz_5hz = temp_page.get_builtin_sts2_paz()

        def file_has_bjt_marker(path):
            try:
                st_check = read(str(path))
                if len(st_check) == 0:
                    return False
                tr_check = st_check[0]
                return str(tr_check.stats.sac.kuser0).strip().upper() == "BJT"
            except Exception:
                return False

        def apply_batch_beijing_time(tr):
            convert_enabled = True
            if self.seismic_page is not None and hasattr(self.seismic_page, "convert_to_beijing"):
                convert_enabled = self.seismic_page.convert_to_beijing.isChecked()
            if convert_enabled:
                already_bjt = False
                try:
                    already_bjt = str(tr.stats.sac.kuser0).strip().upper() == "BJT"
                except Exception:
                    already_bjt = False
                if not already_bjt:
                    tr.stats.starttime = tr.stats.starttime + 8 * 3600
                    try:
                        tr.stats.sac.kuser0 = "BJT"
                    except Exception:
                        pass
            return tr

        processed_files = []
        total = len(sac_files)
        for i, sac_file in enumerate(sac_files, start=1):
            rel_stem = "_".join(sac_file.relative_to(raw_dir).with_suffix("").parts)
            out_path = processed_dir / f"{rel_stem}_processed.sac"
            if self.skip_existing_check.isChecked() and out_path.exists():
                if file_has_bjt_marker(out_path):
                    processed_files.append(out_path)
                    continue
                self.append_status(f"已有中间 SAC 缺少 BJT 时间标记，将重新生成：{out_path.name}")
            try:
                st = read(str(sac_file))
                if len(st) == 0:
                    self.append_status(f"跳过空 SAC：{sac_file}")
                    continue
                st.merge(method=1, fill_value="interpolate")
                tr = st[0].copy()
                try:
                    tr.detrend("demean")
                    tr.detrend("linear")
                except Exception:
                    pass
                if mode == "STS2_PAZ_VEL":
                    tr.simulate(paz_remove=paz_sts2, paz_simulate=paz_5hz, taper=False)
                elif mode == "VEL":
                    st2 = Stream(traces=[tr])
                    st2.remove_response(inventory=inv, output="VEL", pre_filt=pre_filt, water_level=60)
                    tr = st2[0]
                # RAW 模式不做响应处理。
                # 批量预处理与单文件预处理保持一致：在保存 processed SAC 前统一写成北京时间并标记 BJT。
                tr = apply_batch_beijing_time(tr)
                tr.write(str(out_path), format="SAC")
                processed_files.append(out_path)
            except Exception as e:
                self.append_status(f"SAC 预处理失败：{sac_file.name} | {e}")
            if i % 5 == 0 or i == total:
                self.progress_bar.setValue(5 + int(20 * i / max(total, 1)))
                QApplication.processEvents()
        if not processed_files:
            raise ValueError("没有成功生成任何处理后的 SAC 文件。")
        return processed_files

    def batch_compute_energy_from_processed_sac(self, processed_files, energy_out):
        """批量计算处理后 SAC 的 30–80 Hz 线性能量，并合并成长表。"""
        if self.time_frequency_page is None:
            raise ValueError("没有找到地震信号处理/时频分析页面。")
        energy_out = Path(energy_out)
        npz_dir = energy_out / "npz"
        fig_dir = energy_out / "figures"
        csv_dir = energy_out / "energy_csv"
        for d in [npz_dir, fig_dir, csv_dir]:
            d.mkdir(parents=True, exist_ok=True)
        params = self.time_frequency_page.get_params()
        csv_files = []
        total = len(processed_files)
        for i, sac_file in enumerate(processed_files, start=1):
            stem = Path(sac_file).stem
            fmin_str = f"{params['energy_fmin']:g}".replace(".", "p")
            fmax_str = f"{params['energy_fmax']:g}".replace(".", "p")
            npz_path = npz_dir / f"{stem}_PSD.npz"
            png_path = fig_dir / f"{stem}_PSD.png"
            energy_csv_path = csv_dir / f"{stem}_energy_{fmin_str}_{fmax_str}Hz.csv"
            if self.skip_existing_check.isChecked() and energy_csv_path.exists():
                csv_files.append(energy_csv_path)
            else:
                ok = self.time_frequency_page.process_one_sac_file(
                    sac_path=Path(sac_file),
                    npz_path=npz_path,
                    png_path=png_path,
                    energy_csv_path=energy_csv_path,
                    params=params,
                )
                if ok:
                    csv_files.append(energy_csv_path)
            if i % 5 == 0 or i == total:
                self.progress_bar.setValue(25 + int(20 * i / max(total, 1)))
                QApplication.processEvents()
        if not csv_files:
            raise ValueError("没有成功生成任何地震能量 CSV。")
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            if "time" in df.columns and "energy_linear" in df.columns:
                dfs.append(df[["time", "energy_linear"]])
        if not dfs:
            raise ValueError("生成的地震能量 CSV 中没有 time 和 energy_linear 字段。")
        merged = pd.concat(dfs, ignore_index=True)
        merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
        merged["energy_linear"] = pd.to_numeric(merged["energy_linear"], errors="coerce")
        merged = merged.dropna(subset=["time", "energy_linear"]).sort_values("time").drop_duplicates("time")
        if self.batch_start_time.text().strip():
            merged = merged[merged["time"] >= pd.to_datetime(self.batch_start_time.text().strip())]
        if self.batch_end_time.text().strip():
            merged = merged[merged["time"] <= pd.to_datetime(self.batch_end_time.text().strip())]
        energy_path = energy_out / "energy_timeseries.csv"
        merged.to_csv(energy_path, index=False, encoding="utf-8-sig")
        return energy_path

    def run_batch_inversion(self):
        """批量模式：原始 SAC 自动预处理与时频分析，原始水位自动计算水力参数，再反演并按天保存图件。"""
        missing, cfg = self.check_batch_ready()
        if missing:
            msg = "批量反演无法开始，缺少以下信息：\n\n" + "\n".join([f"- {m}" for m in missing])
            self.status_box.setPlainText(msg)
            QMessageBox.warning(self, "配置不完整", msg)
            return
        if self.inversion_page is None or self.hydraulic_page is None or self.time_frequency_page is None:
            QMessageBox.warning(self, "提示", "没有找到水力参数、地震信号处理或反演计算页面。")
            return
        try:
            self.progress_bar.setValue(0)
            station = cfg["station"]
            out_root = Path(cfg["output_dir"])
            seismic_out = out_root / "seismic" / "processed_sac"
            energy_out = out_root / "energy"
            hyd_out = out_root / "hydraulic"
            inv_out = out_root / "inversion"
            log_out = out_root / "logs"
            for d in [seismic_out, energy_out, hyd_out, inv_out, log_out]:
                d.mkdir(parents=True, exist_ok=True)

            self.status_box.setPlainText("正在批量预处理原始 SAC...\n")
            processed_files = self.batch_process_sac_files(cfg["raw_sac_dir"], seismic_out)
            self.status_box.append(f"已生成/发现处理后 SAC：{len(processed_files)} 个。")
            self.progress_bar.setValue(25)
            QApplication.processEvents()

            self.status_box.append("正在计算 30–80 Hz 地震能量长表...")
            energy_path = self.batch_compute_energy_from_processed_sac(processed_files, energy_out)
            self.status_box.append(f"地震能量长表：{energy_path}")
            self.progress_bar.setValue(45)
            QApplication.processEvents()

            self.status_box.append("正在根据全年水位自动计算水力参数...")
            self.hydraulic_page.cross_file.setText(str(station["cross_section_file"]))
            self.hydraulic_page.water_file.setText(str(cfg["water_file"]))
            self.hydraulic_page.output_dir.setText(str(hyd_out))
            self.hydraulic_page.slope_edit.setText(str(station["slope"]))
            self.hydraulic_page.cross_df_standard = None
            self.hydraulic_page.water_df_standard = None
            self.hydraulic_page.effective_df = None
            self.hydraulic_page.read_cross_file(show_message=False)
            self.hydraulic_page.read_water_file(show_message=False)
            self.hydraulic_page.compute_hydraulic_parameters()
            self.hydraulic_page.save_results()
            hydraulic_path = hyd_out / "effective_depth_width.csv"
            self.progress_bar.setValue(65)
            QApplication.processEvents()

            self.status_box.append("正在反演推移质通量...")
            self.inversion_page.energy_file.setText(str(energy_path))
            self.inversion_page.hydraulic_file.setText(str(hydraulic_path))
            self.inversion_page.grain_file.setText(str(station["grain_distribution_file"]))
            self.inversion_page.output_dir.setText(str(inv_out))
            self.inversion_page.r0_edit.setText(str(station["r0_m"]))
            if self.project_page is not None:
                self.project_page.apply_project_config_to_modules(show_message=False)
            self.inversion_page.import_inversion_data()
            self.inversion_page.compute_unit_energy_lookup(show_message=False, force=True)
            self.inversion_page.compute_bedload_flux()
            self.inversion_page.save_results()
            self.progress_bar.setValue(85)
            QApplication.processEvents()

            self.status_box.append("正在按天保存水位–推移质通量图件...")
            daily_dir = self.save_daily_figures(self.inversion_page.result_df, out_root)
            self.progress_bar.setValue(100)
            lines = [
                "批量反演完成", "-" * 70,
                f"处理后 SAC 目录：{seismic_out}",
                f"地震能量长表：{energy_path}",
                f"水力参数表：{hydraulic_path}",
                f"反演结果表：{inv_out / 'bedload_inversion_timeseries.csv'}",
                f"每日图件目录：{daily_dir}",
            ]
            self.status_box.setPlainText("\n".join(lines))
        except Exception:
            detail = traceback.format_exc()
            self.status_box.setPlainText("批量反演失败\n" + "-" * 60 + "\n" + detail)
            QMessageBox.critical(self, "批量反演失败", "详细错误已显示在日志框中。")


class SystemSettingsPage(QWidget):
    """系统设置页面：保存默认参数、缓存和自动化策略。"""

    def __init__(self):
        super().__init__()
        self.project_page = None
        self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)
        title = QLabel("系统设置")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        subtitle = QLabel("保存全局默认参数、自动运行策略、缓存和日志设置。反演时优先读取台站配置，其次读取这里的默认值。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        param_group = QGroupBox("默认物理与计算参数")
        form = QFormLayout(param_group)
        self.rho_s_edit = QLineEdit("2700")
        self.rho_w_edit = QLineEdit("1000")
        self.theta_c_edit = QLineEdit("0.045")
        self.energy_fmin_edit = QLineEdit("30")
        self.energy_fmax_edit = QLineEdit("80")
        self.energy_nfreq_edit = QLineEdit("51")
        self.outlier_window_edit = QLineEdit("60")
        self.outlier_factor_edit = QLineEdit("5")
        form.addRow("沉积物密度 ρ<sub>s</sub>，kg/m³：", self.rho_s_edit)
        form.addRow("水体密度 ρ<sub>w</sub>，kg/m³：", self.rho_w_edit)
        form.addRow("临界 Shields 参数 θ<sub>c</sub>：", self.theta_c_edit)
        form.addRow("能量频带下限，Hz：", self.energy_fmin_edit)
        form.addRow("能量频带上限，Hz：", self.energy_fmax_edit)
        form.addRow("理论频率点数：", self.energy_nfreq_edit)
        form.addRow("q<sub>b</sub> 去异常窗口，min：", self.outlier_window_edit)
        form.addRow("q<sub>b</sub> 异常阈值倍数：", self.outlier_factor_edit)
        layout.addWidget(param_group)

        auto_group = QGroupBox("自动化与缓存")
        auto_form = QFormLayout(auto_group)
        self.enable_cache_check = QCheckBox("启用缓存和断点续算")
        self.enable_cache_check.setChecked(True)
        self.skip_processed_check = QCheckBox("自动跳过已处理文件")
        self.skip_processed_check.setChecked(True)
        self.save_intermediate_check = QCheckBox("保存中间结果")
        self.save_intermediate_check.setChecked(True)
        self.log_dir_edit = QLineEdit()
        auto_form.addRow("缓存：", self.enable_cache_check)
        auto_form.addRow("跳过已处理：", self.skip_processed_check)
        auto_form.addRow("中间结果：", self.save_intermediate_check)
        auto_form.addRow("日志目录：", self.make_path_row(self.log_dir_edit, self.choose_log_dir))
        layout.addWidget(auto_group)

        button_row = QHBoxLayout()
        self.save_button = QPushButton("保存系统设置")
        self.load_button = QPushButton("加载系统设置")
        self.apply_button = QPushButton("应用到当前页面")
        self.save_button.clicked.connect(self.save_settings)
        self.load_button.clicked.connect(self.load_settings)
        self.apply_button.clicked.connect(self.apply_to_current_pages)
        for btn in [self.save_button, self.load_button, self.apply_button]:
            button_row.addWidget(btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        layout.addWidget(self.log_box)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def set_context_pages(self, project_page=None, inversion_page=None, time_frequency_page=None, hydraulic_page=None):
        self.project_page = project_page
        self.inversion_page = inversion_page
        self.time_frequency_page = time_frequency_page
        self.hydraulic_page = hydraulic_page

    def make_path_row(self, edit, callback):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("浏览")
        btn.clicked.connect(callback)
        layout.addWidget(edit)
        layout.addWidget(btn)
        return row

    def choose_log_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择日志目录")
        if path:
            self.log_dir_edit.setText(path)

    def collect_settings(self):
        return {
            "defaults": {
                "rho_s": self.rho_s_edit.text().strip(),
                "rho_w": self.rho_w_edit.text().strip(),
                "theta_c": self.theta_c_edit.text().strip(),
                "energy_fmin": self.energy_fmin_edit.text().strip(),
                "energy_fmax": self.energy_fmax_edit.text().strip(),
                "energy_nfreq": self.energy_nfreq_edit.text().strip(),
                "outlier_window_min": self.outlier_window_edit.text().strip(),
                "outlier_factor": self.outlier_factor_edit.text().strip(),
            },
            "automation": {
                "enable_cache": self.enable_cache_check.isChecked(),
                "skip_processed": self.skip_processed_check.isChecked(),
                "save_intermediate": self.save_intermediate_check.isChecked(),
                "log_dir": self.log_dir_edit.text().strip(),
            },
        }

    def set_settings(self, data):
        data = data or {}
        defaults = data.get("defaults", {})
        automation = data.get("automation", {})
        for key, edit in [
            ("rho_s", self.rho_s_edit), ("rho_w", self.rho_w_edit), ("theta_c", self.theta_c_edit),
            ("energy_fmin", self.energy_fmin_edit), ("energy_fmax", self.energy_fmax_edit),
            ("energy_nfreq", self.energy_nfreq_edit), ("outlier_window_min", self.outlier_window_edit),
            ("outlier_factor", self.outlier_factor_edit),
        ]:
            if key in defaults:
                edit.setText(str(defaults.get(key)))
        self.enable_cache_check.setChecked(bool(automation.get("enable_cache", True)))
        self.skip_processed_check.setChecked(bool(automation.get("skip_processed", True)))
        self.save_intermediate_check.setChecked(bool(automation.get("save_intermediate", True)))
        self.log_dir_edit.setText(str(automation.get("log_dir", "")))

    def save_settings(self):
        settings = self.collect_settings()
        try:
            path = save_project_config_from_page(self.project_page, {"system_settings": settings})
            QMessageBox.information(self, "保存成功", f"系统设置已保存到：\n{path}")
        except Exception:
            path, _ = QFileDialog.getSaveFileName(self, "保存系统设置", "system_settings.json", "JSON Files (*.json);;All Files (*)")
            if not path:
                return
            save_json_safely(path, settings)
            QMessageBox.information(self, "保存成功", f"系统设置已保存到：\n{path}")

    def load_settings(self):
        cfg = load_project_config_from_page(self.project_page)
        settings = cfg.get("system_settings", {})
        if not settings:
            path, _ = QFileDialog.getOpenFileName(self, "加载系统设置", "", "JSON Files (*.json);;All Files (*)")
            if not path:
                return
            settings = load_json_safely(path)
        self.set_settings(settings)
        self.log_box.setPlainText("系统设置已加载。")

    def apply_to_current_pages(self):
        if getattr(self, "time_frequency_page", None) is not None:
            set_widget_text(getattr(self.time_frequency_page, "energy_fmin_edit", None), self.energy_fmin_edit.text().strip())
            set_widget_text(getattr(self.time_frequency_page, "energy_fmax_edit", None), self.energy_fmax_edit.text().strip())
        if getattr(self, "hydraulic_page", None) is not None:
            set_widget_text(getattr(self.hydraulic_page, "rho_w_edit", None), self.rho_w_edit.text().strip())
            set_widget_text(getattr(self.hydraulic_page, "theta_c_edit", None), self.theta_c_edit.text().strip())
        if getattr(self, "inversion_page", None) is not None:
            set_widget_text(getattr(self.inversion_page, "outlier_window_min_edit", None), self.outlier_window_edit.text().strip())
            set_widget_text(getattr(self.inversion_page, "outlier_factor_edit", None), self.outlier_factor_edit.text().strip())
        self.log_box.setPlainText("系统设置已应用到当前相关页面。")


class SeismicSignalProcessingPage(QWidget):
    """地震信号处理：合并地震预处理和时频能量计算。"""

    processed_sac_saved = Signal(str)
    energy_csv_saved = Signal(str)

    def __init__(self):
        super().__init__()
        self.preprocess_page = SeismicPreprocessPage()
        self.time_frequency_page = TimeFrequencyPage()
        self.preprocess_page.processed_sac_saved.connect(self.time_frequency_page.set_input_from_processed_sac)
        self.preprocess_page.processed_sac_saved.connect(self.processed_sac_saved.emit)
        self.time_frequency_page.energy_csv_saved.connect(self.energy_csv_saved.emit)
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.addTab(self.preprocess_page, "1 地震预处理")
        tabs.addTab(self.time_frequency_page, "2 时频能量计算")
        layout.addWidget(tabs)


class FileFormatInfoPage(QWidget):
    """文件格式说明页面。"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        title = QLabel("文件格式说明")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setHtml(
            """
            <h3>河道断面文件</h3>
            <p>必须包含：<code>x</code>, <code>bed_elevation</code></p>
            <pre>x,bed_elevation
0,1521.48
1,1521.36</pre>

            <h3>水位序列文件</h3>
            <p>必须包含：<code>time</code>, <code>water_level</code></p>
            <pre>time,water_level
2025-06-01 10:00:00,1523.20</pre>

            <h3>泥沙级配文件</h3>
            <p>必须包含：<code>D_mm</code>, <code>fraction</code>。fraction 可为比例或百分比，程序会自动归一化。</p>
            <pre>D_mm,fraction
0.5,0.05
1.0,0.20</pre>

            <h3>地震能量文件</h3>
            <p>必须包含：<code>time</code>, <code>energy_linear</code>。<code>energy_linear</code> 是 30–80 Hz 线性能量中值。</p>

            <h3>水力参数输出文件</h3>
            <p><code>effective_depth_width.csv</code>：</p>
            <pre>time, water_level_m, H_eff_m, W_eff_m, D_effective_min_mm, D_effective_max_mm</pre>
            <p>对应含义：time，水位，H<sub>eff</sub>，W<sub>eff</sub>，D<sub>effective,min</sub>，D<sub>effective,max</sub>。</p>

            <h3>反演结果文件</h3>
            <p><code>bedload_inversion_timeseries.csv</code>：</p>
            <pre>time, water_level_m, q_b_kg_m_s, Q_b_kg_s</pre>
            <p>对应含义：time，水位，q<sub>b</sub>，Q<sub>b</sub>。</p>

            <h3>批量反演输出</h3>
            <p>长时序结果保存在一张 <code>bedload_inversion_timeseries.csv</code> 中；每日水位–推移质通量双轴图保存到 <code>daily_figures</code> 文件夹。</p>
            """
        )
        layout.addWidget(text_box)


class PlaceholderPage(QWidget):
    """其他功能页面的占位页面。"""

    def __init__(self, title, description):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: gray; font-size: 14px;")

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addStretch()


class MainWindow(QMainWindow):
    """软件主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于地震信号的推移质通量自动反演与可视化分析软件 V1.0")
        self.resize(1350, 900)
        self.build_ui()
        self.apply_style()

    def build_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(220)

        page_names = [
            "项目管理",
            "泥沙粒径",
            "水力参数",
            "地震信号处理",
            "反演计算",
            "批量反演",
            "文件格式说明",
        ]
        for name in page_names:
            item = QListWidgetItem(name)
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.nav_list.addItem(item)

        self.stack = QStackedWidget()
        self.project_page = ProjectPage()
        self.sediment_grain_page = SedimentGrainPage()
        self.hydraulic_geometry_page = HydraulicGeometryPage()
        self.seismic_processing_page = SeismicSignalProcessingPage()
        self.seismic_page = self.seismic_processing_page.preprocess_page
        self.time_frequency_page = self.seismic_processing_page.time_frequency_page
        self.inversion_page = BedloadInversionPage()
        self.batch_page = AutoRunPage()
        self.file_format_page = FileFormatInfoPage()

        self.seismic_processing_page.processed_sac_saved.connect(self.time_frequency_page.set_input_from_processed_sac)
        self.time_frequency_page.energy_csv_saved.connect(self.inversion_page.set_energy_file)
        self.sediment_grain_page.sediment_saved.connect(self.inversion_page.set_grain_file)
        self.hydraulic_geometry_page.hydraulic_saved.connect(self.inversion_page.set_hydraulic_file)

        self.inversion_page.set_context_pages(
            project_page=self.project_page,
            time_frequency_page=self.time_frequency_page,
            sediment_page=self.sediment_grain_page,
            hydraulic_page=self.hydraulic_geometry_page,
        )
        self.project_page.set_context_pages(
            sediment_page=self.sediment_grain_page,
            hydraulic_page=self.hydraulic_geometry_page,
            inversion_page=self.inversion_page,
        )
        self.batch_page.set_context_pages(
            project_page=self.project_page,
            station_page=None,
            seismic_page=self.seismic_page,
            time_frequency_page=self.time_frequency_page,
            sediment_page=self.sediment_grain_page,
            hydraulic_page=self.hydraulic_geometry_page,
            inversion_page=self.inversion_page,
        )

        for page in [
            self.project_page,
            self.sediment_grain_page,
            self.hydraulic_geometry_page,
            self.seismic_processing_page,
            self.inversion_page,
            self.batch_page,
            self.file_format_page,
        ]:
            self.stack.addWidget(page)

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)
        main_layout.addWidget(self.nav_list)
        main_layout.addWidget(self.stack)
        self.setCentralWidget(central_widget)

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f6f8; }
            QListWidget { background-color: #1f2937; color: white; border: none; font-size: 15px; padding-top: 10px; }
            QListWidget::item { height: 46px; padding-left: 18px; }
            QListWidget::item:selected { background-color: #2563eb; color: white; }
            QLabel { color: #111827; }
            QLineEdit, QTextEdit, QComboBox, QTableWidget { border: 1px solid #d1d5db; border-radius: 5px; padding: 6px; font-size: 13px; background-color: white; }
            QPushButton { background-color: #2563eb; color: white; border: none; border-radius: 5px; padding: 7px 12px; }
            QPushButton:hover { background-color: #1d4ed8; }
            QCheckBox { font-size: 13px; }
            QGroupBox { font-weight: bold; border: 1px solid #d1d5db; border-radius: 6px; margin-top: 8px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)


def global_exception_hook(exc_type, exc_value, exc_traceback):
    """兜底捕获未被按钮函数捕获的异常，并在软件界面弹窗显示。"""
    detail = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    message = (
        f"程序运行时发生未捕获错误。\n\n"
        f"错误类型：{exc_type.__name__}\n"
        f"错误原因：{exc_value}\n\n"
        "完整错误信息已输出到控制台。"
    )
    print(detail, file=sys.stderr)
    try:
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "程序运行错误", message)
    except Exception:
        pass


def main():
    sys.excepthook = global_exception_hook
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
