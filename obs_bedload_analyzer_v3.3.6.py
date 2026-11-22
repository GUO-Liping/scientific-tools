# -*- coding: utf-8 -*-
"""
OBS Bedload Analyzer v3.3.6｜PyQt6。

本文件合并核心算法、波形延迟读取、默认配置与 PyQt6 图形界面。
界面以程序1的 PyQt6 专业多标签页版本为基础，融合程序2的断面水力与实测粒径功能，并移除项目管理页。

算法依据：
1. Tsai et al. (2012), DOI: 10.1029/2011GL050255；
2. Luong et al. (2024), DOI: 10.1029/2024JF007761；
3. Luong et al. (2026), DOI: 10.1029/2025WR040371；
4. seismic_bedload v0.2.0，commit
   e666876e5750d0f27ef6a3f94b638ef7eb9e16d7。

核心算法采用 UTC 时间体系、统一量纲与参数校验；
新增实测级配、断面计算、H_eff/W_eff、断档补零和反演诊断；界面按六步工作流组织。
v3.2.1 分开处理水深、断面水力半径和床面粗糙度，以观测有效频点反演，
并明确记录频带统计量、模型假设和 Luong 2026 场地标定参数。
v3.2.2 新增按原始 PSD 时间或 1–60 分钟 UTC 时间窗反演。
v3.2.5 新增按小时并行、会话内谱缓存和二进制 SAC 按需读取。
v3.2.6 增加并行设置、结果空值补零和预览图合并导出。
v3.2.7 更新站点标识、默认设置、公式排版和结果导出格式。
v3.2.8 引入逐级抗混叠降采样、增量预览与科研图导出。
v3.2.9 简化处理选项和反演结果图，累计推移质输沙量以吨显示与导出。
v3.3.0 使用 PyQtGraph 交互绘图和 Qt 图件导出，优化波形读取、并行处理与参数取值。
v3.3.1 绘图预览将缺测值按零显示，并统一加粗曲线和坐标轴。
v3.3.2 以 Matplotlib 绘制 PSD 与多尺度反演结果，等尺寸排列子图并导出三种尺度。
v3.3.3 水文和反演采用等尺寸四图布局，导出并列的小时、日、月数据。
v3.3.4 支持 PSD 时频数据导入导出和采样率冲突处理。
v3.3.5 合并波形文件问题提示，支持 OBS 四分量导入和独立图件导出。
v3.3.6 水文预览添加河宽，导出结果时直接选择保存目录。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.special import gamma as gamma_function
from scipy.stats import lognorm


APP_VERSION = "3.3.6"
UPSTREAM_VERSION = "0.2.0"
UPSTREAM_COMMIT = "e666876e5750d0f27ef6a3f94b638ef7eb9e16d7"
EPS_PSD = 1.0e-40
MODEL_TYPES = ("Tsai2012", "Luong2024", "Luong2026", "IMHE2026")
MULTIMODE_TYPES = {"Luong2024", "IMHE2026"}


@dataclass
class SedimentParams:
    """泥沙与流体参数，默认值与官方仓库 v0.2.0 一致。"""

    rho_s: float = 2700.0
    rho_f: float = 1000.0
    g: float = 9.81
    corey_shape_factor: float = 0.8
    powers_roundness: float = 3.5
    kinematic_viscosity: float = 1.0e-6
    rise_fall_factor: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        """校验界面运行时传入的流体和泥沙参数。"""

        for name in (
            "rho_s",
            "rho_f",
            "g",
            "corey_shape_factor",
            "powers_roundness",
            "kinematic_viscosity",
            "rise_fall_factor",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"SedimentParams.{name} 必须为有限正数。")
        if self.rho_s <= self.rho_f:
            raise ValueError("颗粒密度 rho_s 必须大于流体密度 rho_f。")


@dataclass
class SeismicParams:
    """瑞利波传播与冲量方向参数。"""

    v0: float = 2206.0
    z0: float = 1000.0
    f0: float = 1.0
    a: float = 0.272
    Q0: float = 20.0
    eta: float = 0.0
    vc0_override: float | None = None
    zeta_override: float | None = None
    restitution: float = 0.0
    fx: float = 0.146
    fy: float = 0.146
    fz: float = 0.539
    Nzz: float = 0.352
    # None 仅使用垂向项。0 rad 是开启水平耦合时的有效方位。
    impact_azimuth_rad: float | None = None

    def __post_init__(self) -> None:
        """校验传播、衰减和冲击方向参数。"""

        for name in ("v0", "z0", "f0", "Q0"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"SeismicParams.{name} 必须为有限正数。")
        for name in ("fx", "fy", "fz", "Nzz"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"SeismicParams.{name} 必须为有限非负数。")
        if not 0.0 <= float(self.a) < 1.0:
            raise ValueError("SeismicParams.a 必须满足 0 <= a < 1。")
        if not 0.0 <= float(self.restitution) <= 1.0:
            raise ValueError("恢复系数 restitution 必须在 [0,1] 内。")
        if self.impact_azimuth_rad is not None and not np.isfinite(self.impact_azimuth_rad):
            raise ValueError("冲击方位角必须为有限数或 None。")
        if self.vc0_override is not None and self.vc0_override <= 0.0:
            raise ValueError("vc0_override 必须为正数或 None。")

    @property
    def zeta(self) -> float:
        """频散指数 ζ=a/(1-a)，或返回现场主动源试验覆盖值。"""

        if self.zeta_override is not None:
            return float(self.zeta_override)
        return self.a / (1.0 - self.a)

    @property
    def vc0(self) -> float:
        """参考频率 f0 处的相速度，或返回现场主动源试验覆盖值。"""

        if self.vc0_override is not None:
            return float(self.vc0_override)
        numerator = self.v0 * gamma_function(1.0 + self.a)
        denominator = (2.0 * np.pi * self.z0 * self.f0) ** self.a
        return float((numerator / denominator) ** (1.0 / (1.0 - self.a)))


@dataclass
class PSDMetadata:
    """Luong 型连续 Welch PSD 的时间与频率诊断信息。"""

    sampling_rate_hz: float
    nperseg: int
    noverlap: int
    hop_seconds: float
    segments_per_block: int
    nominal_block_seconds: float
    blocks_per_hour: float
    aggregation: str


def load_json_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置文件并执行版本、结构和物理范围校验。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("config_version") == "3.2.0":
        # 旧文件显式保存的共享截断选项延续到两个模型。
        sediment = config.get("sediment", {})
        if "clip_tau_c" in sediment:
            sediment.setdefault("clip_tau_c_tsai", sediment["clip_tau_c"])
            sediment.setdefault("clip_tau_c_multimode", sediment["clip_tau_c"])
    if config.get("config_version") in {"3.2.0", "3.2.1", "3.2.2", "3.2.3", "3.2.4", "3.2.5", "3.2.6", "3.2.7", "3.2.8", "3.2.9", "3.3.0", "3.3.1", "3.3.2", "3.3.3", "3.3.4", "3.3.5"}:
        config["config_version"] = APP_VERSION
    if config.get("config_version") != APP_VERSION:
        raise ValueError(
            f"配置版本 {config.get('config_version')!r} 与程序版本 "
            f"{APP_VERSION!r} 不一致。"
        )
    for section in (
        "psd",
        "inversion",
        "site",
        "sediment",
        "seismic",
        "multimode",
        "empirical2026",
        "waveform",
        "performance",
    ):
        if section not in config:
            raise ValueError(f"配置缺少必需分区：{section}")
    validate_config(config)
    return config


def _require_positive(name: str, value: float) -> float:
    """将输入转换为正浮点数，否则给出含参数名的错误。"""

    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} 必须为有限正数，当前值为 {value!r}。")
    return number


def validate_config(config: Mapping[str, Any]) -> None:
    """在启动计算前检查工程参数的内部一致性和基本物理范围。"""

    psd = config["psd"]
    nperseg = int(psd["nperseg"])
    if nperseg < 8 or nperseg & (nperseg - 1):
        raise ValueError("psd.nperseg 必须是至少为 8 的 2 的整数次幂。")
    overlap = float(psd["overlap_fraction"])
    if not 0.0 <= overlap < 1.0:
        raise ValueError("psd.overlap_fraction 必须满足 0 <= overlap < 1。")
    if str(psd["aggregation"]).lower() not in {"median", "mean"}:
        raise ValueError("psd.aggregation 仅支持 median 或 mean。")
    psd_low = _require_positive("psd.display_fmin_hz", psd["display_fmin_hz"])
    psd_high = _require_positive("psd.display_fmax_hz", psd["display_fmax_hz"])
    if psd_low >= psd_high:
        raise ValueError("PSD 显示频带下限必须小于上限。")

    inversion = config["inversion"]
    inv_low = _require_positive(
        "inversion.f_inv_min_hz", inversion["f_inv_min_hz"]
    )
    inv_high = _require_positive(
        "inversion.f_inv_max_hz", inversion["f_inv_max_hz"]
    )
    if inv_low >= inv_high:
        raise ValueError("反演频带下限必须小于上限。")
    if int(inversion["n_frequency_bins"]) < 2:
        raise ValueError("inversion.n_frequency_bins 至少为 2。")
    if inversion.get("band_statistic", "median") not in {"median", "mean"}:
        raise ValueError("反演频带统计量须为 median 或 mean。")
    if inversion.get("frequency_grid", "native") not in {"native", "interpolated"}:
        raise ValueError("反演频率轴须为 native 或 interpolated。")
    _inversion_resolution_minutes(inversion.get("resolution_minutes", 0))
    if inversion["model_type"] not in MODEL_TYPES:
        raise ValueError(
            "inversion.model_type 仅支持 " + "、".join(MODEL_TYPES) + "。"
        )

    site = config["site"]
    _require_positive("site.channel_width_m", site["channel_width_m"])
    slope = _require_positive("site.slope", site["slope"])
    if slope >= 1.0:
        raise ValueError("site.slope 应为无量纲坡降 S=tan(theta)，且应小于 1。")
    _require_positive("site.source_distance_m", site["source_distance_m"])

    sediment = config["sediment"]
    d50 = _require_positive("sediment.D50_m", sediment["D50_m"])
    if sediment["D_min_m"] is not None:
        d_min = _require_positive("sediment.D_min_m", sediment["D_min_m"])
        if d_min >= d50:
            raise ValueError("sediment.D_min_m 必须小于 D50_m。")
    if sediment["D_max_m"] is not None:
        d_max = _require_positive("sediment.D_max_m", sediment["D_max_m"])
        if d_max <= d50:
            raise ValueError("sediment.D_max_m 必须大于 D50_m。")
    _require_positive("sediment.grain_log_std", sediment["grain_log_std"])
    if int(sediment["grain_points"]) < 8:
        raise ValueError("sediment.grain_points 至少为 8。")
    rho_s = _require_positive("sediment.rho_s_kg_m3", sediment["rho_s_kg_m3"])
    rho_f = _require_positive("sediment.rho_f_kg_m3", sediment["rho_f_kg_m3"])
    if rho_s <= rho_f:
        raise ValueError("沉积物密度 rho_s 必须大于流体密度 rho_f。")
    _require_positive("sediment.tau_c50", sediment["tau_c50"])
    if sediment.get("roughness_m") is not None:
        _require_positive("河床粗糙度 ks", sediment["roughness_m"])
    if sediment.get("tsai_transport_stage", "excess") not in {"excess", "published"}:
        raise ValueError("Tsai 输移阶段须为 excess 或 published。")
    for key in ("clip_tau_c_tsai", "clip_tau_c_multimode"):
        if key in sediment and not isinstance(sediment[key], bool):
            raise ValueError(f"sediment.{key} 必须为布尔值。")

    seismic = config["seismic"]
    for key in ("v0_m_s", "z0_m", "f0_hz", "Q0"):
        _require_positive(f"seismic.{key}", seismic[key])
    if not 0.0 <= float(seismic["a"]) < 1.0:
        raise ValueError("seismic.a 必须满足 0 <= a < 1。")
    if not 0.0 <= float(seismic["restitution_gamma"]) <= 1.0:
        raise ValueError("恢复系数 restitution_gamma 必须在 [0, 1] 内。")
    if seismic["vc0_override_m_s"] is not None:
        _require_positive(
            "seismic.vc0_override_m_s", seismic["vc0_override_m_s"]
        )

    multimode = config["multimode"]
    hop_min = _require_positive("multimode.hop_t_min_s", multimode["hop_t_min_s"])
    hop_max = _require_positive("multimode.hop_t_max_s", multimode["hop_t_max_s"])
    if hop_min >= hop_max:
        raise ValueError("multimode.hop_t_min_s 必须小于 hop_t_max_s。")
    _require_positive(
        "multimode.hop_log_median_s", multimode["hop_log_median_s"]
    )
    _require_positive("multimode.hop_log_sigma", multimode["hop_log_sigma"])
    if int(multimode["hop_points"]) < 8:
        raise ValueError("multimode.hop_points 至少为 8。")

    empirical = config["empirical2026"]
    for key in ("k", "m", "n"):
        _require_positive(f"empirical2026.{key}", empirical[key])
    for key in ("A_db", "B_db"):
        value = float(empirical[key])
        if not np.isfinite(value):
            raise ValueError(f"empirical2026.{key} 必须为有限数。")
    tau_cr = float(empirical["tau_cr"])
    if not np.isfinite(tau_cr) or tau_cr < 0.0:
        raise ValueError("empirical2026.tau_cr 必须为有限非负数。")
    _require_positive(
        "empirical2026.hydraulic_radius_depth_ratio",
        empirical["hydraulic_radius_depth_ratio"],
    )

    waveform = config.get("waveform", {})
    pre_filt = [float(value) for value in waveform["pre_filt_hz"]]
    if len(pre_filt) != 4 or not (
        0.0 < pre_filt[0] < pre_filt[1] < pre_filt[2] < pre_filt[3]
    ):
        raise ValueError("waveform.pre_filt_hz 必须是四个严格递增的正频率。")
    maximum_gap = float(waveform["maximum_interpolated_gap_s"])
    if not np.isfinite(maximum_gap) or maximum_gap < 0.0:
        raise ValueError("maximum_interpolated_gap_s 必须为有限非负数。")
    selection_hours = float(waveform["default_selection_hours"])
    if not np.isfinite(selection_hours) or selection_hours <= 0.0:
        raise ValueError("waveform.default_selection_hours 必须为有限正数。")

    performance = config["performance"]
    if int(performance["psd_chunk_blocks"]) < 1:
        raise ValueError("performance.psd_chunk_blocks 至少为 1。")
    if int(performance["plot_max_points"]) < 1000:
        raise ValueError("performance.plot_max_points 至少为 1000。")


def normalize_pdf(x: np.ndarray, density: np.ndarray, name: str) -> np.ndarray:
    """使用梯形积分归一化概率密度，阻止零积分或非有限密度进入模型。"""

    grid = np.asarray(x, dtype=float)
    pdf = np.asarray(density, dtype=float)
    if grid.ndim != 1 or pdf.ndim != 1 or grid.size != pdf.size:
        raise ValueError(f"{name} 的网格和密度必须是一维等长数组。")
    if grid.size < 2 or np.any(~np.isfinite(grid)) or np.any(grid <= 0.0) or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} 网格至少含两个有限、严格递增的正数。")
    if np.any(~np.isfinite(pdf)) or np.any(pdf < 0.0):
        raise ValueError(f"{name} 密度必须为有限非负数。")
    integral = float(np.trapezoid(pdf, grid))
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError(f"{name} 的概率积分必须大于 0。")
    return pdf / integral


def log_raised_cosine_pdf(D: np.ndarray, median: float, support_half_width: float) -> np.ndarray:
    """计算对数粒径空间的 raised-cosine 密度（尚未除以 D）。"""

    grains = np.asarray(D, dtype=float)
    d50 = _require_positive("D50", median)
    width = _require_positive("对数 raised-cosine 半宽 s", support_half_width)
    if np.any(grains <= 0.0):
        raise ValueError("粒径 D 必须全部大于 0。")
    delta = np.log(grains) - np.log(d50)
    return np.where(
        np.abs(delta) < width,
        0.5 / width * (1.0 + np.cos(np.pi * delta / width)),
        0.0,
    )


def build_grain_distribution(
    D50: float,
    log_std: float,
    n_points: int = 80,
    D_min: float | None = None,
    D_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    按官方仓库方法生成粒径网格及其物理粒径空间 PDF。

    raised-cosine 先在 ln(D) 空间定义，再除以 D 完成变量变换。
    可用 D_min/D_max 按现场筛分范围截断，截断后重新归一化。
    """

    d50 = _require_positive("D50", D50)
    sigma = _require_positive("粒径对数标准差", log_std)
    count = int(n_points)
    if count < 8:
        raise ValueError("粒径积分点数至少为 8。")
    support = sigma / np.sqrt(1.0 / 3.0 - 2.0 / np.pi**2)
    lower = float(np.exp(np.log(d50) - support))
    upper = float(np.exp(np.log(d50) + support))
    if D_min is not None:
        lower = max(lower, _require_positive("D_min", D_min))
    if D_max is not None:
        upper = min(upper, _require_positive("D_max", D_max))
    if lower >= upper:
        raise ValueError(
            f"粒径截断范围无效：D_min={lower:.6g} m，D_max={upper:.6g} m。"
        )
    grains = np.linspace(lower, upper, count)
    density = log_raised_cosine_pdf(grains, d50, support) / grains
    return grains, normalize_pdf(grains, density, "粒径 PDF")


def build_hop_time_distribution(
    t_min: float = 0.01,
    t_max: float = 0.50,
    log_median: float = 0.08,
    log_sigma: float = 1.0,
    n_points: int = 70,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 Luong 2024 使用的截断对数正态冲击间隔分布。"""

    lower = _require_positive("t_min", t_min)
    upper = _require_positive("t_max", t_max)
    median = _require_positive("跃移时间对数正态中位数", log_median)
    sigma = _require_positive("跃移时间对数标准差", log_sigma)
    count = int(n_points)
    if lower >= upper:
        raise ValueError("t_min 必须小于 t_max。")
    if count < 8:
        raise ValueError("跃移时间积分点数至少为 8。")
    times = np.linspace(lower, upper, count)
    distribution = lognorm(s=sigma, scale=median)
    norm = distribution.cdf(upper) - distribution.cdf(lower)
    raw = distribution.pdf(times) / norm
    return times, normalize_pdf(times, raw, "跃移时间 PDF")


def estimate_critical_shields(slope: float, method: str = "gimbert") -> float:
    """根据坡降估计 D50 临界 Shields 参数。"""

    S = _require_positive("无量纲坡降 S", slope)
    if method == "gimbert":
        dummy = 0.407 * np.log(142.0 * S)
        return float(
            np.exp(
                2.59e-2 * dummy**4
                + 8.94e-2 * dummy**3
                + 0.142 * dummy**2
                + 0.41 * dummy
                - 3.14
            )
        )
    if method == "lamb":
        return float(0.15 * S**0.25)
    raise ValueError(f"未知临界 Shields 估算方法：{method}")


class _BaseBedloadModel:
    """两类物理模型共用的传播项和输入检查。"""

    def __init__(
        self,
        sediment_params: SedimentParams | None = None,
        seismic_params: SeismicParams | None = None,
    ) -> None:
        self.sediment = sediment_params or SedimentParams()
        self.seismic = seismic_params or SeismicParams()
        self._spectral_cache: dict[tuple[bytes, float], np.ndarray] = {}

    def hydraulic_scales(self, H, slope, D50, hydraulic_radius_m, ks):
        """几何水深控制相对淹没度，水力半径控制床面剪切。"""
        radius = H if hydraulic_radius_m is None else _require_positive(
            "水力半径 Rh", hydraulic_radius_m
        )
        roughness = 3.0 * D50 if ks is None else _require_positive("粗糙度 ks", ks)
        u_star = np.sqrt(self.sediment.g * radius * slope)
        upper_velocity = 8.1 * u_star * (H / roughness) ** (1.0 / 6.0)
        return u_star, roughness, upper_velocity

    def seismic_spectral_factor(
        self, frequency_hz: np.ndarray, source_distance_m: float
    ) -> np.ndarray:
        """计算 Tsai 2012 式(5)、(9)的瑞利波频散和衰减项。"""

        f = np.asarray(frequency_hz, dtype=float)
        r0 = _require_positive("台站距离 r0", source_distance_m)
        if f.ndim != 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0.0):
            raise ValueError("模型频率必须是一维正数数组。")
        cache_key = (np.ascontiguousarray(f).tobytes(), r0)
        cached = self._spectral_cache.get(cache_key)
        if cached is not None:
            return cached

        sp = self.seismic
        vc = sp.vc0 * (f / sp.f0) ** (-sp.zeta)
        vu = vc / (1.0 + sp.zeta)
        beta = (
            2.0
            * np.pi
            * r0
            * (1.0 + sp.zeta)
            * f ** (1.0 + sp.zeta - sp.eta)
            / (sp.vc0 * sp.Q0 * sp.f0 ** (sp.zeta - sp.eta))
        )
        chi = (
            2.0 * np.log1p(1.0 / beta) * np.exp(-2.0 * beta)
            + (1.0 - np.exp(-beta))
            * np.exp(-beta)
            * np.sqrt(2.0 * np.pi / beta)
        )
        result = f**3 / vc**3 / vu**2 * chi
        # 原始频点模式允许各时刻缺失频点不同，限制缓存大小。
        if len(self._spectral_cache) < 32:
            self._spectral_cache[cache_key] = result
        return result

    def critical_shields_by_size(
        self,
        D: np.ndarray,
        D50: float,
        tau_c50: float | None,
        slope: float,
        clip: bool,
    ) -> np.ndarray:
        """按 Parker (1990) 粒径指数 -0.9 扩展临界 Shields 参数。"""

        base = (
            estimate_critical_shields(slope)
            if tau_c50 is None
            else _require_positive("tau_c50", tau_c50)
        )
        critical = base * (np.asarray(D, dtype=float) / D50) ** (-0.9)
        if clip:
            critical = np.clip(critical, 0.03, 0.06)
        return critical


class SaltationModel(_BaseBedloadModel):
    """Tsai et al. (2012) 跃移模式物理模型。"""

    def estimate_drag_coefficient(self, D: np.ndarray) -> np.ndarray:
        """使用 Dietrich (1982) 经验式计算天然颗粒阻力系数。"""

        sd = self.sediment
        grains = np.asarray(D, dtype=float)
        R = (sd.rho_s - sd.rho_f) / sd.rho_f
        d_star = np.log10(R * sd.g * grains**3 / sd.kinematic_viscosity**2)
        r1 = (
            -3.76715
            + 1.92944 * d_star
            - 0.09815 * d_star**2
            - 0.00575 * d_star**3
            + 0.00056 * d_star**4
        )
        csf = sd.corey_shape_factor
        r2 = (
            np.log10(1.0 - (1.0 - csf) / 0.85)
            - (1.0 - csf) ** 2.3 * np.tanh(d_star - 4.6)
            + 0.3 * (0.5 - csf) * (1.0 - csf) ** 2 * (d_star - 4.6)
        )
        r3 = (
            0.65 - csf / 2.83 * np.tanh(d_star - 4.6)
        ) ** (1.0 + (3.5 - sd.powers_roundness) / 2.5)
        w_star = r3 * 10.0 ** (r1 + r2)
        settling = (w_star * R * sd.g * sd.kinematic_viscosity) ** (1.0 / 3.0)
        return (4.0 / 3.0) * R * sd.g * grains / settling**2

    def forward_psd(
        self,
        f: np.ndarray,
        D: np.ndarray,
        H: float,
        W: float,
        slope: float,
        r0: float,
        qb: float,
        D50: float,
        pdf_D: np.ndarray,
        tau_c50: float | None = 0.045,
        clip_tau_c: bool = False,
        *,
        hydraulic_radius_m: float | None = None,
        ks: float | None = None,
        transport_stage_form: str = "excess",
    ) -> np.ndarray:
        """计算给定单宽体积通量 qb 对应的地震速度 PSD。"""

        frequency = np.asarray(f, dtype=float)
        grains = np.asarray(D, dtype=float)
        density = normalize_pdf(grains, pdf_D, "粒径 PDF")
        depth = _require_positive("水深 H", H)
        width = _require_positive("河宽 W", W)
        S = _require_positive("无量纲坡降 S", slope)
        flux = float(qb)
        if not np.isfinite(flux) or flux < 0.0:
            raise ValueError("qb 必须为有限非负数。")
        d50 = _require_positive("D50", D50)

        sd = self.sediment
        R = (sd.rho_s - sd.rho_f) / sd.rho_f
        u_star, _, flow_velocity = self.hydraulic_scales(
            depth, S, d50, hydraulic_radius_m, ks
        )
        shields = u_star**2 / (R * sd.g * grains)
        critical = self.critical_shields_by_size(
            grains, d50, tau_c50, S, clip_tau_c
        )
        transport_stage = shields / critical
        if transport_stage_form not in {"excess", "published"}:
            raise ValueError("Tsai 输移阶段须为 excess 或 published。")
        moving = transport_stage > 1.0
        stage = np.where(
            moving,
            transport_stage - 1.0 if transport_stage_form == "excess" else transport_stage,
            0.0,
        )
        bedload_velocity = (
            1.56 * np.sqrt(R * sd.g * grains) * stage**0.56
        )
        layer_height = 1.44 * grains * stage**0.50
        bedload_velocity = np.clip(bedload_velocity, 0.0, flow_velocity)
        layer_height = np.clip(layer_height, 0.0, depth)

        volume = np.pi * grains**3 / 6.0
        mass = sd.rho_s * volume
        drag = self.estimate_drag_coefficient(grains)
        terminal_velocity = np.sqrt(4.0 * R * sd.g * grains / (3.0 * drag))
        angle = np.arctan(S)
        hb_hat = (
            3.0
            * drag
            * sd.rho_f
            * layer_height
            / (2.0 * sd.rho_s * grains * np.cos(angle))
        )
        impact_velocity = (
            terminal_velocity
            * np.cos(angle)
            * np.sqrt(-np.expm1(-hb_hat))
        )
        denominator = hb_hat + 2.0 * np.log1p(np.sqrt(-np.expm1(-hb_hat)))
        mean_settling = np.divide(
            hb_hat * terminal_velocity * np.cos(angle),
            denominator,
            out=np.zeros_like(hb_hat),
            where=denominator > 0.0,
        )
        rate = np.divide(
            sd.rise_fall_factor * width * flux * mean_settling,
            volume * bedload_velocity * layer_height,
            out=np.zeros_like(volume),
            where=(volume * bedload_velocity * layer_height) > 0.0,
        )

        source = (
            rate
            * np.pi**2
            * mass**2
            * impact_velocity**2
            / sd.rho_s**2
        )
        source = np.where(moving, source, 0.0)
        spectral = self.seismic_spectral_factor(frequency, r0)
        total = np.trapezoid(density * source, grains) * spectral
        return np.maximum(total, 0.0)


class MultimodeModel(_BaseBedloadModel):
    """Luong et al. (2024) 式(12)多输移模式模型。"""

    def _bedload_velocity(
        self,
        D: np.ndarray,
        H: float,
        slope: float,
        D50: float,
        tau_c: np.ndarray,
        hydraulic_radius_m: float | None,
        ks: float | None,
    ) -> np.ndarray:
        """按论文在 H<0.4 m 与较高水深分别选用式(14)和式(13)。"""

        sd = self.sediment
        R = (sd.rho_s - sd.rho_f) / sd.rho_f
        u_star, roughness, upper_velocity = self.hydraulic_scales(
            H, slope, D50, hydraulic_radius_m, ks
        )
        shields = u_star**2 / (R * sd.g * D)
        if H < 0.4:
            velocity = (
                30.5
                * shields
                * np.sqrt(R * sd.g * D)
                * (D / roughness) ** 0.583
            )
        else:
            excess_stage = np.maximum(shields / tau_c - 1.0, 0.0)
            velocity = (
                1.56
                * np.sqrt(R * sd.g * D)
                * excess_stage**0.56
            )
        return np.clip(velocity, 0.0, upper_velocity)

    def forward_psd(
        self,
        f: np.ndarray,
        D: np.ndarray,
        H: float,
        W: float,
        slope: float,
        r0: float,
        qb: float,
        D50: float,
        pdf_D: np.ndarray,
        hop_times: np.ndarray,
        pdf_t: np.ndarray,
        tau_c50: float | None = 0.045,
        clip_tau_c: bool = True,
        *,
        hydraulic_radius_m: float | None = None,
        ks: float | None = None,
    ) -> np.ndarray:
        """数值积分 Luong 2024 式(12)，输出地震速度 PSD。"""

        frequency = np.asarray(f, dtype=float)
        grains = np.asarray(D, dtype=float)
        grain_pdf = normalize_pdf(grains, pdf_D, "粒径 PDF")
        times = np.asarray(hop_times, dtype=float)
        time_pdf = normalize_pdf(times, pdf_t, "跃移时间 PDF")
        depth = _require_positive("水深 H", H)
        width = _require_positive("河宽 W", W)
        S = _require_positive("无量纲坡降 S", slope)
        d50 = _require_positive("D50", D50)
        flux = float(qb)
        if not np.isfinite(flux) or flux < 0.0:
            raise ValueError("qb 必须为有限非负数。")

        critical = self.critical_shields_by_size(
            grains, d50, tau_c50, S, clip_tau_c
        )
        velocity = self._bedload_velocity(
            grains, depth, S, d50, critical, hydraulic_radius_m, ks
        )
        volume = np.pi * grains**3 / 6.0
        # ∫p(t)/t dt 可先计算，数学上等价于仓库中的二维梯形积分。
        inverse_hop_time = float(np.trapezoid(time_pdf / times, times))
        source = self._impact_source(volume, velocity, inverse_hop_time, width, flux)
        spectral = self.seismic_spectral_factor(frequency, r0)
        total = np.trapezoid(grain_pdf * source, grains) * spectral
        return np.maximum(total, 0.0)

    def _impact_coupling(self):
        sp = self.seismic
        azimuth = sp.impact_azimuth_rad
        nxz = 0.8 * np.cos(azimuth) if azimuth is not None else 0.0
        nyz = 0.8 * np.sin(azimuth) if azimuth is not None else 0.0
        return (1.0 + sp.restitution) * (
            abs(sp.fx * nxz) + abs(sp.fy * nyz) + abs(sp.fz * sp.Nzz)
        )

    def _impact_source(self, volume, velocity, inverse_hop_time, width, flux):
        density = self.sediment.rho_s
        mass = density * volume
        rate = np.divide(
            width * flux * inverse_hop_time,
            volume * velocity,
            out=np.zeros_like(volume),
            where=(volume * velocity) > 0.0,
        )

        return rate * mass**2 * np.pi**2 * (self._impact_coupling() * velocity)**2 / (4.0 * density**2)


class IMHEModel2026(MultimodeModel):
    """Local revision of the Luong (2024) physical model."""

    def _bedload_velocity(self, D, H, slope, D50, tau_c, hydraulic_radius_m, ks):
        velocity = super()._bedload_velocity(D, H, slope, D50, tau_c, hydraulic_radius_m, ks)
        relative_density = (self.sediment.rho_s - self.sediment.rho_f) / self.sediment.rho_f
        radius = H if hydraulic_radius_m is None else hydraulic_radius_m
        shields = radius * slope / (relative_density * D)
        # Also enforce incipient motion in the shallow Julien branch.
        return np.where(shields > tau_c, velocity, 0.0)

    def _impact_source(self, volume, velocity, inverse_hop_time, width, flux):
        # In the product of impact rate and squared impulse, the velocity
        # cancels once, and m² /(rho_s² Vp) reduces to Vp. No division by Ub.
        scale = (np.pi**2 / 4.0) * self._impact_coupling()**2 * inverse_hop_time
        return scale * width * flux * volume * velocity


class SeismicHydraulicModel:
    """Luong et al. (2026) 式(13)地震—水力经验反演模型。"""

    def __init__(
        self, sediment_params: SedimentParams | None = None
    ) -> None:
        self.sediment = sediment_params or SedimentParams()

    @staticmethod
    def calibrate_reference_psd(observed_psd_linear, shields) -> tuple[float, float]:
        """在独立标定数据上拟合 A、B，供正式反演时固定使用。"""
        observed = np.asarray(observed_psd_linear, dtype=float)
        tau = np.asarray(shields, dtype=float)
        if observed.ndim != 1 or observed.shape != tau.shape:
            raise ValueError("标定 PSD 和 Shields 数据须为一维等长数组。")
        valid = np.isfinite(observed) & (observed > 0.0) & np.isfinite(tau) & (tau > 0.0)
        x = np.log10(tau[valid])
        if np.unique(x).size < 2:
            raise ValueError("至少需要两个不同的有效 Shields 值标定参考 PSD。")
        A, B = np.polyfit(x, 10.0 * np.log10(observed[valid]), 1)
        return float(A), float(B)

    @staticmethod
    def reference_psd_db(
        shields: np.ndarray,
        A_db: float = 29.0,
        B_db: float = -127.0,
    ) -> np.ndarray:
        """按论文式(14)计算参考 PSD（dB）。"""

        tau = np.asarray(shields, dtype=float)
        A = float(A_db)
        B = float(B_db)
        if not np.isfinite(A) or not np.isfinite(B):
            raise ValueError("Luong 2026 的 A、B 必须为有限数。")
        result = np.full(tau.shape, np.nan, dtype=float)
        valid = np.isfinite(tau) & (tau > 0.0)
        result[valid] = A * np.log10(tau[valid]) + B
        return result

    def inverse_bedload(
        self,
        observed_band_psd_linear: np.ndarray,
        depth_m: np.ndarray,
        slope: float,
        D50: float,
        tau_cr: float = 0.045,
        k: float = 5.2,
        m: float = 0.25,
        n: float = 1.36,
        A_db: float = 29.0,
        B_db: float = -127.0,
        hydraulic_radius_depth_ratio: float = 1.0,
        *,
        hydraulic_radius_m: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        由频带 PSD 与水深计算 Luong 2026 单宽体积通量。

        式(13)中的 ``P/P_re`` 是线性功率比。观测 PSD 转换为 dB 后，
        与式(14)的 ``P_re,dB`` 作差，再以 ``10**(ΔdB/10)`` 恢复
        线性比值。宽浅河道默认采用 ``R_h≈H``。
        """

        observed = np.asarray(observed_band_psd_linear, dtype=float)
        depth = np.asarray(depth_m, dtype=float)
        if observed.shape != depth.shape:
            raise ValueError("观测 PSD 与水深数组形状必须一致。")
        S = _require_positive("无量纲坡降 S", slope)
        d50 = _require_positive("D50", D50)
        ratio_rh = _require_positive(
            "水力半径/水深比", hydraulic_radius_depth_ratio
        )
        scale = _require_positive("Luong 2026 系数 k", k)
        seismic_exponent = _require_positive("Luong 2026 系数 m", m)
        shear_exponent = _require_positive("Luong 2026 系数 n", n)
        critical = float(tau_cr)
        if not np.isfinite(critical) or critical < 0.0:
            raise ValueError("Luong 2026 临界 Shields 参数必须为有限非负数。")

        sd = self.sediment
        relative_density = (sd.rho_s - sd.rho_f) / sd.rho_f
        hydraulic_radius = depth * ratio_rh
        if hydraulic_radius_m is not None:
            measured_radius = np.asarray(hydraulic_radius_m, dtype=float)
            if measured_radius.shape != depth.shape:
                raise ValueError("断面水力半径与水深数组形状必须一致。")
            use_measured = np.isfinite(measured_radius) & (measured_radius > 0.0)
            hydraulic_radius = np.where(use_measured, measured_radius, hydraulic_radius)
        shields = hydraulic_radius * S / (relative_density * d50)
        reference_db = self.reference_psd_db(shields, A_db, B_db)
        observed_db = 10.0 * np.log10(np.maximum(observed, EPS_PSD))
        power_ratio = 10.0 ** ((observed_db - reference_db) / 10.0)
        excess = np.maximum(shields - critical, 0.0)
        dimensionless_flux = (
            scale * power_ratio**seismic_exponent * excess**shear_exponent
        )
        conversion = np.sqrt(relative_density * sd.g * d50**3)
        volumetric_flux = dimensionless_flux * conversion
        valid = (
            np.isfinite(observed)
            & (observed > 0.0)
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(reference_db)
        )
        for values in (
            shields,
            reference_db,
            power_ratio,
            dimensionless_flux,
            volumetric_flux,
        ):
            values[~valid] = np.nan
        return {
            "shields": shields,
            "reference_psd_db": reference_db,
            "power_ratio": power_ratio,
            "dimensionless_flux": dimensionless_flux,
            "qb_vol_m2_s": volumetric_flux,
            "shear_hydraulic_radius_m": hydraulic_radius,
        }


def compute_luong_minute_psd(
    data: np.ndarray,
    sampling_rate_hz: float,
    start_time_utc: Any,
    nperseg: int = 2**14,
    overlap_fraction: float = 0.5,
    fmin: float = 1.0,
    fmax: float = 40.0,
    aggregation: str = "median",
    chunk_blocks: int = 256,
) -> tuple[pd.DataFrame, PSDMetadata]:
    """
    按 Luong 2024 数据流程计算连续 Welch PSD。

    先以 nperseg 和 50% 重叠生成连续 Hann 修正周期图，再将约 1 min
    内的若干谱段在“线性 PSD”空间取中位数。若采样率仅 100 Hz，
    2^14 点窗长达 163.84 s，每个近 1 min 块只能包含一个谱段，
    输出约 43.9 块/h；函数会如实保留该时间分辨率，不伪造 60 点/h。
    """

    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError("PSD 输入波形必须是一维数组。")
    if values.size < 8 or np.any(~np.isfinite(values)):
        raise ValueError("PSD 输入波形过短或含 NaN/Inf。")
    fs = _require_positive("采样率", sampling_rate_hz)
    segment_length = int(nperseg)
    if segment_length < 8:
        raise ValueError("NPERSEG 至少为 8。")
    if segment_length > values.size:
        raise ValueError(
            f"波形仅 {values.size} 点，小于 NPERSEG={segment_length}；"
            "请导入更长波形或减小窗长。"
        )
    overlap = float(overlap_fraction)
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Welch 重叠率必须满足 0 <= overlap < 1。")
    noverlap = int(round(segment_length * overlap))
    hop_points = segment_length - noverlap
    if hop_points <= 0:
        raise ValueError("Welch 步长必须大于 0。")
    low = _require_positive("PSD fmin", fmin)
    high = _require_positive("PSD fmax", fmax)
    nyquist = fs / 2.0
    if not low < high <= nyquist:
        raise ValueError(
            f"PSD 频带必须满足 0 < fmin < fmax <= Nyquist({nyquist:g} Hz)。"
        )
    reducer = aggregation.lower()
    if reducer not in {"median", "mean"}:
        raise ValueError("PSD 聚合方法仅支持 median 或 mean。")

    frequency = np.fft.rfftfreq(segment_length, d=1.0 / fs)
    mask = (frequency >= low) & (frequency <= high)
    selected_frequency = frequency[mask]
    if selected_frequency.size == 0:
        raise ValueError("选定 PSD 频带内没有频率点。")

    hop_seconds = hop_points / fs
    segments_per_block = max(1, int(round(60.0 / hop_seconds)))
    total_segments = 1 + (values.size - segment_length) // hop_points
    block_count = total_segments // segments_per_block
    if block_count < 1:
        raise ValueError("有效 Welch 谱段不足以组成一个输出块。")

    output = np.empty((block_count, selected_frequency.size), dtype=float)
    block_batch = max(1, int(chunk_blocks))
    start = pd.Timestamp(start_time_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    # 每批只保留有限数量的 Welch 谱段，避免长时段一次构造巨型矩阵。
    for block0 in range(0, block_count, block_batch):
        block1 = min(block_count, block0 + block_batch)
        segment0 = block0 * segments_per_block
        segment_count = (block1 - block0) * segments_per_block
        sample0 = segment0 * hop_points
        sample_count = segment_length + (segment_count - 1) * hop_points
        chunk = values[sample0 : sample0 + sample_count]
        f_chunk, _, spectrum = spectrogram(
            chunk,
            fs=fs,
            window="hann",
            nperseg=segment_length,
            noverlap=noverlap,
            nfft=segment_length,
            detrend="constant",
            scaling="density",
            mode="psd",
        )
        if not np.allclose(f_chunk, frequency, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("PSD 分块计算出现不一致的频率轴。")
        selected = spectrum[mask, :segment_count]
        grouped = selected.reshape(
            selected_frequency.size, block1 - block0, segments_per_block
        )
        if reducer == "median":
            reduced = np.nanmedian(grouped, axis=2)
        else:
            reduced = np.nanmean(grouped, axis=2)
        output[block0:block1, :] = reduced.T

    first_segment = np.arange(block_count) * segments_per_block
    mean_segment = first_segment + 0.5 * (segments_per_block - 1)
    center_seconds = (
        mean_segment * hop_points + segment_length / 2.0
    ) / fs
    output_times = start + pd.to_timedelta(center_seconds, unit="s")

    frame = pd.DataFrame(
        output,
        index=pd.DatetimeIndex(output_times, name="time_utc"),
        columns=selected_frequency,
    )
    nominal_seconds = segments_per_block * hop_seconds
    metadata = PSDMetadata(
        sampling_rate_hz=fs,
        nperseg=segment_length,
        noverlap=noverlap,
        hop_seconds=hop_seconds,
        segments_per_block=segments_per_block,
        nominal_block_seconds=nominal_seconds,
        blocks_per_hour=3600.0 / nominal_seconds,
        aggregation=reducer,
    )
    return frame, metadata


def align_depth_to_psd(
    depth_series: pd.Series, psd_times: pd.DatetimeIndex
) -> np.ndarray:
    """在原水深观测范围内插值到 PSD 时间，不进行无依据的两端外推。"""

    if depth_series.empty:
        raise ValueError("水深序列为空。")
    source = depth_series.copy()
    source.index = pd.DatetimeIndex(source.index)
    if source.index.tz is None:
        source.index = source.index.tz_localize("UTC")
    else:
        source.index = source.index.tz_convert("UTC")
    source = source.groupby(level=0).mean().sort_index().astype(float)
    target = pd.DatetimeIndex(psd_times)
    if target.tz is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    union = source.index.union(target).sort_values()
    interpolated = (
        source.reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(target)
    )
    return interpolated.to_numpy(dtype=float)


def minmax_decimate(
    x: np.ndarray, y: np.ndarray, max_points: int = 120_000
) -> tuple[np.ndarray, np.ndarray]:
    """
    以每个时间桶的最小值和最大值降采样，保留尖峰并加快波形绘制。

    该函数生成波形预览包络。预处理、PSD 和反演使用所选分析采样率的波形。
    """

    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("降采样输入必须是一维数组。")
    if x_values.size != y_values.size:
        raise ValueError("降采样的 x、y 长度必须一致。")
    indices = minmax_decimation_indices(y_values, max_points)
    return x_values[indices], y_values[indices]


def minmax_decimation_indices(
    y: np.ndarray, max_points: int = 120_000
) -> np.ndarray:
    """返回保留每个桶极值的有序样本索引，避免先构造完整时间轴。"""

    y_values = np.asarray(y)
    if y_values.ndim != 1:
        raise ValueError("降采样输入必须是一维数组。")
    limit = max(1000, int(max_points))
    if y_values.size <= limit:
        return np.arange(y_values.size, dtype=np.int64)
    bucket_count = max(1, limit // 2)
    edges = np.linspace(0, y_values.size, bucket_count + 1, dtype=np.int64)
    out_indices = np.empty(bucket_count * 2, dtype=np.int64)
    cursor = 0
    for start, end in zip(edges[:-1], edges[1:]):
        segment = y_values[start:end]
        if segment.size == 0:
            continue
        finite = np.isfinite(segment)
        if not np.any(finite):
            indices = (start, start)
        else:
            local = np.flatnonzero(finite)
            local_min = local[int(np.argmin(segment[finite]))]
            local_max = local[int(np.argmax(segment[finite]))]
            indices = tuple(sorted((start + local_min, start + local_max)))
        for index in indices:
            out_indices[cursor] = index
            cursor += 1
    return out_indices[:cursor]


def _dataclass_from_mapping(cls: type, values: Mapping[str, Any]) -> Any:
    """仅将 dataclass 已声明字段从配置映射中取出。"""

    names = cls.__dataclass_fields__.keys()
    return cls(**{name: values[name] for name in names if name in values})


def invert_bedload_timeseries(
    df_psd: pd.DataFrame,
    depth_series: pd.Series,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    """固定宽度入口，使用与 GUI 相同的线性 PSD 反演。"""
    return invert_bedload_timeseries_enhanced(df_psd, depth_series, params)


def integrate_flux_mass(time_index: pd.DatetimeIndex, flux_kg_s: np.ndarray) -> float:
    """
    按实际时间戳用梯形积分计算累积质量。

    只积分相邻且均有效的时间块，并跳过大于典型步长三倍的缺测区间；
    这样既避免固定按 60 s 误计，也不会跨越长数据缺口虚构输沙量。
    """

    times = pd.DatetimeIndex(time_index)
    values = np.asarray(flux_kg_s, dtype=float)
    if len(times) != values.size:
        raise ValueError("时间戳数量与通量数量不一致。")
    if values.size < 2:
        return 0.0
    seconds = times.asi8.astype(float) / 1.0e9
    delta = np.diff(seconds)
    positive_delta = delta[np.isfinite(delta) & (delta > 0.0)]
    if positive_delta.size == 0:
        return 0.0
    typical_step = float(np.median(positive_delta))
    valid_pairs = (
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & (delta > 0.0)
        & (delta <= 3.0 * typical_step)
    )
    pair_mass = (
        0.5
        * (values[:-1] + values[1:])
        * delta
    )
    return float(np.sum(pair_mass[valid_pairs]))

# =============================================================================
# 内置默认配置
# =============================================================================

DEFAULT_CONFIG = json.loads(r"""{
  "config_version": "3.3.6",
  "algorithm_basis": {
    "paper_tsai": "10.1029/2011GL050255",
    "paper_luong_2024": "10.1029/2024JF007761",
    "paper_luong_2026": "10.1029/2025WR040371",
    "github_release": "seismic_bedload 0.2.0",
    "github_commit": "e666876e5750d0f27ef6a3f94b638ef7eb9e16d7",
    "priority": "paper_corrected"
  },
  "psd": {
    "_说明": "Luong 2024: 2^14点Hann窗、50%重叠、近1 min谱段在线性PSD空间取中位数。",
    "nperseg": 16384,
    "overlap_fraction": 0.5,
    "aggregation": "median",
    "display_fmin_hz": 1.0,
    "display_fmax_hz": 50.0
  },
  "inversion": {
    "_说明": "20–150 Hz 为当前 OBS 工程频带，应按仪器有效响应和场地噪声校核。",
    "model_type": "IMHE2026",
    "f_inv_min_hz": 20.0,
    "f_inv_max_hz": 150.0,
    "n_frequency_bins": 60,
    "band_statistic": "median",
    "frequency_grid": "native",
    "resolution_minutes": 0
  },
  "site": {
    "_说明": "以下保留附件中的现场占位值。slope是无量纲坡降S=tan(theta)，算法中不再二次取tan。",
    "channel_width_m": 132.5,
    "slope": 0.0069814304,
    "source_distance_m": 150.0,
    "bed_elevation_m": 2892.0
  },
  "sediment": {
    "_说明": "D50、粒径分布宽度与最大粒径必须替换为现场筛分或图像识别结果。",
    "D50_m": 0.02,
    "D_min_m": null,
    "D_max_m": 0.08,
    "grain_log_std": 0.45,
    "grain_points": 80,
    "rho_s_kg_m3": 2700.0,
    "rho_f_kg_m3": 1000.0,
    "tau_c50": 0.045,
    "clip_tau_c_tsai": false,
    "clip_tau_c_multimode": true,
    "roughness_m": null,
    "tsai_transport_stage": "excess"
  },
  "seismic": {
    "_说明": "vc0和zeta应优先由主动源/锤击试验校准；null时采用Tsai 2012通用岩石参数推导。",
    "v0_m_s": 2206.0,
    "z0_m": 1000.0,
    "f0_hz": 1.0,
    "a": 0.272,
    "Q0": 20.0,
    "eta": 0.0,
    "vc0_override_m_s": null,
    "zeta_override": null,
    "restitution_gamma": 0.0,
    "fz": 0.539,
    "Nzz": 0.352
  },
  "multimode": {
    "_说明": "Luong 2024/GitHub示例使用的截断对数正态冲击间隔参数。",
    "hop_t_min_s": 0.01,
    "hop_t_max_s": 0.5,
    "hop_log_median_s": 0.08,
    "hop_log_sigma": 1.0,
    "hop_points": 70
  },
  "empirical2026": {
    "_说明": "Luong 2026式(13)-(15)的ADLP标定参数；P/Pre按dB差转换为线性功率比。新场地宜用同步输沙观测重新标定。",
    "k": 5.2,
    "m": 0.25,
    "n": 1.36,
    "A_db": 29.0,
    "B_db": -127.0,
    "tau_cr": 0.045,
    "hydraulic_radius_depth_ratio": 1.0
  },
  "waveform": {
    "_说明": "manual_paz仅适用于附件中对应仪器；不同OBS必须使用正确StationXML或PAZ。",
    "response_mode": "manual_paz",
    "pre_filt_hz": [
      0.2,
      0.5,
      35.0,
      45.0
    ],
    "manual_paz": {
      "poles": [
        -30.61,
        -15.8,
        -0.1693,
        -0.09732,
        -0.03333
      ],
      "zeros": [
        0.0,
        0.0,
        0.0,
        -30.21,
        -16.15
      ],
      "gain": 1021.9,
      "adc_sensitivity": 1677700.0
    },
    "maximum_interpolated_gap_s": 1.0,
    "default_selection_hours": 168.0
  },
  "performance": {
    "_说明": "PSD 按时间块计算。预处理和反演使用所选分析采样率，降低采样率时逐级执行抗混叠滤波。波形预览另按显示范围提取极值。",
    "psd_chunk_blocks": 256,
    "plot_max_points": 120000
  },
  "literature_profiles": {
    "_说明": "参考参数只用于复现论文，不应直接替代现场参数。",
    "Tsai2012_reference": {
      "D50_m": 0.4,
      "channel_width_m": 50.0,
      "slope": 0.024439,
      "source_distance_m": 600.0,
      "vc0_m_s": 1295.0,
      "zeta": 0.374,
      "Q0": 20.0
    },
    "Luong2024_ADLP": {
      "D50_m": 0.009,
      "grain_log_std": 0.85,
      "channel_width_m": 10.0,
      "slope": 0.012218,
      "source_distance_m": 17.0,
      "vc0_m_s": 250.0,
      "zeta": 0.089,
      "Q0": 20.0,
      "inversion_band_hz": [
        30.0,
        80.0
      ]
    },
    "Luong2026_ADLP": {
      "D50_m": 0.005,
      "slope": 0.012,
      "inversion_band_hz": [
        30.0,
        80.0
      ],
      "A_db": 29.0,
      "B_db": -127.0,
      "k": 5.2,
      "m": 0.25,
      "n": 1.36,
      "tau_cr": 0.045
    }
  }
}""")


def load_default_config() -> dict[str, Any]:
    """返回内置默认配置的独立副本，并执行完整参数校验。"""

    config = json.loads(json.dumps(DEFAULT_CONFIG, ensure_ascii=False))
    validate_config(config)
    return config


# =============================================================================
# 波形文件头扫描与按时段延迟读取
# =============================================================================

@dataclass(frozen=True)
class WaveformHeaderRecord:
    """单个文件内一条 Trace 的轻量头信息。"""

    path: str
    trace_id: str
    channel: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    sampling_rate_hz: float
    npts: int
    format_name: str
    start_ns: int | None = None

    @property
    def duration_seconds(self) -> float:
        """返回头信息对应的时间长度。"""

        return max(
            0.0, (self.end_utc - self.start_utc).total_seconds()
        )


@dataclass(frozen=True)
class CoverageInterval:
    """同一通道合并后的连续覆盖时间段。"""

    trace_id: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    file_count: int


def _as_utc_timestamp(value) -> pd.Timestamp:
    """将 ObsPy/Python 时间对象统一为 UTC Timestamp。"""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


from contextlib import contextmanager


@contextmanager
def _sac_spacing_warning_filter():
    """Suppress only the unchanged-spacing notice, preserving SAC interval rounding."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^Sample spacing read from SAC file \((?P<spacing>[0-9.]+) when rounded to "
                r"nanoseconds\) was rounded off? to microsecond precision "
                r"\((?P=spacing)\).*#3408"
            ),
            category=UserWarning,
            module=r"^obspy\.io\.sac\.util$",
        )
        yield


def _read_obspy_waveform(reader, path, **kwargs):
    with _sac_spacing_warning_filter():
        return reader(path, **kwargs)


@dataclass(frozen=True)
class WaveformReadIssue:
    path: str
    reason: str
    headers: tuple[WaveformHeaderRecord, ...] = ()


@dataclass
class WaveformScanReport:
    records: list[WaveformHeaderRecord]
    issues: list[WaveformReadIssue]


OBS_FILE_COMPONENTS = {".bhz": "BHZ", ".bhn": "BHN", ".bhe": "BHE", ".hyd": "HYD"}


def waveform_channel(path, stats, format_name):
    if format_name == "SAC" and Path(path).suffix.lower() in OBS_FILE_COMPONENTS:
        return OBS_FILE_COMPONENTS[Path(path).suffix.lower()]
    return str(stats.channel)


def waveform_trace_id(path, trace, format_name):
    fields = trace.id.rsplit(".", 1)
    return fields[0] + "." + waveform_channel(path, trace.stats, format_name)


def waveform_header(path, trace, format_name=None):
    stats = trace.stats
    format_name = format_name or str(getattr(stats, "_format", "") or "AUTO")
    fs = float(stats.sampling_rate)
    if not np.isfinite(fs) or fs <= 0 or stats.npts <= 0:
        raise ValueError("文件头的采样率或样点数无效 / Invalid waveform sample rate or count.")
    return WaveformHeaderRecord(
        path, waveform_trace_id(path, trace, format_name), waveform_channel(path, stats, format_name),
        pd.Timestamp(stats.starttime.ns, unit="ns", tz="UTC"),
        pd.Timestamp(stats.endtime.ns, unit="ns", tz="UTC"),
        fs, int(stats.npts), format_name, stats.starttime.ns,
    )


def scan_waveform_headers(file_paths, workers=1, progress=None):
    """Keep readable records and file-specific failures in one scan report."""
    from obspy import read
    from obspy.io.sac import SACTrace

    paths = list(dict.fromkeys(str(Path(path)) for path in file_paths))
    records = []
    issues = []

    def scan(path):
        try:
            found = read(path, headonly=True)
            if not found:
                raise ValueError("文件中没有波形记录 / No waveform traces in file.")
            return [waveform_header(path, trace) for trace in found], None
        except Exception as exc:
            # Recover timing metadata only so skipped SAC files still mark their gap.
            timing = ()
            if "actual and theoretical file size" in str(exc).lower():
                try:
                    header = SACTrace.read(path, headonly=True, checksize=False)
                    timing = (waveform_header(path, header.to_obspy_trace(), "SAC"),)
                except Exception:
                    pass
            return [], WaveformReadIssue(path, str(exc), timing)

    def collect(scans):
        for i, (found, issue) in enumerate(scans, 1):
            records.extend(found)
            if issue is not None:
                issues.append(issue)
            if progress is not None and (i % 32 == 0 or i == len(paths)):
                progress(i, len(paths))

    count = min(len(paths), workers, logical_cpu_count())
    with _FALLBACK_READ_LOCK, _sac_spacing_warning_filter():
        if count > 1:
            with ThreadPoolExecutor(max_workers=count, thread_name_prefix="obs-headers") as pool:
                collect(pool.map(scan, paths))
        else:
            collect(map(scan, paths))
    records.sort(key=lambda item: (item.trace_id, item.start_utc, item.path))
    return WaveformScanReport(records, issues)


def read_waveform_headers(file_paths: Iterable[str | Path], workers: int = 1) -> list[WaveformHeaderRecord]:
    """Read headers for callers that require every input file to be readable."""
    report = scan_waveform_headers(file_paths, workers)
    if report.issues:
        details = "\n".join(f"{Path(issue.path).name}: {issue.reason}" for issue in report.issues[:12])
        raise RuntimeError(f"{len(report.issues)} 个文件无法读取头信息：\n{details}")
    if not report.records:
        raise RuntimeError("所选文件中没有可识别的地震波形头信息。")
    return report.records


def merge_coverage_intervals(
    records: Iterable[WaveformHeaderRecord],
    tolerance_seconds: float = 0.01,
) -> list[CoverageInterval]:
    """按通道合并相互重叠或近似连续的文件头时间段。"""

    tolerance = max(0.0, float(tolerance_seconds))
    grouped: dict[str, list[WaveformHeaderRecord]] = {}
    for record in records:
        grouped.setdefault(record.trace_id, []).append(record)

    merged: list[CoverageInterval] = []
    for trace_id, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item.start_utc)
        start = items[0].start_utc
        end = items[0].end_utc
        count = 1
        for item in items[1:]:
            gap = (item.start_utc - end).total_seconds()
            if gap <= tolerance:
                end = max(end, item.end_utc)
                count += 1
            else:
                merged.append(
                    CoverageInterval(trace_id, start, end, count)
                )
                start, end, count = item.start_utc, item.end_utc, 1
        merged.append(CoverageInterval(trace_id, start, end, count))
    return merged


def records_overlapping_window(
    records: Iterable[WaveformHeaderRecord],
    start_utc,
    end_utc,
) -> list[WaveformHeaderRecord]:
    """筛出与目标闭区间相交的文件头记录。"""

    start = _as_utc_timestamp(start_utc)
    end = _as_utc_timestamp(end_utc)
    if start >= end:
        raise ValueError("目标起始时间必须早于结束时间。")
    return [
        record
        for record in records
        if record.end_utc >= start and record.start_utc <= end
    ]


def read_waveform_window(
    records: Iterable[WaveformHeaderRecord],
    start_utc,
    end_utc,
):
    """
    读取目标时段内的单一垂向通道，并将文件间断档统一补零。

    对 MiniSEED，ObsPy 仍按 starttime/endtime 延迟截取文件；合并后使用
    ``fill_value=0`` 填充内部缺口，并对目标时间窗两端执行零值补齐。这样
    不连续文件可形成一条规则采样的连续序列，PSD 在缺测时段对应零信号。
    """

    try:
        from obspy import Stream, UTCDateTime, read
    except ImportError as exc:
        raise RuntimeError("读取地震波形需要安装 ObsPy。") from exc

    all_records = list(records)
    selected = records_overlapping_window(all_records, start_utc, end_utc)
    if not selected:
        # 目标窗口可能完全位于两个文件之间。此时仍从全局记录确定唯一通道，
        # 并读取距离窗口最近的文件以取得采样率，随后整窗补零。
        vertical_all = sorted(
            {record.trace_id for record in all_records if record.channel.upper().endswith("Z")}
        )
        available_all = sorted({record.trace_id for record in all_records})
        candidate_all = vertical_all or available_all
        if len(candidate_all) != 1:
            raise RuntimeError(
                "目标时段没有实测样本，且无法唯一确定补零通道；候选为："
                + ", ".join(candidate_all)
            )
        trace_id = candidate_all[0]
        channel_records = [r for r in all_records if r.trace_id == trace_id]
        if not channel_records:
            raise RuntimeError("没有可用于确定采样率的波形记录。")
        target_mid = _as_utc_timestamp(start_utc) + (
            _as_utc_timestamp(end_utc) - _as_utc_timestamp(start_utc)
        ) / 2
        nearest = min(
            channel_records,
            key=lambda r: min(
                abs((r.start_utc - target_mid).total_seconds()),
                abs((r.end_utc - target_mid).total_seconds()),
            ),
        )
        selected = [nearest]
    else:
        vertical_ids = sorted(
            {
                record.trace_id
                for record in selected
                if record.channel.upper().endswith("Z")
            }
        )
        available_ids = sorted({record.trace_id for record in selected})
        candidate_ids = vertical_ids or available_ids
        if len(candidate_ids) != 1:
            raise RuntimeError(
                "目标时段必须对应一个垂向通道；当前候选为："
                + ", ".join(candidate_ids)
            )
        trace_id = candidate_ids[0]

    paths = sorted({record.path for record in selected if record.trace_id == trace_id})
    if not paths:
        paths = [selected[0].path]

    start = UTCDateTime(_as_utc_timestamp(start_utc).to_pydatetime())
    end = UTCDateTime(_as_utc_timestamp(end_utc).to_pydatetime())
    if end <= start:
        raise ValueError("目标波形结束时间必须晚于开始时间。")

    stream = Stream()
    for path in paths:
        # 若窗口完全位于断档中，按时间裁切可能返回空流；此时读取头记录对应
        # 的短片段以获得通道及采样率，后续会将目标窗整体补零。
        partial = _read_obspy_waveform(read, path, starttime=start, endtime=end)
        if not partial:
            partial = _read_obspy_waveform(read, path)
        stream += partial
    stream = stream.select(id=trace_id)
    if not stream:
        raise RuntimeError("未读取到可用于建立连续时间轴的目标通道。")

    sampling_rates = {float(trace.stats.sampling_rate) for trace in stream}
    if len(sampling_rates) != 1:
        raise RuntimeError("同一通道文件采样率不一致，无法在统一时间轴上补零。")

    # 先合并内部断档，再严格裁切并补齐两端。
    stream.merge(method=1, fill_value=0)
    stream.trim(start, end, pad=True, fill_value=0, nearest_sample=True)
    if not stream or int(stream[0].stats.npts) < 2:
        raise RuntimeError("目标时段补零后仍无足够样本。")
    return stream, paths, trace_id

# =============================================================================
# PyQt6 图形界面
# =============================================================================

# -*- coding: utf-8 -*-
"""
===============================================================================
OBS Bedload Analyzer v3.3.6 - PyQt6
-------------------------------------------------------------------------------
基于文献与开源算法库:
1. Tsai, V.C. et al., 2012. A physical model for seismic noise generation from
   sediment transport in rivers. Geophysical Research Letters, 39(2).
2. Luong, L. et al., 2024. Seismic modeling of bedload transport in a gravel-bed
   alluvial channel. Journal of Geophysical Research: Earth Surface, 129(9).
3. Luong, L. et al., 2026. An empirical model combining seismic noise and shear
   stress to predict bedload flux. Water Resources Research, 62(2).
4. GitHub 仓库: https://github.com/locluong09/seismic-based-bedload-transport-models

核心功能模块:
- 0. 项目管理与工程路径持久化
- 1. 地震波形导入、去趋势、去均值与仪器响应校正 (Counts -> m/s)
- 2. 水位/水深/断面导入、UTC 统一与 H_eff/W_eff 计算
- 3. 实测粒径分布导入与活动粒径条件化
- 3. 连续 1-min 滑动 Welch PSD 功率谱计算 (消除小时切割导致的缝隙断层)
- 4. 物理模型参数配置 (粒径分布、河床坡度、台站距离等)
- 5. 推移质通量 qb/Qb 自动反演、数据表格导出及图表高清导出
===============================================================================
"""

import sys
import os
import traceback
import time
import numpy as np
import pandas as pd
from pathlib import Path

# 配置 Matplotlib 与 Qt6 绑定
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# 全局设置 Matplotlib 中文字体支持，防止 Glyph 缺失警告
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# PyQt6 UI 组件与绘图组件
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QProgressBar, QStatusBar, QDoubleSpinBox, QSpinBox, QTextEdit, QScrollArea,
    QDateTimeEdit, QCheckBox, QStackedWidget, QFormLayout, QPlainTextEdit,
    QHeaderView, QSizePolicy, QAbstractSpinBox, QAbstractItemView, QFrame,
    QSplitter
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QDateTime, QTimer, QEvent
)
from PyQt6.QtGui import (
    QFont,
    QColor,
    QIcon,
    QPixmap,
    QPainter,
    QPen,
    QPainterPath,
    QIntValidator,
    QAction,
)

# 检查是否安装了 ObsPy
try:
    import obspy
    from obspy import read, Stream
    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False




# =============================================================================
# 0. 矢量图标绘制函数
# =============================================================================

def create_scientific_icon(icon_type="waveform"):
    """使用 QPainter 动态绘制简洁、标准的科学矢量图标"""
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(0, 0, 0, 0))  # 透明背景
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    if icon_type == "waveform":  # 地震波形 / 信号处理图标
        painter.setBrush(QColor(2, 132, 199))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(1, 1, 30, 30, 6, 6)
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        path = QPainterPath()
        pts = [(3, 16), (7, 16), (10, 8), (14, 24), (18, 4), (22, 28), (25, 12), (29, 16)]
        path.moveTo(pts[0][0], pts[0][1])
        for pt in pts[1:]:
            path.lineTo(pt[0], pt[1])
        painter.drawPath(path)
        
    elif icon_type == "river":  # 水位 / 水深波浪图标
        painter.setBrush(QColor(14, 165, 233))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(1, 1, 30, 30, 6, 6)
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(3, 12); path.quadTo(10, 6, 16, 12); path.quadTo(22, 18, 29, 12)
        path.moveTo(3, 20); path.quadTo(10, 14, 16, 20); path.quadTo(22, 26, 29, 20)
        painter.drawPath(path)
        
    elif icon_type == "psd":  # 频谱柱状图图标
        painter.setBrush(QColor(139, 92, 246))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(1, 1, 30, 30, 6, 6)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRect(5, 18, 4, 10)
        painter.drawRect(11, 10, 4, 18)
        painter.drawRect(17, 6, 4, 22)
        painter.drawRect(23, 14, 4, 14)
        
    elif icon_type == "settings":  # 参数设置齿轮图标
        painter.setBrush(QColor(100, 116, 139))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(1, 1, 30, 30, 6, 6)
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        painter.drawEllipse(8, 8, 16, 16)
        painter.drawLine(16, 4, 16, 28)
        painter.drawLine(4, 16, 28, 16)
        
    elif icon_type == "inversion":  # 通量反演曲线图标
        painter.setBrush(QColor(16, 185, 129))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(1, 1, 30, 30, 6, 6)
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(4, 26); path.lineTo(12, 18); path.lineTo(18, 22); path.lineTo(28, 6)
        painter.drawPath(path)

    elif icon_type == "theme":  # 日月浅色/深色主题切换图标
        painter.setBrush(QColor(245, 158, 11))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(1, 1, 30, 30, 6, 6)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(6, 6, 20, 20)
        painter.setBrush(QColor(245, 158, 11))
        painter.drawEllipse(11, 6, 15, 20)
        
    painter.end()
    return QIcon(pixmap)


# =============================================================================
# 0.1 物理过程矢量原理示意组件
# =============================================================================

class ModelSchematicWidget(QWidget):
    """以低配色 CAD 图纸风格绘制模型的物理因果链。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.is_dark = True
        self.model_key = "Luong2024"

    def set_theme(self, is_dark):
        self.is_dark = is_dark
        self.update()

    def set_model(self, model_key):
        self.model_key = str(model_key)
        self.update()

    @staticmethod
    def _arrow(painter, x1, y1, x2, y2, color):
        """绘制 CAD 风格的单线箭头。"""
        painter.setPen(QPen(color, 1.4))
        painter.drawLine(x1, y1, x2, y2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        for offset in (2.55, -2.55):
            painter.drawLine(
                x2,
                y2,
                int(x2 + 7 * np.cos(angle + offset)),
                int(y2 + 7 * np.sin(angle + offset)),
            )

    @staticmethod
    def _box(painter, rect, number, title, lines, fill, border, text, accent):
        """绘制带编号与标题栏的技术图框。"""
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1.2))
        painter.drawRect(*rect)
        x, y, width, height = rect
        painter.drawLine(x, y + 34, x + width, y + 34)
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(x, y, 30, 34)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(
            x, y, 30, 34,
            Qt.AlignmentFlag.AlignCenter,
            f"{number:02d}",
        )
        painter.setPen(text)
        painter.setFont(QFont("Microsoft YaHei", 8, QFont.Weight.Bold))
        painter.drawText(x + 38, y + 22, title)
        painter.setFont(QFont("Microsoft YaHei", 8))
        for index, line in enumerate(lines):
            painter.drawText(x + 10, y + 55 + 19 * index, line)
        painter.setPen(QPen(border, 0.8, Qt.PenStyle.DotLine))
        painter.drawLine(x + 10, y + height - 22, x + width - 10, y + height - 22)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        bg = QColor("#111827") if self.is_dark else QColor("#f8fafc")
        text = QColor("#d7dee8") if self.is_dark else QColor("#334155")
        grid = QColor("#475569") if self.is_dark else QColor("#94a3b8")
        faint = QColor("#253247") if self.is_dark else QColor("#e2e8f0")
        accent = QColor("#2f8fbf") if self.is_dark else QColor("#256b91")
        card = QColor("#172033") if self.is_dark else QColor("#ffffff")

        painter.setBrush(bg)
        painter.setPen(QPen(grid, 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # 淡化的工程图网格仅用于辅助阅读，不参与物理编码。
        painter.setPen(QPen(faint, 0.6, Qt.PenStyle.DotLine))
        for xpos in range(20, w, 40):
            painter.drawLine(xpos, 1, xpos, h - 2)
        for ypos in range(20, h, 40):
            painter.drawLine(1, ypos, w - 2, ypos)

        margin = 14
        gap = 18
        box_w = max(150, int((w - 2 * margin - 3 * gap) / 4))
        box_h = h - 62
        y = 20
        x_values = [margin + index * (box_w + gap) for index in range(4)]

        if self.model_key == "Tsai2012":
            contents = [
                ("水力驱动", ["H(t), S → u*", "τ*(D) 与 τc*(D)"]),
                ("单一跃移", ["p(D), Ub, Hb", "颗粒法向冲击"]),
                ("地震传播", ["vc(f), vu(f), Q(f)", "χ(β), 台站距离 r0"]),
                ("通量反演", ["Pobs / P(qb=1)", "qb → Qb"]),
            ]
        elif self.model_key == "Luong2026":
            contents = [
                ("水力基准", ["Rh, S, D50", "τ*=RhS/(R·D50)"]),
                ("参考功率", ["Pre,dB=A log10τ*+B", "同剪应力基准 PSD"]),
                ("功率修正", ["P/Pre 线性功率比", "指数 m 表征偏离"]),
                ("经验反演", ["k, m, n 与 τcr*", "q*b → qb → Qb"]),
            ]
        else:
            contents = [
                ("水力与粒径", ["H(t), S, p(D)", "τ*(D), 多粒径输移"]),
                ("多输移模式", ["滚动 / 滑动 / 跃移", "p(tD), Ub, 斜向冲量"]),
                ("波场与衰减", ["vc(f), vu(f), Q(f)", "χ(β), r0, γ, fz"]),
                ("双重积分", ["∫p(D)∫p(tD)P", "qb → Qb"]),
            ]

        for index, (title, lines) in enumerate(contents):
            self._box(
                painter,
                (x_values[index], y, box_w, box_h),
                index + 1,
                title,
                lines,
                card,
                grid,
                text,
                accent,
            )
            if index < 3:
                self._arrow(
                    painter,
                    x_values[index] + box_w + 3,
                    y + box_h // 2,
                    x_values[index + 1] - 5,
                    y + box_h // 2,
                    accent,
                )

        icon_top = y + 101
        icon_bottom = y + box_h - 30

        # 水面、倾斜河床、水深标注与颗粒。
        x0, x1 = x_values[0] + 14, x_values[0] + box_w - 14
        painter.setPen(QPen(accent, 1.5))
        painter.drawLine(x0, icon_top + 2, x1, icon_top + 8)
        painter.setPen(QPen(text, 1.5))
        painter.drawLine(x0, icon_bottom, x1, icon_bottom - 13)
        self._arrow(painter, x0 + 16, icon_top + 5, x0 + 16, icon_bottom - 4, accent)
        painter.setBrush(card)
        painter.setPen(QPen(text, 1.1))
        for fraction in (0.50, 0.68, 0.84):
            px = int(x0 + fraction * (x1 - x0))
            painter.drawEllipse(px, icon_bottom - 14, 7, 7)

        # 颗粒轨迹或参考 PSD。
        left = x_values[1] + 18
        right = x_values[1] + box_w - 18
        if self.model_key == "Luong2026":
            painter.setPen(QPen(text, 1.2))
            painter.drawLine(left, icon_bottom, right, icon_bottom)
            painter.drawLine(left, icon_bottom, left, icon_top)
            curve = QPainterPath()
            curve.moveTo(left + 5, icon_bottom - 8)
            curve.cubicTo(
                left + 35,
                icon_bottom - 22,
                right - 45,
                icon_top + 20,
                right - 6,
                icon_top + 5,
            )
            painter.setPen(QPen(accent, 1.8))
            painter.drawPath(curve)
        else:
            painter.setPen(QPen(accent, 1.5, Qt.PenStyle.DashLine))
            arc = QPainterPath()
            arc.moveTo(left, icon_bottom)
            arc.quadTo((left + right) // 2, icon_top, right, icon_bottom)
            painter.drawPath(arc)
            painter.setBrush(card)
            painter.setPen(QPen(text, 1.1))
            painter.drawEllipse(left - 4, icon_bottom - 4, 9, 9)
            painter.drawEllipse(right - 4, icon_bottom - 4, 9, 9)
            if self.model_key == "Luong2024":
                for offset in (30, 46, 62):
                    painter.drawEllipse(left + offset, icon_bottom - 7, 7, 7)

        # 波场衰减或 P/Pre 柱状对比。
        left = x_values[2] + 18
        right = x_values[2] + box_w - 18
        if self.model_key == "Luong2026":
            painter.setBrush(QColor("#64748b"))
            painter.setPen(QPen(text, 0.8))
            painter.drawRect(left + 15, icon_bottom - 48, 26, 48)
            painter.setBrush(accent)
            painter.drawRect(left + 62, icon_bottom - 78, 26, 78)
            painter.setPen(text)
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(left + 15, icon_bottom + 15, "Pre")
            painter.drawText(left + 69, icon_bottom + 15, "P")
        else:
            painter.setPen(QPen(accent, 1.6))
            wave = QPainterPath()
            wave.moveTo(left, (icon_top + icon_bottom) // 2)
            span = max(1, right - left)
            for xpos in range(left, right + 1, 4):
                fraction = (xpos - left) / span
                amplitude = 28 * (1.0 - 0.75 * fraction)
                ypos = (
                    (icon_top + icon_bottom) // 2
                    + amplitude * np.sin(fraction * 8.0 * np.pi)
                )
                wave.lineTo(xpos, int(ypos))
            painter.drawPath(wave)

        # 最终通量输出。
        left = x_values[3] + 18
        right = x_values[3] + box_w - 18
        painter.setPen(QPen(text, 1.0))
        painter.drawLine(left, icon_bottom, right, icon_bottom)
        painter.drawLine(left, icon_bottom, left, icon_top)
        flux_curve = QPainterPath()
        flux_curve.moveTo(left + 5, icon_bottom - 4)
        flux_curve.cubicTo(
            left + 42,
            icon_bottom - 10,
            right - 42,
            icon_top + 55,
            right - 6,
            icon_top + 8,
        )
        painter.setPen(QPen(accent, 1.8))
        painter.drawPath(flux_curve)

        painter.setPen(text)
        painter.setFont(QFont("Microsoft YaHei", 8))
        caption = {
            "Tsai2012": "Tsai 2012：粒径积分的单一跃移物理模型",
            "Luong2024": "Luong 2024：粒径—冲击间隔双重积分的多输移模式模型",
            "Luong2026": "Luong 2026：参考PSD与超额Shields剪应力耦合的经验模型",
        }[self.model_key]
        painter.drawText(margin, h - 14, caption)
        painter.end()


class FormulaCanvas(FigureCanvas):
    """以论文排版方式渲染当前模型的关键公式。"""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 2.0))
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(190)
        self.model_key = "Luong2024"
        self.is_dark = True
        self.render_formula()

    def set_model(self, model_key):
        self.model_key = str(model_key)
        self.render_formula()

    def set_theme(self, is_dark):
        self.is_dark = bool(is_dark)
        self.render_formula()

    def render_formula(self):
        bg = "#111314" if self.is_dark else "#ffffff"
        fg = "#f1f3f4" if self.is_dark else "#263445"
        muted = "#a7adb0" if self.is_dark else "#64748b"
        formulas = {
            "Tsai2012": [
                (
                    r"$P(f)=\int p(D)\,\dot N(D)\,"
                    r"\mathcal{K}_{T}(f,D)\,\mathrm{d}D$",
                    "(1)",
                ),
                (
                    r"$U=8.1u_*[H/k_s]^{1/6},\quad"
                    r"U_b\propto T_{\rm eff}^{0.56},\quad"
                    r"H_b\propto T_{\rm eff}^{0.50}$",
                    "(2)",
                ),
                (
                    r"$q_b=\frac{\mathcal{S}_{\mathcal{B}}(P_{\rm obs})}"
                    r"{\mathcal{S}_{\mathcal{B}}(P_{q_b=1})}$",
                    "(3)",
                ),
            ],
            "Luong2024": [
                (
                    r"$P(f)=\int_D\!\int_{t_D}p(D)p_t(t_D)"
                    r"\frac{q_bW\,|\mathbf{I}\!\cdot\!\mathbf{N}|^2}"
                    r"{V_pU_bt_D}\,\mathcal{K}_{R}(f)\,"
                    r"\mathrm{d}t_D\,\mathrm{d}D$",
                    "(1)",
                ),
                (
                    r"$\mathcal{K}_{R}(f)="
                    r"\frac{\pi^{2}f^{3}}{4\rho_s^{2}v_c^{3}v_u^{2}}"
                    r"\,\chi(\beta)$",
                    "(2)",
                ),
                (
                    r"$q_b=\frac{\mathcal{S}_{\mathcal{B}}(P_{\rm obs})}"
                    r"{\mathcal{S}_{\mathcal{B}}(P_{q_b=1})}$",
                    "(3)",
                ),
            ],
            "Luong2026": [
                (
                    r"$q_b^{*}=k\left(\frac{P}{P_{\rm re}}\right)^m"
                    r"\left[\tau^{*}-\tau_{\rm cr}^{*}\right]_+^n$",
                    "(1)",
                ),
                (
                    r"$\tau^{*}=\frac{R_hS}{R\,D_{50}},\qquad"
                    r"P_{\rm re,dB}=A\log_{10}\tau^{*}+B$",
                    "(2)",
                ),
                (
                    r"$\frac{P}{P_{\rm re}}="
                    r"10^{(P_{\rm dB}-P_{\rm re,dB})/10}$",
                    "(3)",
                ),
            ],
        }
        formulas["IMHE2026"] = [
            (r"$\tau^* = R_hS/(R D),\quad U_b=0\;\;\mathrm{if}\;\;\tau^*\leq\tau_c^*(D)$", "(1)"),
            (r"$P_T(f)=C\,Wq_b\!\int p(D)V_pU_b(D)\,\mathrm{d}D\;\mathcal{K}(f)$", "(2)"),
            formulas["Luong2024"][2],
        ]
        self.figure.clear()
        self.figure.patch.set_facecolor(bg)
        axis = self.figure.add_subplot(111)
        axis.set_facecolor(bg)
        axis.axis("off")
        y_positions = (0.82, 0.51, 0.20)
        for index, (formula, tag) in enumerate(formulas[self.model_key]):
            axis.text(
                0.05,
                y_positions[index],
                formula,
                ha="left",
                va="center",
                color=fg,
                fontsize=11.5,
                transform=axis.transAxes,
            )
            axis.text(
                0.965,
                y_positions[index],
                tag,
                ha="right",
                va="center",
                color=muted,
                fontsize=9.0,
                transform=axis.transAxes,
            )
        self.figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.draw_idle()


# =============================================================================
# 2. 波形与 PSD 工具函数
# =============================================================================

def process_waveform_continuous(st_raw, paz, pre_filt=(0.2, 0.5, 35.0, 45.0)):
    """对连续波形去趋势、去均值并校正仪器响应；缺口保持零填充。"""

    st = st_raw.copy()
    st.merge(method=1, fill_value=0)
    st.detrend("linear")
    st.detrend("demean")

    if paz is not None and HAS_OBSPY:
        fs = float(st[0].stats.sampling_rate)
        nyquist = fs / 2.0
        pf = tuple(float(value) for value in pre_filt)
        if len(pf) != 4 or not (
            0.0 < pf[0] < pf[1] < pf[2] < pf[3] < nyquist
        ):
            raise ValueError(
                "预滤波必须为四个严格递增的正频率，且最高频率小于 "
                f"Nyquist={nyquist:g} Hz；当前值为 {pf!r}。"
            )
        for tr in st:
            tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pf)

    return st

# =============================================================================
# 3. QSS 主题配置
# =============================================================================

DARK_THEME_QSS = """
QMainWindow { background-color: #0c0d0e; color: #f1f3f4; }
QSplitter::handle { background-color: #2d3032; }
QSplitter::handle:hover { background-color: #9aabbe; }
QWidget { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 10pt; color: #f1f3f4; }
QMenuBar { background-color: #17191b; color: #f1f3f4; border-bottom: 1px solid #2d3032; }
QMenuBar::item:selected { background-color: #2d3032; border-radius: 4px; }
QMenu { background-color: #17191b; color: #f1f3f4; border: 1px solid #2d3032; }
QMenu::item:selected { background-color: #2f3843; color: #ffffff; }

QScrollArea, QScrollArea > QWidget > QWidget { background-color: #17191b; border: none; }

QTabWidget::pane { border: 1px solid #2d3032; background-color: #17191b; border-radius: 8px; }
QTabBar::tab { background-color: #0c0d0e; color: #a7adb0; padding: 8px 18px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; }
QTabBar::tab:selected { background-color: #17191b; color: #edf1f5; border-bottom: 2px solid #9aabbe; }

QGroupBox { font-weight: bold; border: 1px solid #2d3032; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: #17191b; color: #edf1f5; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; color: #edf1f5; }

QPushButton { background-color: #2d3032; color: #f1f3f4; border: 1px solid #414649; padding: 7px 15px; border-radius: 6px; font-weight: bold; }
QPushButton:hover { background-color: #2f3843; color: #ffffff; border-color: #59687a; }
QPushButton:disabled, QToolButton:disabled { background-color: #1c1f21; color: #757d81; border-color: #2d3032; }

QLineEdit, QDoubleSpinBox, QSpinBox, QDateTimeEdit, QComboBox { background-color: #0c0d0e; border: 1px solid #2d3032; border-radius: 6px; padding: 6px; min-height: 24px; color: #f1f3f4; selection-background-color: #2f3843; }
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QDateTimeEdit:focus, QComboBox:focus { border-color: #59687a; }
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QDateTimeEdit:disabled, QComboBox:disabled { background-color: #17191b; color: #a7adb0; }
QDoubleSpinBox, QSpinBox, QDateTimeEdit { padding-right: 30px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 26px; background-color: #2d3032; border-left: 1px solid #414649; }
QAbstractSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 5px; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 5px; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background-color: #2f3843; }
QComboBox QAbstractItemView { background-color: #17191b; color: #f1f3f4; selection-background-color: #2f3843; border: 1px solid #2d3032; }

QCalendarWidget { background-color: #111314; color: #f1f3f4; border: 1px solid #414649; }
QCalendarWidget QWidget#qt_calendar_navigationbar { background-color: #17191b; border-bottom: 1px solid #414649; }
QCalendarWidget QToolButton { background-color: #17191b; color: #f1f3f4; border: none; min-height: 26px; padding: 3px 8px; }
QCalendarWidget QToolButton:hover { background-color: #2d3032; color: #ffffff; }
QCalendarWidget QSpinBox { background-color: #0c0d0e; color: #f1f3f4; selection-background-color: #2f3843; }
QCalendarWidget QAbstractItemView:enabled { background-color: #111314; color: #dce0e1; selection-background-color: #354354; selection-color: #ffffff; outline: 0; }
QCalendarWidget QAbstractItemView:disabled { color: #757d81; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: #757d81; min-height: 38px; border-radius: 5px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed { background: #9aabbe; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { height: 0; background: none; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal { background: #757d81; min-width: 38px; border-radius: 5px; }
QScrollBar::handle:horizontal:hover, QScrollBar::handle:horizontal:pressed { background: #9aabbe; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { width: 0; background: none; }
QSlider::groove:horizontal { height: 6px; background: #2d3032; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #59687a; border-radius: 3px; }
QSlider::handle:horizontal { background: #dce0e1; border: 2px solid #59687a; width: 16px; height: 16px; margin: -6px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { background: #ffffff; border-color: #9aabbe; }

QTableWidget { background-color: #0c0d0e; color: #f1f3f4; gridline-color: #2d3032; border-radius: 6px; alternate-background-color: #1c1f21; }
QTableWidget::item { color: #f1f3f4; }
QTableWidget::item:selected { background-color: #2f3843; color: #ffffff; }
QTableWidget QTableCornerButton::section { background-color: #17191b; border: 1px solid #2d3032; }
QHeaderView::section { background-color: #17191b; color: #edf1f5; font-weight: bold; border: 1px solid #2d3032; padding: 6px; }

QStatusBar { background-color: #0c0d0e; color: #a7adb0; }
QProgressBar { border: 1px solid #2d3032; border-radius: 6px; text-align: center; background-color: #0c0d0e; color: #f1f3f4; }
QProgressBar::chunk { background-color: #9aabbe; border-radius: 6px; }

QTextEdit, QPlainTextEdit { background-color: #0c0d0e; color: #f1f3f4; border: 1px solid #2d3032; border-radius: 6px; font-family: 'Microsoft YaHei', 'Segoe UI'; selection-background-color: #2f3843; }
QCheckBox { color: #f1f3f4; spacing: 8px; }

QToolButton { background-color: #2d3032; color: #f1f3f4; border: 1px solid #414649; border-radius: 4px; margin: 2px; padding: 3px; }
QToolButton:hover { background-color: #2f3843; color: #ffffff; }

QMessageBox { background-color: #17191b; color: #f1f3f4; }
QMessageBox QLabel { color: #f1f3f4; font-size: 10pt; }
QMessageBox QPushButton { background-color: #2d3032; color: #f1f3f4; min-width: 70px; border: 1px solid #414649; border-radius: 4px; padding: 5px; }
"""

LIGHT_THEME_QSS = """
QMainWindow { background-color: #f8fafc; color: #0f172a; }
QSplitter::handle { background-color: #cbd5e1; }
QSplitter::handle:hover { background-color: #0284c7; }
QWidget { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 10pt; color: #0f172a; }
QMenuBar { background-color: #ffffff; color: #0f172a; border-bottom: 1px solid #e2e8f0; }
QMenuBar::item:selected { background-color: #e2e8f0; border-radius: 4px; }
QMenu { background-color: #ffffff; color: #0f172a; border: 1px solid #e2e8f0; }
QMenu::item:selected { background-color: #0284c7; color: #ffffff; }

QScrollArea, QScrollArea > QWidget > QWidget { background-color: #ffffff; border: none; }

QTabWidget::pane { border: 1px solid #cbd5e1; background-color: #ffffff; border-radius: 8px; }
QTabBar::tab { background-color: #f1f5f9; color: #64748b; padding: 8px 18px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; }
QTabBar::tab:selected { background-color: #ffffff; color: #0284c7; border-bottom: 2px solid #0284c7; }

QGroupBox { font-weight: bold; border: 1px solid #cbd5e1; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: #ffffff; color: #0284c7; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; color: #0284c7; }

QPushButton { background-color: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; padding: 7px 15px; border-radius: 6px; font-weight: bold; }
QPushButton:hover { background-color: #0284c7; color: #ffffff; border-color: #0284c7; }

QLineEdit, QDoubleSpinBox, QSpinBox, QDateTimeEdit, QComboBox { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px; min-height: 24px; color: #0f172a; selection-background-color: #0284c7; }
QDoubleSpinBox, QSpinBox, QDateTimeEdit { padding-right: 30px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 26px; background-color: #e2e8f0; border-left: 1px solid #cbd5e1; }
QAbstractSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 5px; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 5px; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background-color: #0284c7; }
QComboBox QAbstractItemView { background-color: #ffffff; color: #0f172a; selection-background-color: #0284c7; border: 1px solid #cbd5e1; }

QCalendarWidget { background-color: #ffffff; color: #0f172a; border: 1px solid #94a3b8; }
QCalendarWidget QWidget#qt_calendar_navigationbar { background-color: #f1f5f9; border-bottom: 1px solid #cbd5e1; }
QCalendarWidget QToolButton { background-color: #f1f5f9; color: #0f172a; border: none; min-height: 26px; padding: 3px 8px; }
QCalendarWidget QToolButton:hover { background-color: #e2e8f0; color: #0f172a; }
QCalendarWidget QSpinBox { background-color: #ffffff; color: #0f172a; selection-background-color: #0284c7; }
QCalendarWidget QAbstractItemView:enabled { background-color: #ffffff; color: #0f172a; selection-background-color: #256b91; selection-color: #ffffff; outline: 0; }
QCalendarWidget QAbstractItemView:disabled { color: #94a3b8; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: #94a3b8; min-height: 38px; border-radius: 5px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed { background: #256b91; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { height: 0; background: none; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal { background: #94a3b8; min-width: 38px; border-radius: 5px; }
QScrollBar::handle:horizontal:hover, QScrollBar::handle:horizontal:pressed { background: #256b91; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { width: 0; background: none; }
QSlider::groove:horizontal { height: 6px; background: #cbd5e1; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #256b91; border-radius: 3px; }
QSlider::handle:horizontal { background: #ffffff; border: 2px solid #256b91; width: 16px; height: 16px; margin: -6px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { background: #f8fafc; border-color: #0284c7; }

QTableWidget { background-color: #ffffff; color: #0f172a; gridline-color: #cbd5e1; border-radius: 6px; alternate-background-color: #f1f5f9; }
QTableWidget::item { color: #0f172a; background-color: #ffffff; }
QTableWidget::item:selected { background-color: #0284c7; color: #ffffff; }
QTableWidget QTableCornerButton::section { background-color: #f8fafc; border: 1px solid #cbd5e1; }
QHeaderView::section { background-color: #f8fafc; color: #0284c7; font-weight: bold; border: 1px solid #cbd5e1; padding: 6px; }

QStatusBar { background-color: #ffffff; color: #64748b; }
QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; text-align: center; background-color: #f8fafc; color: #0f172a; }
QProgressBar::chunk { background-color: #0284c7; border-radius: 6px; }

QScrollArea { border: none; background-color: transparent; }
QTextEdit, QPlainTextEdit { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; font-family: 'Microsoft YaHei', 'Segoe UI'; selection-background-color: #0284c7; }
QCheckBox { color: #0f172a; spacing: 8px; }

QToolButton { background-color: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 4px; margin: 2px; padding: 3px; }
QToolButton:hover { background-color: #0284c7; color: #ffffff; }

QMessageBox { background-color: #ffffff; color: #0f172a; }
QMessageBox QLabel { color: #0f172a; font-size: 10pt; }
QMessageBox QPushButton { background-color: #f1f5f9; color: #0f172a; min-width: 70px; border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px; }
"""


# =============================================================================
# 4. 后台 Worker 计算线程 (实时反馈计算百分比进度 progress_sig)
# =============================================================================

class HeaderScanWorker(QThread):
    """在后台仅读取多个波形文件头。"""

    finished_sig = pyqtSignal(object)
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)

    def __init__(self, file_paths, workers=1, *, allow_partial=False):
        super().__init__()
        self.file_paths = list(file_paths)
        self.workers = workers
        self.allow_partial = allow_partial

    def run(self):
        try:
            self.progress_sig.emit(10, "正在快速扫描波形文件头...")
            if self.allow_partial:
                def progress(done, total):
                    self.progress_sig.emit(round(100 * done / total), f"波形文件头 {done:,}/{total:,}")
                report = scan_waveform_headers(self.file_paths, self.workers, progress)
                self.finished_sig.emit(report)
            else:
                records = read_waveform_headers(self.file_paths, self.workers)
                self.progress_sig.emit(100, f"文件头扫描完成：{len(records)} 条Trace记录")
                self.finished_sig.emit(records)
        except Exception as exc:
            self.error_sig.emit(
                f"波形文件头扫描异常: {exc}\n{traceback.format_exc()}"
            )


class WaveformWorker(QThread):
    finished_sig = pyqtSignal(object, object, object)
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)

    def __init__(
        self,
        header_records,
        start_utc,
        end_utc,
        paz_dict,
        pre_filt,
        maximum_gap_seconds=1.0,
    ):
        super().__init__()
        self.header_records = list(header_records)
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.paz_dict = paz_dict
        self.pre_filt = pre_filt
        self.maximum_gap_seconds = float(maximum_gap_seconds)

    def run(self):
        try:
            if not HAS_OBSPY:
                raise RuntimeError("ObsPy 库未安装，请在命令行执行: pip install obspy")

            self.progress_sig.emit(10, "正在按目标时段读取相交的数据片段...")
            st_raw, paths, trace_id = read_waveform_window(
                self.header_records, self.start_utc, self.end_utc
            )
            self.progress_sig.emit(
                35,
                f"已读取 {len(paths)} 个相交文件，通道 {trace_id}",
            )
            sampling_rates = {
                float(trace.stats.sampling_rate) for trace in st_raw
            }
            if len(sampling_rates) != 1:
                raise RuntimeError("同一通道的多个文件采样率不一致，不能直接合并。")

            gaps = list(st_raw.get_gaps())
            long_gaps = [
                gap for gap in gaps if float(gap[6]) > self.maximum_gap_seconds
            ]
            gap_note = (
                f"；检测到 {len(gaps)} 个断档，其中 {len(long_gaps)} 个超过 "
                f"{self.maximum_gap_seconds:g} s，均按零值填充"
                if gaps else "；未检测到内部断档"
            )
            self.progress_sig.emit(50, "正在合并目标时段并以 0 填充全部断档...")
            st_raw.merge(method=1, fill_value=0)
            self.progress_sig.emit(58, f"连续时间轴已建立{gap_note}")
            
            self.progress_sig.emit(70, "正在进行去趋势与去均值处理...")
            st_det = st_raw.copy()
            st_det.detrend("linear")
            st_det.detrend("demean")
            
            self.progress_sig.emit(85, "正在进行仪器响应校正 (Counts -> m/s)...")
            st_vel = process_waveform_continuous(st_raw, self.paz_dict, self.pre_filt)
            
            self.progress_sig.emit(100, "波形预处理完成！")
            self.finished_sig.emit(st_raw[0], st_det[0], st_vel[0])
        except Exception as e:
            err_msg = f"波形预处理异常: {str(e)}\n{traceback.format_exc()}"
            self.error_sig.emit(err_msg)


class PSDWorker(QThread):
    finished_sig = pyqtSignal(object, object)
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)

    def __init__(
        self,
        tr_vel,
        nperseg,
        fmin,
        fmax,
        overlap_fraction=0.5,
        aggregation="median",
        chunk_blocks=256,
    ):
        super().__init__()
        self.tr_vel = tr_vel
        self.nperseg = nperseg
        self.fmin = fmin
        self.fmax = fmax
        self.overlap_fraction = float(overlap_fraction)
        self.aggregation = str(aggregation)
        self.chunk_blocks = int(chunk_blocks)

    def run(self):
        try:
            self.progress_sig.emit(
                10,
                "正在分块生成 2^N 点 Hann 谱段"
                f"（{self.overlap_fraction:.0%}重叠）...",
            )
            df_psd, metadata = compute_luong_minute_psd(
                data=np.asarray(self.tr_vel.data, dtype=float),
                sampling_rate_hz=float(self.tr_vel.stats.sampling_rate),
                start_time_utc=self.tr_vel.stats.starttime.datetime,
                nperseg=self.nperseg,
                overlap_fraction=self.overlap_fraction,
                fmin=self.fmin,
                fmax=self.fmax,
                aggregation=self.aggregation,
                chunk_blocks=self.chunk_blocks,
            )
            self.progress_sig.emit(
                100,
                f"PSD完成：{self.overlap_fraction:.0%}重叠，"
                f"线性PSD{self.aggregation}，"
                f"约{metadata.blocks_per_hour:.1f}块/h",
            )
            self.finished_sig.emit(df_psd, metadata)
        except Exception as e:
            err_msg = f"PSD 计算异常: {str(e)}\n{traceback.format_exc()}"
            self.error_sig.emit(err_msg)


class InversionWorker(QThread):
    finished_sig = pyqtSignal(object)
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)

    def __init__(self, df_psd, river_series, model_params):
        super().__init__()
        self.df_psd = df_psd
        self.river_series = river_series
        self.params = model_params

    def run(self):
        try:
            self.progress_sig.emit(
                5, "正在同轴插值水深并构建粒径/冲击间隔概率分布..."
            )
            res_df = invert_bedload_timeseries(
                self.df_psd, self.river_series, self.params
            )
            self.progress_sig.emit(100, "推移质通量反演成功！")
            self.finished_sig.emit(res_df)
        except Exception as e:
            err_msg = f"反演异常: {str(e)}\n{traceback.format_exc()}"
            self.error_sig.emit(err_msg)


# =============================================================================
# 5. 主 GUI 界面
# =============================================================================

class OBSAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_default_config()
        self.setWindowTitle(
            f"OBS Analyzer v{APP_VERSION} - Bedload Transport Inversion System"
        )
        self.setWindowIcon(create_scientific_icon("waveform"))
        self.resize(1450, 920)
        
        self.is_dark_theme = True

        self.st_raw_tr = None
        self.st_det_tr = None
        self.st_vel_tr = None
        self.waveform_headers: list[WaveformHeaderRecord] = []
        self.df_river = None
        self.river_series_utc = None
        self.df_psd = None
        self.psd_metadata = None
        self.df_result = None

        self.init_ui()
        self.configure_spinboxes()
        self.apply_theme()

    def init_ui(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("文件 (&F)")
        act_open_wf = QAction("打开波形文件...", self)
        act_open_wf.triggered.connect(self.on_load_waveform)
        file_menu.addAction(act_open_wf)
        
        act_open_river = QAction("打开水位数据...", self)
        act_open_river.triggered.connect(self.on_load_river)
        file_menu.addAction(act_open_river)
        file_menu.addSeparator()
        
        act_exit = QAction("退出系统", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        view_menu = menubar.addMenu("视图 (&V)")
        self.act_toggle_theme = QAction("切换至浅色主题", self)
        self.act_toggle_theme.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.act_toggle_theme)

        help_menu = menubar.addMenu("帮助 (&H)")
        act_about = QAction("关于 OBS Analyzer", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

        self.setCentralWidget(main_layout_widget := QWidget())
        main_layout = QVBoxLayout(main_layout_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.btn_theme_toggle = QPushButton(" 主题切换 (深/浅)")
        self.btn_theme_toggle.setIcon(create_scientific_icon("theme"))
        self.btn_theme_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        self.tabs.setCornerWidget(self.btn_theme_toggle, Qt.Corner.TopRightCorner)

        # 底部状态栏与确定性进度条
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready | 系统就绪")

        self.setup_tab_waveform()
        self.setup_tab_river()
        self.setup_tab_psd()
        self.setup_tab_model()
        self.setup_tab_inversion()

    def wrap_in_scroll(self, widget, width=390):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setFixedWidth(width)
        return scroll

    def apply_theme(self):
        started = time.perf_counter()
        self.setUpdatesEnabled(False)
        if self.is_dark_theme:
            self.setStyleSheet(DARK_THEME_QSS)
            self.act_toggle_theme.setText("切换至浅色主题")
        else:
            self.setStyleSheet(LIGHT_THEME_QSS)
            self.act_toggle_theme.setText("切换至深色主题")
            
        if hasattr(self, 'schematic_widget'):
            self.schematic_widget.set_theme(self.is_dark_theme)
        if hasattr(self, "formula_canvas"):
            self.formula_canvas.set_theme(self.is_dark_theme)
        self.style_existing_canvases()
        self.setUpdatesEnabled(True)
        self.update()
        self.status_bar.showMessage(
            f"主题已切换（{(time.perf_counter() - started) * 1000:.0f} ms）",
            2500,
        )

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def apply_canvas_style(self, fig, ax_list):
        bg_color = '#17191b' if self.is_dark_theme else '#ffffff'
        axis_bg = '#0c0d0e' if self.is_dark_theme else '#ffffff'
        text_color = '#f1f3f4' if self.is_dark_theme else '#0f172a'
        title_color = '#edf1f5' if self.is_dark_theme else '#0284c7'
        grid_color = '#2d3032' if self.is_dark_theme else '#cbd5e1'
        spine_color = '#414649' if self.is_dark_theme else '#94a3b8'
        
        fig.patch.set_facecolor(bg_color)
        if not isinstance(ax_list, (list, np.ndarray)):
            ax_list = [ax_list]
            
        for ax in ax_list:
            ax.set_facecolor(axis_bg)
            ax.tick_params(colors=text_color, which='both', labelsize=9)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(title_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
            ax.grid(True, color=grid_color, linestyle='--', alpha=0.6)

            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(bg_color)
                leg.get_frame().set_edgecolor(spine_color)
                for txt in leg.get_texts():
                    txt.set_color(text_color)

    def redraw_all_canvases(self):
        if self.st_raw_tr is not None:
            self.plot_waveform_canvas()
        if self.river_series_utc is not None:
            self.plot_river_canvas()
        if self.df_psd is not None:
            self.plot_psd_canvas()
        if self.df_result is not None:
            self.plot_inv_canvas()

    def style_existing_canvases(self):
        """只修改现有图元配色，不重新计算或重绘数据。"""

        for name in ("fig_wf", "fig_river", "fig_psd", "fig_inv"):
            figure = getattr(self, name, None)
            if figure is None:
                continue
            self.apply_canvas_style(figure, list(figure.axes))
            canvas = getattr(self, name.replace("fig_", "canvas_"), None)
            if canvas is not None:
                canvas.draw_idle()

    def configure_spinboxes(self):
        """统一微调控件，并彻底禁止滚轮无意修改数值。"""

        for spin in self.findChildren(QAbstractSpinBox):
            spin.setMinimumWidth(135)
            spin.setAccelerated(True)
            spin.setKeyboardTracking(False)
            spin.setButtonSymbols(
                QAbstractSpinBox.ButtonSymbols.UpDownArrows
            )
            if isinstance(spin, QDoubleSpinBox):
                spin.setStepType(
                    QAbstractSpinBox.StepType.AdaptiveDecimalStepType
                )
            spin.installEventFilter(self)

        # 下拉框在滚轮下也会静默改变当前项，同样禁用。
        for combo in self.findChildren(QComboBox):
            combo.installEventFilter(self)

    def eventFilter(self, watched, event):
        """阻止滚轮修改数值、日期和下拉选项；键盘与按钮操作不受影响。"""

        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(watched, (QAbstractSpinBox, QComboBox))
        ):
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    # =========================================================================
    # 槽函数：实时百分比进度条响应更新
    # =========================================================================
    def on_progress_update(self, val, msg):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(val)
        self.status_bar.showMessage(msg)

    # =========================================================================
    # Tab 1: 波形预处理
    # =========================================================================
    def setup_tab_waveform(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_load = QGroupBox("1. 波形文件加载与快速头扫描")
        gl = QVBoxLayout(group_load)
        self.btn_load_wf = QPushButton(
            "选择多个 BHZ / MiniSEED / MSD / SAC 文件..."
        )
        self.btn_load_wf.clicked.connect(self.on_load_waveform)
        self.lbl_wf_info = QLabel("未选择文件；此步骤不会读取或绘制波形样本。")
        self.lbl_wf_info.setWordWrap(True)
        gl.addWidget(self.btn_load_wf)
        gl.addWidget(self.lbl_wf_info)

        self.table_wf_headers = QTableWidget(0, 6)
        self.table_wf_headers.setHorizontalHeaderLabels(
            ["文件", "Trace ID", "开始时间 UTC", "结束时间 UTC", "fs (Hz)", "npts"]
        )
        self.table_wf_headers.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table_wf_headers.setAlternatingRowColors(True)
        self.table_wf_headers.setMinimumHeight(180)
        self.table_wf_headers.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in range(1, 6):
            self.table_wf_headers.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        gl.addWidget(self.table_wf_headers)

        self.txt_wf_coverage = QPlainTextEdit()
        self.txt_wf_coverage.setReadOnly(True)
        self.txt_wf_coverage.setMaximumHeight(155)
        self.txt_wf_coverage.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.NoWrap
        )
        self.txt_wf_coverage.setFont(QFont("Consolas", 9))
        self.txt_wf_coverage.setPlaceholderText(
            "扫描完成后显示同一通道的全部连续覆盖时间段。"
        )
        gl.addWidget(QLabel("合并后的数据覆盖时间段："))
        gl.addWidget(self.txt_wf_coverage)
        panel_layout.addWidget(group_load)

        group_interval = QGroupBox("2. 目标区间选择")
        il = QFormLayout(group_interval)
        self.dt_wf_start = QDateTimeEdit()
        self.dt_wf_end = QDateTimeEdit()
        for editor in (self.dt_wf_start, self.dt_wf_end):
            editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz 'UTC'")
            editor.setTimeSpec(Qt.TimeSpec.UTC)
            editor.setCalendarPopup(True)
            editor.setEnabled(False)
            editor.dateTimeChanged.connect(self.update_waveform_selection_label)
        il.addRow("开始时间：", self.dt_wf_start)
        il.addRow("结束时间：", self.dt_wf_end)
        self.lbl_wf_selection = QLabel("等待文件头扫描。")
        self.lbl_wf_selection.setWordWrap(True)
        il.addRow(self.lbl_wf_selection)
        panel_layout.addWidget(group_interval)

        group_paz = QGroupBox("3. 响应校正与执行预处理")
        pl = QGridLayout(group_paz)
        
        pl.addWidget(QLabel("Gain:"), 0, 0)
        paz_cfg = self.config["waveform"]["manual_paz"]
        self.spin_gain = QDoubleSpinBox(); self.spin_gain.setRange(1e-5, 1e9); self.spin_gain.setValue(paz_cfg["gain"])
        pl.addWidget(self.spin_gain, 0, 1)

        pl.addWidget(QLabel("ADC Sens:"), 1, 0)
        self.spin_adc = QDoubleSpinBox(); self.spin_adc.setRange(1.0, 1e12); self.spin_adc.setValue(paz_cfg["adc_sensitivity"])
        pl.addWidget(self.spin_adc, 1, 1)

        pl.addWidget(QLabel("预滤波 (Hz):"), 2, 0)
        self.txt_prefilt = QLineEdit(
            ", ".join(str(value) for value in self.config["waveform"]["pre_filt_hz"])
        )
        pl.addWidget(self.txt_prefilt, 2, 1)

        self.btn_process_wf = QPushButton(
            "读取所选区间并执行预处理 (Counts → m/s)"
        )
        self.btn_process_wf.setEnabled(False)
        self.btn_process_wf.clicked.connect(self.on_process_waveform)
        pl.addWidget(self.btn_process_wf, 3, 0, 1, 2)

        panel_layout.addWidget(group_paz)
        panel_layout.addStretch()

        layout.addWidget(self.wrap_in_scroll(panel, 500))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_wf = Figure(figsize=(8, 6))
        self.canvas_wf = FigureCanvas(self.fig_wf)
        plot_layout.addWidget(NavigationToolbar(self.canvas_wf, self))
        plot_layout.addWidget(self.canvas_wf)
        layout.addWidget(plot_panel)

        self.tabs.addTab(tab, create_scientific_icon("waveform"), "1. 波形预处理")

    # =========================================================================
    # Tab 2: 水位与水深
    # =========================================================================
    def setup_tab_river(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_river = QGroupBox("1. 水位文件读取")
        rl = QGridLayout(group_river)

        self.btn_load_river = QPushButton("加载水位文件 (Excel/CSV)...")
        self.btn_load_river.clicked.connect(self.on_load_river)
        rl.addWidget(self.btn_load_river, 0, 0, 1, 2)

        rl.addWidget(QLabel("河床高程 Z_bed (m):"), 1, 0)
        self.spin_bed_elev = QDoubleSpinBox(); self.spin_bed_elev.setRange(-1000.0, 10000.0); self.spin_bed_elev.setValue(self.config["site"]["bed_elevation_m"])
        rl.addWidget(self.spin_bed_elev, 1, 1)

        rl.addWidget(QLabel("原始数据时区:"), 2, 0)
        self.combo_tz = QComboBox()
        self.combo_tz.addItems(["Asia/Shanghai (北京时间)", "UTC", "US/Mountain", "US/Pacific"])
        rl.addWidget(self.combo_tz, 2, 1)

        self.btn_apply_river = QPushButton("转换并对齐水深 H(t)")
        self.btn_apply_river.clicked.connect(self.on_apply_river)
        rl.addWidget(self.btn_apply_river, 3, 0, 1, 2)

        panel_layout.addWidget(group_river)

        group_table = QGroupBox("数据预览")
        tbl = QVBoxLayout(group_table)
        self.table_river = QTableWidget()
        tbl.addWidget(self.table_river)
        panel_layout.addWidget(group_table)

        layout.addWidget(self.wrap_in_scroll(panel))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_river = Figure(figsize=(8, 6))
        self.canvas_river = FigureCanvas(self.fig_river)
        plot_layout.addWidget(NavigationToolbar(self.canvas_river, self))
        plot_layout.addWidget(self.canvas_river)
        layout.addWidget(plot_panel)

        self.tabs.addTab(tab, create_scientific_icon("river"), "2. 水位与水深")

    # =========================================================================
    # Tab 3: PSD 时频谱
    # =========================================================================
    def setup_tab_psd(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_psd = QGroupBox("1. PSD 功率谱密度 (Welch)")
        pl = QGridLayout(group_psd)

        pl.addWidget(QLabel("窗长 NPERSEG:"), 0, 0)
        self.combo_nperseg = QComboBox()
        self.combo_nperseg.addItems(["16384 (2^14)", "8192 (2^13)", "4096 (2^12)"])
        configured_nperseg = int(self.config["psd"]["nperseg"])
        for index in range(self.combo_nperseg.count()):
            if self.combo_nperseg.itemText(index).startswith(str(configured_nperseg)):
                self.combo_nperseg.setCurrentIndex(index)
                break
        pl.addWidget(self.combo_nperseg, 0, 1)

        pl.addWidget(QLabel("显示 Fmin (Hz):"), 1, 0)
        self.spin_psd_fmin = QDoubleSpinBox()
        self.spin_psd_fmin.setRange(0.001, 10000.0)  # 解除 99.99 限制
        self.spin_psd_fmin.setValue(self.config["psd"]["display_fmin_hz"])
        pl.addWidget(self.spin_psd_fmin, 1, 1)

        pl.addWidget(QLabel("显示 Fmax (Hz):"), 2, 0)
        self.spin_psd_fmax = QDoubleSpinBox()
        self.spin_psd_fmax.setRange(0.001, 10000.0)  # 解除 99.99 限制
        self.spin_psd_fmax.setValue(self.config["psd"]["display_fmax_hz"])
        pl.addWidget(self.spin_psd_fmax, 2, 1)

        self.chk_show_depth_psd = QCheckBox("在PSD图中显示水深曲线及右侧坐标")
        self.chk_show_depth_psd.setChecked(True)
        self.chk_show_depth_psd.toggled.connect(self.plot_psd_canvas)
        pl.addWidget(self.chk_show_depth_psd, 3, 0, 1, 2)

        overlap_percent = 100.0 * float(self.config["psd"]["overlap_fraction"])
        self.btn_calc_psd = QPushButton(
            f"计算 Luong 近1-min PSD（{overlap_percent:g}%重叠）"
        )
        self.btn_calc_psd.clicked.connect(self.on_calc_psd)
        pl.addWidget(self.btn_calc_psd, 4, 0, 1, 2)

        panel_layout.addWidget(group_psd)
        panel_layout.addStretch()

        layout.addWidget(self.wrap_in_scroll(panel))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_psd = Figure(figsize=(8, 6))
        self.canvas_psd = FigureCanvas(self.fig_psd)
        plot_layout.addWidget(NavigationToolbar(self.canvas_psd, self))
        plot_layout.addWidget(self.canvas_psd)
        layout.addWidget(plot_panel)

        self.tabs.addTab(tab, create_scientific_icon("psd"), "3. PSD 时频谱图")

    # =========================================================================
    # Tab 4: 模型配置
    # =========================================================================
    def setup_tab_model(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group_sel = QGroupBox("1. 反演模型选择")
        sl = QHBoxLayout(group_sel)
        self.combo_model = QComboBox()
        model_items = [
            ("Tsai et al. (2012) 单一跃移物理模型", "Tsai2012"),
            ("Luong et al. (2024) 多输移模式物理模型", "Luong2024"),
            ("Luong et al. (2026) 地震—水力经验模型", "Luong2026"),
        ]
        for label, key in model_items:
            self.combo_model.addItem(label, key)
        configured_model = self.config["inversion"]["model_type"]
        for index in range(self.combo_model.count()):
            if self.combo_model.itemData(index) == configured_model:
                self.combo_model.setCurrentIndex(index)
                break
        sl.addWidget(QLabel("模型算法:"))
        sl.addWidget(self.combo_model)
        sl.addStretch()
        layout.addWidget(group_sel)

        sediment_cfg = self.config["sediment"]
        site_cfg = self.config["site"]
        seismic_cfg = self.config["seismic"]
        multimode_cfg = self.config["multimode"]
        empirical_cfg = self.config["empirical2026"]
        inversion_cfg = self.config["inversion"]

        def make_double(value, lower, upper, decimals=6):
            spin = QDoubleSpinBox()
            spin.setRange(lower, upper)
            spin.setDecimals(decimals)
            spin.setValue(float(value))
            return spin

        def add_double(form, controls, key, label, value, lower, upper, decimals=6):
            control = make_double(value, lower, upper, decimals)
            controls[key] = control
            form.addRow(label, control)

        def add_integer(form, controls, key, label, value, lower=8, upper=5000):
            control = QSpinBox()
            control.setRange(lower, upper)
            control.setValue(int(value))
            controls[key] = control
            form.addRow(label, control)

        self.model_controls = {}
        self.model_param_stack = QStackedWidget()

        def physical_page(model_key, include_multimode=False):
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            controls = {}
            self.model_controls[model_key] = controls
            form.addRow(QLabel("<b>泥沙与粒径积分</b>"))
            add_double(form, controls, "D50", "中值粒径 D50 (m)", sediment_cfg["D50_m"], 1e-5, 5.0, 6)
            add_double(form, controls, "D_min", "最小粒径 Dmin (m；0=自动)", sediment_cfg["D_min_m"] or 0.0, 0.0, 5.0, 6)
            add_double(form, controls, "D_max", "最大粒径 Dmax (m)", sediment_cfg["D_max_m"], 1e-5, 10.0, 6)
            add_double(form, controls, "grain_log_std", "对数粒径标准差 σ", sediment_cfg["grain_log_std"], 0.01, 5.0, 5)
            add_integer(form, controls, "grain_points", "粒径积分点数", sediment_cfg["grain_points"])
            add_double(form, controls, "rho_s", "颗粒密度 ρs (kg/m³)", sediment_cfg["rho_s_kg_m3"], 1001, 6000, 2)
            add_double(form, controls, "rho_f", "流体密度 ρf (kg/m³)", sediment_cfg["rho_f_kg_m3"], 100, 2000, 2)
            form.addRow(QLabel("<b>河道、水力与反演</b>"))
            add_double(form, controls, "W", "有效河宽 W (m)", site_cfg["channel_width_m"], 0.01, 10000, 3)
            add_double(form, controls, "slope", "无量纲坡降 S=tan(θ)", site_cfg["slope"], 1e-7, 0.999, 8)
            add_double(form, controls, "r0", "台站距离 r0 (m)", site_cfg["source_distance_m"], 0.01, 100000, 3)
            add_double(form, controls, "tau_c50", "临界 Shields τc,50*", sediment_cfg["tau_c50"], 0.0001, 1.0, 5)
            add_double(form, controls, "f_inv_min", "反演频带下限 (Hz)", inversion_cfg["f_inv_min_hz"], 0.001, 10000, 3)
            add_double(form, controls, "f_inv_max", "反演频带上限 (Hz)", inversion_cfg["f_inv_max_hz"], 0.002, 10000, 3)
            add_integer(form, controls, "n_frequency_bins", "模型频率点数", inversion_cfg["n_frequency_bins"], 2, 2000)
            form.addRow(QLabel("<b>瑞利波传播</b>"))
            for key, label, value, lo, hi, dec in [
                ("v0", "介质速度 v0 (m/s)", seismic_cfg["v0_m_s"], 1, 20000, 3),
                ("z0", "参考深度 z0 (m)", seismic_cfg["z0_m"], 0.01, 100000, 3),
                ("f0", "参考频率 f0 (Hz)", seismic_cfg["f0_hz"], 0.001, 1000, 3),
                ("a", "速度幂指数 a", seismic_cfg["a"], 0, 0.999, 6),
                ("Q0", "品质因子 Q0", seismic_cfg["Q0"], 0.01, 10000, 3),
                ("eta", "Q频率指数 η", seismic_cfg["eta"], -5, 5, 6),
                ("vc0_override", "vc0覆盖值 (m/s；0=自动)", seismic_cfg["vc0_override_m_s"] or 0.0, 0, 20000, 3),
                ("zeta_override", "ζ覆盖值（-1=自动）", seismic_cfg["zeta_override"] if seismic_cfg["zeta_override"] is not None else -1.0, -1, 10, 6),
            ]:
                add_double(form, controls, key, label, value, lo, hi, dec)
            if include_multimode:
                form.addRow(QLabel("<b>Luong 2024 多模式冲击</b>"))
                for key, label, value, lo, hi, dec in [
                    ("restitution", "恢复系数 γ", seismic_cfg["restitution_gamma"], 0, 1, 5),
                    ("fz", "法向速度比 fz", seismic_cfg["fz"], 0, 10, 5),
                    ("Nzz", "垂向方向系数 Nzz", seismic_cfg["Nzz"], 0, 10, 5),
                    ("hop_t_min", "冲击间隔下限 (s)", multimode_cfg["hop_t_min_s"], 1e-5, 100, 6),
                    ("hop_t_max", "冲击间隔上限 (s)", multimode_cfg["hop_t_max_s"], 1e-4, 1000, 6),
                    ("hop_log_median", "对数正态中位数 (s)", multimode_cfg["hop_log_median_s"], 1e-5, 100, 6),
                    ("hop_log_sigma", "对数正态 σ", multimode_cfg["hop_log_sigma"], 0.001, 10, 6),
                ]:
                    add_double(form, controls, key, label, value, lo, hi, dec)
                add_integer(form, controls, "hop_points", "冲击间隔积分点数", multimode_cfg["hop_points"])
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            return scroll

        tsai_page = physical_page("Tsai2012", include_multimode=False)
        luong_page = physical_page("Luong2024", include_multimode=True)

        empirical_page = QWidget()
        empirical_form = QFormLayout(empirical_page)
        empirical = {}
        self.model_controls["Luong2026"] = empirical
        empirical_form.addRow(QLabel("<b>水力、粒径与频带</b>"))
        add_double(empirical_form, empirical, "D50", "中值粒径 D50 (m)", sediment_cfg["D50_m"], 1e-5, 5, 6)
        add_double(empirical_form, empirical, "rho_s", "颗粒密度 ρs (kg/m³)", sediment_cfg["rho_s_kg_m3"], 1001, 6000, 2)
        add_double(empirical_form, empirical, "rho_f", "流体密度 ρf (kg/m³)", sediment_cfg["rho_f_kg_m3"], 100, 2000, 2)
        add_double(empirical_form, empirical, "W", "有效河宽 W (m)", site_cfg["channel_width_m"], 0.01, 10000, 3)
        add_double(empirical_form, empirical, "slope", "无量纲坡降 S=tan(θ)", site_cfg["slope"], 1e-7, 0.999, 8)
        add_double(empirical_form, empirical, "f_inv_min", "PSD频带下限 (Hz)", inversion_cfg["f_inv_min_hz"], 0.001, 10000, 3)
        add_double(empirical_form, empirical, "f_inv_max", "PSD频带上限 (Hz)", inversion_cfg["f_inv_max_hz"], 0.002, 10000, 3)
        add_integer(empirical_form, empirical, "n_frequency_bins", "PSD频率点数", inversion_cfg["n_frequency_bins"], 2, 2000)
        empirical_form.addRow(QLabel("<b>Luong 2026 式(13)–(15)</b>"))
        for key, label, value, lo, hi, dec in [
            ("k", "尺度系数 k", empirical_cfg["k"], 1e-8, 1e8, 6),
            ("m", "PSD修正指数 m", empirical_cfg["m"], 1e-6, 20, 6),
            ("n", "超额剪应力指数 n", empirical_cfg["n"], 1e-6, 20, 6),
            ("A_db", "参考PSD斜率 A (dB)", empirical_cfg["A_db"], -1000, 1000, 5),
            ("B_db", "参考PSD截距 B (dB)", empirical_cfg["B_db"], -1000, 1000, 5),
            ("tau_cr", "临界 Shields τcr*", empirical_cfg["tau_cr"], 0, 1, 6),
            ("hydraulic_radius_depth_ratio", "水力半径/水深 Rh/H", empirical_cfg["hydraulic_radius_depth_ratio"], 0.001, 10, 6),
        ]:
            add_double(empirical_form, empirical, key, label, value, lo, hi, dec)
        empirical_note = QLabel(
            "A、B、k、m、n为ADLP论文标定值；迁移至新场地时应优先采用同步PSD、"
            "水深和实测推移质数据重新标定。"
        )
        empirical_note.setWordWrap(True)
        empirical_form.addRow(empirical_note)
        empirical_scroll = QScrollArea()
        empirical_scroll.setWidgetResizable(True)
        empirical_scroll.setWidget(empirical_page)

        self.model_param_stack.addWidget(tsai_page)
        self.model_param_stack.addWidget(luong_page)
        self.model_param_stack.addWidget(empirical_scroll)
        self.model_page_index = {
            "Tsai2012": 0,
            "Luong2024": 1,
            "Luong2026": 2,
        }
        self.model_defaults = {
            model_key: {
                key: control.value()
                for key, control in controls.items()
            }
            for model_key, controls in self.model_controls.items()
        }

        content = QHBoxLayout()
        group_params = QGroupBox("2. 当前模型参数")
        params_layout = QVBoxLayout(group_params)
        params_actions = QHBoxLayout()
        params_actions.addStretch()
        self.btn_restore_model_defaults = QPushButton("恢复当前模型默认参数")
        self.btn_restore_model_defaults.setToolTip(
            "将当前模型页面的全部参数恢复为配置文件中的默认值"
        )
        self.btn_restore_model_defaults.clicked.connect(
            self.restore_current_model_defaults
        )
        params_actions.addWidget(self.btn_restore_model_defaults)
        params_layout.addLayout(params_actions)
        params_layout.addWidget(self.model_param_stack)
        group_params.setMinimumWidth(455)
        content.addWidget(group_params, 5)

        group_schematic = QGroupBox("3. 物理机制分区图与关键公式")
        schematic_layout = QVBoxLayout(group_schematic)
        self.schematic_widget = ModelSchematicWidget()
        schematic_layout.addWidget(self.schematic_widget)
        self.formula_canvas = FormulaCanvas(self)
        schematic_layout.addWidget(self.formula_canvas)
        content.addWidget(group_schematic, 7)
        layout.addLayout(content, 1)

        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        self.on_model_changed()
        self.tabs.addTab(tab, create_scientific_icon("settings"), "4. 模型参数")

    # =========================================================================
    # Tab 5: 反演与结果
    # =========================================================================
    def setup_tab_inversion(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_exec = QGroupBox("1. 床载反演计算")
        el = QVBoxLayout(group_exec)
        self.btn_run_inv = QPushButton("启动推移质通量 qb 反演")
        self.btn_run_inv.clicked.connect(self.on_run_inversion)
        el.addWidget(self.btn_run_inv)
        panel_layout.addWidget(group_exec)

        group_summary = QGroupBox("2. 推移质反演结果汇总")
        sl = QVBoxLayout(group_summary)
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        sl.addWidget(self.txt_summary)
        panel_layout.addWidget(group_summary)

        group_export = QGroupBox("3. 数据导出")
        exl = QVBoxLayout(group_export)
        self.btn_export_excel = QPushButton("导出 Excel / CSV 数据...")
        self.btn_export_excel.clicked.connect(self.on_export_excel)
        self.btn_export_img = QPushButton("导出高分辨率图像包 (PNG)...")
        self.btn_export_img.clicked.connect(self.on_export_images)
        exl.addWidget(self.btn_export_excel)
        exl.addWidget(self.btn_export_img)
        panel_layout.addWidget(group_export)

        layout.addWidget(self.wrap_in_scroll(panel))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_inv = Figure(figsize=(8, 6))
        self.canvas_inv = FigureCanvas(self.fig_inv)
        plot_layout.addWidget(NavigationToolbar(self.canvas_inv, self))
        plot_layout.addWidget(self.canvas_inv)
        layout.addWidget(plot_panel)

        self.tabs.addTab(tab, create_scientific_icon("inversion"), "5. 反演与结果")

    # =========================================================================
    # 绘图逻辑
    # =========================================================================

    def trace_plot_data(self, trace):
        """构造保尖峰的轻量绘图数据，不改变完整计算序列。"""

        values = np.asarray(trace.data)
        indices = minmax_decimation_indices(
            values, self.config["performance"]["plot_max_points"]
        )
        start_num = mdates.date2num(trace.stats.starttime.datetime)
        times = start_num + indices / (
            float(trace.stats.sampling_rate) * 86400.0
        )
        return times, values[indices]

    def plot_waveform_canvas(self):
        if self.st_raw_tr is None: return
        self.fig_wf.clear()
        ax1 = self.fig_wf.add_subplot(311)
        ax2 = self.fig_wf.add_subplot(312, sharex=ax1)
        ax3 = self.fig_wf.add_subplot(313, sharex=ax1)

        self.apply_canvas_style(self.fig_wf, [ax1, ax2, ax3])

        t_raw, y_raw = self.trace_plot_data(self.st_raw_tr)
        t_det, y_det = self.trace_plot_data(self.st_det_tr)
        t_vel, y_vel = self.trace_plot_data(self.st_vel_tr)
        ax1.plot(t_raw, y_raw, color='#f43f5e', lw=0.6, label='Raw Counts')
        ax1.legend(loc='upper right')
        ax1.set_ylabel('Counts')
        ax1.set_title(f"OBS 波形数据: {self.st_raw_tr.stats.starttime} -> {self.st_raw_tr.stats.endtime}")

        ax2.plot(t_det, y_det, color='#fb923c', lw=0.6, label='Detrend & Demean')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Counts')

        ax3.plot(t_vel, y_vel, color='#4ade80', lw=0.6, label='Velocity (m/s)')
        ax3.legend(loc='upper right')
        ax3.set_ylabel('m/s')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

        self.fig_wf.tight_layout()
        self.canvas_wf.draw_idle()

    def plot_river_canvas(self):
        if self.river_series_utc is None: return
        self.fig_river.clear()
        ax = self.fig_river.add_subplot(111)
        self.apply_canvas_style(self.fig_river, ax)

        ax.plot(self.river_series_utc.index, self.river_series_utc.values, color='#38bdf8', lw=1.2, label='Water Depth H (m)')
        ax.set_ylabel('Water Depth H (m)')
        ax.set_xlabel('Time (UTC)')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

        self.fig_river.tight_layout()
        self.canvas_river.draw_idle()

    def plot_psd_canvas(self):
        """绘制 PSD 时频谱；显示层清理无效针点，并可叠加水深。"""

        if self.df_psd is None or self.df_psd.empty:
            return

        from matplotlib import colormaps

        df_psd = self.df_psd.copy()
        df_psd.index = pd.DatetimeIndex(df_psd.index)
        df_psd = df_psd.sort_index()

        frequency_all = np.asarray(
            pd.to_numeric(df_psd.columns, errors="coerce"),
            dtype=float,
        )
        valid_frequency = np.isfinite(frequency_all)
        if not valid_frequency.any():
            return
        df_psd = df_psd.iloc[:, valid_frequency]
        frequency_all = frequency_all[valid_frequency]
        order = np.argsort(frequency_all)
        freqs = frequency_all[order]
        psd_linear = df_psd.iloc[:, order].to_numpy(dtype=float)
        if psd_linear.size == 0:
            return

        psd_index = pd.DatetimeIndex(df_psd.index)
        time_num = mdates.date2num(psd_index.to_pydatetime())

        # 只生成显示副本：零、负值与 Inf 不强制写成 -300 dB。
        positive = np.isfinite(psd_linear) & (psd_linear > 0.0)
        p_db = np.full(psd_linear.T.shape, np.nan, dtype=float)
        p_db[positive.T] = 10.0 * np.log10(psd_linear.T[positive.T])

        # 仅修补被两个有效邻点夹住的孤立空像素，不跨越真实缺测段。
        p_show = p_db.copy()
        if p_show.shape[1] >= 3:
            center = p_show[:, 1:-1]
            left = p_show[:, :-2]
            right = p_show[:, 2:]
            isolated = (
                ~np.isfinite(center)
                & np.isfinite(left)
                & np.isfinite(right)
            )
            center[isolated] = (0.5 * (left + right))[isolated]
        if p_show.shape[0] >= 3:
            center = p_show[1:-1, :]
            lower = p_show[:-2, :]
            upper = p_show[2:, :]
            isolated = (
                ~np.isfinite(center)
                & np.isfinite(lower)
                & np.isfinite(upper)
            )
            center[isolated] = (0.5 * (lower + upper))[isolated]

        finite_db = p_show[np.isfinite(p_show)]
        if finite_db.size:
            vmin, vmax = np.percentile(finite_db, [5.0, 95.0])
            if not np.isfinite(vmin):
                vmin = -200.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = -200.0, -100.0

        # 极端低值只在显示层截断，原始 PSD 与反演输入保持不变。
        p_show = np.clip(p_show, vmin, vmax)
        p_masked = np.ma.masked_invalid(p_show)
        cmap = colormaps["viridis"].copy()
        cmap.set_bad(cmap(0.0))

        self.fig_psd.clear()
        grid = self.fig_psd.add_gridspec(
            2, 1, height_ratios=(0.08, 0.92), hspace=0.20
        )
        cax = self.fig_psd.add_subplot(grid[0, 0])
        ax1 = self.fig_psd.add_subplot(grid[1, 0])
        self.apply_canvas_style(self.fig_psd, ax1)

        ax1.grid(False)

        if len(time_num) > 1:
            time_steps = np.diff(time_num)
            time_steps = time_steps[
                np.isfinite(time_steps) & (time_steps > 0.0)
            ]
            half_time_step = (
                0.5 * np.median(time_steps)
                if time_steps.size
                else 0.5 / 86400.0
            )
        else:
            half_time_step = 0.5 / 86400.0
        t0_num = float(time_num[0] - half_time_step)
        t1_num = float(time_num[-1] + half_time_step)
        if t1_num <= t0_num:
            t1_num = t0_num + 1.0 / 86400.0

        if len(freqs) > 1:
            frequency_steps = np.diff(freqs)
            frequency_steps = frequency_steps[
                np.isfinite(frequency_steps) & (frequency_steps > 0.0)
            ]
            half_frequency_step = (
                0.5 * np.median(frequency_steps)
                if frequency_steps.size
                else 0.5
            )
        else:
            half_frequency_step = 0.5
        f0 = max(0.0, float(freqs[0] - half_frequency_step))
        f1 = float(freqs[-1] + half_frequency_step)

        im = ax1.imshow(
            p_masked,
            aspect="auto",
            origin="lower",
            extent=[t0_num, t1_num, f0, f1],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation=(
                "hanning"
                if min(p_masked.shape) >= 3
                else "nearest"
            ),
            resample=True,
        )
        ax1.set_xlim(t0_num, t1_num)
        ax1.set_ylim(f0, f1)
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_xlabel("Time (UTC)")
        ax1.xaxis_date()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

        cbar = self.fig_psd.colorbar(im, cax=cax, orientation="horizontal")
        txt_col = '#f8fafc' if self.is_dark_theme else '#0f172a'
        cbar.set_label("PSD (dB rel. (m/s)²/Hz)", color=txt_col)
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(axis="x", colors=txt_col, labelsize=8)
        cax.set_facecolor(
            '#1e293b' if self.is_dark_theme else '#ffffff'
        )

        show_depth = (
            self.river_series_utc is not None
            and not self.river_series_utc.empty
            and self.chk_show_depth_psd.isChecked()
        )
        if show_depth:
            river = self.river_series_utc.copy()
            if isinstance(river, pd.DataFrame):
                river = river.iloc[:, 0] if river.shape[1] else None
            if river is not None:
                river.index = pd.DatetimeIndex(river.index)
                river = pd.to_numeric(river, errors="coerce")
                river = river.replace([np.inf, -np.inf], np.nan).dropna()
                river = river[~river.index.isna()]
                if river.index.has_duplicates:
                    river = river.groupby(level=0).mean()
                river = river.sort_index()

            if river is not None and not river.empty:
                river_index = pd.DatetimeIndex(river.index)
                if psd_index.tz is not None and river_index.tz is None:
                    river.index = river_index.tz_localize("UTC").tz_convert(
                        psd_index.tz
                    )
                elif psd_index.tz is None and river_index.tz is not None:
                    river.index = (
                        river_index.tz_convert("UTC").tz_localize(None)
                    )
                elif psd_index.tz is not None and river_index.tz is not None:
                    river.index = river_index.tz_convert(psd_index.tz)

                if len(river) == 1:
                    r_on_psd = pd.Series(
                        float(river.iloc[0]),
                        index=psd_index,
                        dtype=float,
                    )
                else:
                    combined_index = (
                        pd.DatetimeIndex(river.index)
                        .union(psd_index)
                        .sort_values()
                    )
                    r_on_psd = (
                        river.reindex(combined_index)
                        .interpolate(
                            method="time",
                            limit_direction="both",
                        )
                        .reindex(psd_index)
                    )

                depth_values = r_on_psd.to_numpy(dtype=float)
                valid_depth = np.isfinite(depth_values)
                if valid_depth.any():
                    ax2 = ax1.twinx()
                    self.apply_canvas_style(self.fig_psd, ax2)
                    ax2.grid(False)
                    ax2.plot(
                        time_num[valid_depth],
                        depth_values[valid_depth],
                        color="#38bdf8",
                        linewidth=1.8,
                        alpha=0.95,
                        zorder=10,
                        label="River Depth H (m)",
                    )
                    ax2.set_xlim(t0_num, t1_num)
                    ax2.set_ylabel(
                        "River Depth H (m)",
                        color="#38bdf8",
                        labelpad=10,
                    )
                    ax2.tick_params(axis="y", colors="#38bdf8")
                    ax2.spines["right"].set_color("#38bdf8")
                    legend = ax2.legend(
                        loc="upper left",
                        frameon=True,
                        fontsize=8,
                    )
                    if legend is not None:
                        legend.get_frame().set_alpha(0.80)

        self.fig_psd.subplots_adjust(
            left=0.075, right=0.88, top=0.90, bottom=0.14
        )
        self.canvas_psd.draw_idle()

    def plot_inv_canvas(self):
        if self.df_result is None: return
        self.fig_inv.clear()
        ax1 = self.fig_inv.add_subplot(211)
        ax2 = self.fig_inv.add_subplot(212, sharex=ax1)

        self.apply_canvas_style(self.fig_inv, [ax1, ax2])

        ax1.plot(self.df_result.index, self.df_result['river_depth_m'], color='#38bdf8', lw=1.2)
        ax1.set_ylabel('River Depth H (m)')
        ax1.set_title("反演结果: 水深 H(t) 与 推移质通量 Qb(t)")

        ax2.plot(self.df_result.index, self.df_result['Qb_mass_kg_s'], color='#4ade80', lw=1.0)
        ax2.set_ylabel('Bedload Qb (kg/s)')
        ax2.set_xlabel('Time (UTC)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

        self.fig_inv.tight_layout()
        self.canvas_inv.draw_idle()

    # =========================================================================
    # 事件处理 Handlers (绑定实时进度响应)
    # =========================================================================

    def on_load_waveform(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择一个或多个波形文件",
            "",
            "Seismic waveforms (*.BHZ *.bhz *.mseed *.MSEED *.miniseed "
            "*.msd *.MSD *.sac *.SAC);;All Files (*)",
        )
        if files:
            self.wf_files = sorted(files)
            self.waveform_headers = []
            self.btn_process_wf.setEnabled(False)
            self.dt_wf_start.setEnabled(False)
            self.dt_wf_end.setEnabled(False)
            self.table_wf_headers.setRowCount(0)
            self.txt_wf_coverage.clear()
            self.lbl_wf_info.setText(
                f"已选择 {len(files)} 个文件，正在仅读取文件头；"
                "波形样本尚未加载。"
            )
            self.header_worker = HeaderScanWorker(self.wf_files)
            self.header_worker.progress_sig.connect(self.on_progress_update)
            self.header_worker.finished_sig.connect(
                self.on_waveform_headers_scanned
            )
            self.header_worker.error_sig.connect(self.on_worker_error)
            self.header_worker.start()

    def on_waveform_headers_scanned(self, records):
        """展示逐文件头与合并覆盖段，并初始化安全的目标时段。"""

        self.waveform_headers = list(records)
        self.table_wf_headers.setRowCount(len(self.waveform_headers))
        for row, record in enumerate(self.waveform_headers):
            values = [
                os.path.basename(record.path),
                record.trace_id,
                record.start_utc.isoformat(),
                record.end_utc.isoformat(),
                f"{record.sampling_rate_hz:g}",
                f"{record.npts:,}",
            ]
            for column, value in enumerate(values):
                self.table_wf_headers.setItem(
                    row, column, QTableWidgetItem(value)
                )

        intervals = merge_coverage_intervals(self.waveform_headers)
        coverage_lines = []

        def format_utc(timestamp):
            text = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
            return f"{text.rstrip('0').rstrip('.')} UTC"

        def format_duration(seconds):
            milliseconds = int(round(max(0.0, seconds) * 1000.0))
            days, milliseconds = divmod(milliseconds, 86_400_000)
            hours, milliseconds = divmod(milliseconds, 3_600_000)
            minutes, milliseconds = divmod(milliseconds, 60_000)
            whole_seconds, milliseconds = divmod(milliseconds, 1000)
            return (
                f"{days} d  {hours:02d} h  {minutes:02d} min  "
                f"{whole_seconds:02d}.{milliseconds:03d} s"
            )

        for index, interval in enumerate(intervals, start=1):
            duration_s = (
                interval.end_utc - interval.start_utc
            ).total_seconds()
            coverage_lines.extend(
                [
                    f"[{index:02d}]  Trace ID : {interval.trace_id}",
                    f"      Start UTC: {format_utc(interval.start_utc)}",
                    f"      End UTC  : {format_utc(interval.end_utc)}",
                    f"      Duration : {format_duration(duration_s)}",
                    f"      Files    : {interval.file_count}",
                    "",
                ]
            )
        self.txt_wf_coverage.setPlainText(
            "\n".join(coverage_lines).rstrip()
        )

        first = intervals[0]
        default_hours = float(
            self.config["waveform"]["default_selection_hours"]
        )
        default_end = min(
            first.end_utc,
            first.start_utc + pd.to_timedelta(default_hours, unit="h"),
        )
        global_start = min(record.start_utc for record in self.waveform_headers)
        global_end = max(record.end_utc for record in self.waveform_headers)
        q_min = QDateTime.fromMSecsSinceEpoch(
            int(global_start.timestamp() * 1000), Qt.TimeSpec.UTC
        )
        q_max = QDateTime.fromMSecsSinceEpoch(
            int(global_end.timestamp() * 1000), Qt.TimeSpec.UTC
        )
        for editor in (self.dt_wf_start, self.dt_wf_end):
            editor.setMinimumDateTime(q_min)
            editor.setMaximumDateTime(q_max)
            editor.setEnabled(True)
        self.dt_wf_start.setDateTime(
            QDateTime.fromMSecsSinceEpoch(
                int(first.start_utc.timestamp() * 1000), Qt.TimeSpec.UTC
            )
        )
        self.dt_wf_end.setDateTime(
            QDateTime.fromMSecsSinceEpoch(
                int(default_end.timestamp() * 1000), Qt.TimeSpec.UTC
            )
        )
        self.btn_process_wf.setEnabled(True)
        self.lbl_wf_info.setText(
            f"文件头扫描完成：{len(self.wf_files)} 个文件，"
            f"{len(self.waveform_headers)} 条 Trace，"
            f"{len(intervals)} 个连续覆盖段。波形样本仍未加载。"
        )
        self.progress_bar.setVisible(False)
        self.update_waveform_selection_label()

    def update_waveform_selection_label(self):
        """显示目标区间时长，提醒用户当前尚未执行读取。"""

        if not hasattr(self, "dt_wf_start"):
            return
        start_ms = self.dt_wf_start.dateTime().toMSecsSinceEpoch()
        end_ms = self.dt_wf_end.dateTime().toMSecsSinceEpoch()
        duration_h = (end_ms - start_ms) / 3_600_000.0
        if duration_h <= 0.0:
            self.lbl_wf_selection.setText("区间无效：结束时间必须晚于开始时间。")
        else:
            self.lbl_wf_selection.setText(
                f"目标时长 {duration_h:.3f} h；点击执行后才读取并可视化。"
            )

    def on_process_waveform(self):
        if not self.waveform_headers:
            QMessageBox.warning(self, "警告", "请先完成波形文件头扫描！")
            return

        start_ms = self.dt_wf_start.dateTime().toMSecsSinceEpoch()
        end_ms = self.dt_wf_end.dateTime().toMSecsSinceEpoch()
        if start_ms >= end_ms:
            QMessageBox.warning(
                self, "目标时段错误", "结束时间必须晚于开始时间。"
            )
            return
        start_utc = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        end_utc = pd.Timestamp(end_ms, unit="ms", tz="UTC")

        gain = self.spin_gain.value()
        adc = self.spin_adc.value()
        paz_cfg = self.config["waveform"]["manual_paz"]
        paz = {
            "poles": [complex(value) for value in paz_cfg["poles"]],
            "zeros": [complex(value) for value in paz_cfg["zeros"]],
            "gain": gain,
            "sensitivity": gain * adc,
        }
        try:
            pf_vals = [
                float(value.strip()) for value in self.txt_prefilt.text().split(",")
            ]
        except ValueError:
            QMessageBox.warning(
                self, "预滤波参数错误", "请输入四个以英文逗号分隔的数值。"
            )
            return
        if len(pf_vals) != 4:
            QMessageBox.warning(
                self, "预滤波参数错误", "预滤波必须恰好包含四个频率。"
            )
            return

        self.wf_worker = WaveformWorker(
            self.waveform_headers,
            start_utc,
            end_utc,
            paz,
            pf_vals,
            self.config["waveform"]["maximum_interpolated_gap_s"],
        )
        self.wf_worker.progress_sig.connect(self.on_progress_update)
        self.wf_worker.finished_sig.connect(self.on_waveform_processed)
        self.wf_worker.error_sig.connect(self.on_worker_error)
        self.wf_worker.start()

    def on_waveform_processed(self, tr_raw, tr_det, tr_vel):
        self.st_raw_tr = tr_raw; self.st_det_tr = tr_det; self.st_vel_tr = tr_vel
        self.df_psd = None
        self.psd_metadata = None
        self.df_result = None
        self.fig_psd.clear()
        self.canvas_psd.draw_idle()
        self.fig_inv.clear()
        self.canvas_inv.draw_idle()
        self.plot_waveform_canvas()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(
            "目标时段波形预处理完成；可继续输入水深并计算 PSD。"
        )

    def on_load_river(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择水位文件", "", "Data Files (*.xlsx *.xls *.csv)")
        if file:
            if file.endswith(".csv"):
                self.df_river = pd.read_csv(file)
            else:
                self.df_river = pd.read_excel(file)
                
            self.river_file = file
            self.status_bar.showMessage(f"已加载水位: {os.path.basename(file)}")

            self.table_river.setRowCount(min(15, len(self.df_river)))
            self.table_river.setColumnCount(len(self.df_river.columns))
            self.table_river.setHorizontalHeaderLabels(self.df_river.columns.astype(str))

            for r in range(min(15, len(self.df_river))):
                for c in range(len(self.df_river.columns)):
                    val = str(self.df_river.iloc[r, c])
                    self.table_river.setItem(r, c, QTableWidgetItem(val))

    def on_apply_river(self):
        if self.df_river is None:
            QMessageBox.warning(self, "警告", "请先加载水位数据！")
            return

        bed_elev = self.spin_bed_elev.value()
        tz_choice = self.combo_tz.currentText()

        t_col = self.df_river.columns[0]
        h_col = self.df_river.columns[1]

        t = pd.to_datetime(self.df_river[t_col], errors="coerce")
        h = pd.to_numeric(self.df_river[h_col], errors="coerce")
        depth = h - bed_elev

        timezone_name = tz_choice.split()[0]
        if t.dt.tz is None:
            t = t.dt.tz_localize(
                timezone_name, ambiguous="NaT", nonexistent="shift_forward"
            )
        else:
            t = t.dt.tz_convert(timezone_name)
        t = t.dt.tz_convert("UTC")

        valid = t.notna() & depth.notna()
        s_river = pd.Series(
            depth.loc[valid].to_numpy(dtype=float),
            index=pd.DatetimeIndex(t.loc[valid]),
        )
        s_river = s_river.groupby(level=0).mean().sort_index()
        if s_river.empty:
            QMessageBox.warning(self, "水深数据错误", "没有可用的时间—水位数据。")
            return
        self.river_series_utc = s_river
        self.plot_river_canvas()
        self.status_bar.showMessage("水位转水深处理完成！")

    def on_calc_psd(self):
        if self.st_vel_tr is None:
            QMessageBox.warning(self, "警告", "请先进行波形预处理！")
            return

        nperseg = int(self.combo_nperseg.currentText().split()[0])
        fmin = self.spin_psd_fmin.value()
        fmax = self.spin_psd_fmax.value()

        self.psd_worker = PSDWorker(
            self.st_vel_tr,
            nperseg,
            fmin,
            fmax,
            self.config["psd"]["overlap_fraction"],
            self.config["psd"]["aggregation"],
            self.config["performance"]["psd_chunk_blocks"],
        )
        self.psd_worker.progress_sig.connect(self.on_progress_update)
        self.psd_worker.finished_sig.connect(self.on_psd_computed)
        self.psd_worker.error_sig.connect(self.on_worker_error)
        self.psd_worker.start()

    def on_psd_computed(self, df_psd, metadata):
        self.df_psd = df_psd
        self.psd_metadata = metadata
        self.plot_psd_canvas()
        self.status_bar.showMessage(
            "PSD完成："
            f"{metadata.nperseg}点窗，"
            f"{metadata.noverlap / metadata.nperseg:.0%}重叠，"
            f"时间块约{metadata.nominal_block_seconds:.1f}s"
        )

    def on_model_changed(self):
        """同步模型参数页、机制图和 LaTeX 公式。"""

        if not hasattr(self, "combo_model"):
            return
        model_key = str(self.combo_model.currentData())
        self.model_param_stack.setCurrentIndex(
            self.model_page_index[model_key]
        )
        self.schematic_widget.set_model(model_key)
        self.formula_canvas.set_model(model_key)

    def current_model_values(self):
        """读取当前模型页面中实际可见且有效的参数。"""

        model_key = str(self.combo_model.currentData())
        values = {
            key: control.value()
            for key, control in self.model_controls[model_key].items()
        }
        return model_key, values

    def restore_current_model_defaults(self):
        """恢复当前模型参数页的配置文件默认值。"""

        model_key = str(self.combo_model.currentData())
        defaults = self.model_defaults.get(model_key, {})
        controls = self.model_controls.get(model_key, {})
        for key, value in defaults.items():
            control = controls.get(key)
            if control is None:
                continue
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self.status_bar.showMessage(
            f"{self.combo_model.currentText()} 参数已恢复为默认值。",
            3500,
        )

    def on_run_inversion(self):
        if self.df_psd is None:
            QMessageBox.warning(self, "警告", "请先计算 PSD 功率谱！")
            return
        if self.river_series_utc is None:
            QMessageBox.warning(self, "警告", "请先转换水深数据！")
            return

        model_key, values = self.current_model_values()
        sediment_cfg = self.config["sediment"]
        params = {
            "model_type": model_key,
            "D50": values["D50"],
            "D_min": (
                None if values.get("D_min", 0.0) <= 0.0
                else values["D_min"]
            ),
            "D_max": values.get("D_max"),
            "grain_log_std": values.get(
                "grain_log_std", sediment_cfg["grain_log_std"]
            ),
            "grain_points": int(
                values.get("grain_points", sediment_cfg["grain_points"])
            ),
            "W": values["W"],
            "slope": values["slope"],
            "r0": values.get("r0", self.config["site"]["source_distance_m"]),
            "tau_c50": values.get(
                "tau_c50", sediment_cfg["tau_c50"]
            ),
            "clip_tau_c": sediment_cfg.get(
                "clip_tau_c_tsai" if model_key == "Tsai2012" else "clip_tau_c_multimode",
                model_key == "Luong2024",
            ),
            "f_inv_min": values["f_inv_min"],
            "f_inv_max": values["f_inv_max"],
            "n_frequency_bins": int(values["n_frequency_bins"]),
            "sediment_params": {
                "rho_s": values["rho_s"],
                "rho_f": values["rho_f"],
            },
        }
        if model_key in {"Tsai2012", "Luong2024"}:
            params["seismic_params"] = {
                "v0": values["v0"],
                "z0": values["z0"],
                "f0": values["f0"],
                "a": values["a"],
                "Q0": values["Q0"],
                "eta": values["eta"],
                "vc0_override": (
                    None
                    if values["vc0_override"] <= 0.0
                    else values["vc0_override"]
                ),
                "zeta_override": (
                    None
                    if values["zeta_override"] < 0.0
                    else values["zeta_override"]
                ),
                "restitution": values.get("restitution", 0.0),
                "fz": values.get("fz", self.config["seismic"]["fz"]),
                "Nzz": values.get("Nzz", self.config["seismic"]["Nzz"]),
            }
        if model_key == "Luong2024":
            params.update(
                {
                    "hop_t_min": values["hop_t_min"],
                    "hop_t_max": values["hop_t_max"],
                    "hop_log_median": values["hop_log_median"],
                    "hop_log_sigma": values["hop_log_sigma"],
                    "hop_points": int(values["hop_points"]),
                }
            )
        if model_key == "Luong2026":
            params["empirical_params"] = {
                key: values[key]
                for key in (
                    "tau_cr",
                    "k",
                    "m",
                    "n",
                    "A_db",
                    "B_db",
                    "hydraulic_radius_depth_ratio",
                )
            }

        self.inv_worker = InversionWorker(self.df_psd, self.river_series_utc, params)
        self.inv_worker.progress_sig.connect(self.on_progress_update)
        self.inv_worker.finished_sig.connect(self.on_inversion_completed)
        self.inv_worker.error_sig.connect(self.on_worker_error)
        self.inv_worker.start()

    def on_inversion_completed(self, df_res):
        self.df_result = df_res
        model_key, model_values = self.current_model_values()

        Qb_s = df_res['Qb_mass_kg_s'].dropna()
        qb_s = df_res['qb_vol_m2_s'].dropna()
        
        valid_blocks = len(Qb_s)
        if valid_blocks >= 2:
            cadence_seconds = float(
                np.median(np.diff(Qb_s.index.asi8)) / 1.0e9
            )
            total_time_hours = (
                Qb_s.index[-1] - Qb_s.index[0]
            ).total_seconds() / 3600.0
        else:
            cadence_seconds = 0.0
            total_time_hours = 0.0

        active_mask = Qb_s > 0.01
        active_blocks = int(active_mask.sum())
        active_ratio = (
            active_blocks / valid_blocks * 100.0 if valid_blocks > 0 else 0.0
        )
        active_hours = active_blocks * cadence_seconds / 3600.0

        total_mass_kg = integrate_flux_mass(
            df_res.index, df_res["Qb_mass_kg_s"].to_numpy()
        )
        total_mass_tonnes = total_mass_kg / 1000.0
        total_vol_m3 = total_mass_kg / model_values["rho_s"]
        
        mean_Qb = Qb_s.mean()
        max_Qb = Qb_s.max()
        p95_Qb = np.percentile(Qb_s, 95) if len(Qb_s) > 0 else 0.0
        std_Qb = Qb_s.std()
        
        mean_qb_vol = qb_s.mean()
        max_qb_vol = qb_s.max()

        stats_text = (
            f"模型：{model_key}\n\n"
            "累积输移量\n"
            f"  • 估算总输沙质量：{total_mass_tonnes:.2f} t "
            f"（{total_mass_kg:.3e} kg）\n"
            f"  • 估算总输沙体积：{total_vol_m3:.3f} m³\n\n"
            "质量通量 Qb\n"
            f"  • 平均值：{mean_Qb:.5g} kg/s\n"
            f"  • 峰值：{max_Qb:.5g} kg/s\n"
            f"  • P95：{p95_Qb:.5g} kg/s\n"
            f"  • 标准差：{std_Qb:.5g} kg/s\n\n"
            "单宽体积通量 qb\n"
            f"  • 平均值：{mean_qb_vol:.4e} m²/s\n"
            f"  • 峰值：{max_qb_vol:.4e} m²/s\n\n"
            "时间覆盖\n"
            f"  • 有效跨度：{total_time_hours:.2f} h（{valid_blocks} 个PSD块）\n"
            f"  • 活跃估算时长：{active_hours:.2f} h\n"
            f"  • 活跃时间占比：{active_ratio:.1f}%\n\n"
            "主要参数快照\n"
            f"  • D50={model_values['D50']:.6g} m；"
            f"W={model_values['W']:.6g} m；"
            f"S={model_values['slope']:.8g}\n"
            f"  • ρs={model_values['rho_s']:.6g} kg/m³；"
            f"反演频带={model_values['f_inv_min']:.6g}–"
            f"{model_values['f_inv_max']:.6g} Hz\n"
        )
        self.txt_summary.setText(stats_text)
        self.plot_inv_canvas()

    def on_export_excel(self):
        if self.df_result is None:
            QMessageBox.warning(self, "警告", "尚无可导出的反演结果！")
            return
        file, _ = QFileDialog.getSaveFileName(self, "保存数据", "OBS_Bedload_Inversion.xlsx", "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if file:
            df_exp = self.df_result.copy()
            df_exp.index = df_exp.index.tz_localize(None)
            if file.endswith(".csv"):
                df_exp.to_csv(file)
            else:
                for column in df_exp.select_dtypes(include="datetimetz"):
                    df_exp[column] = df_exp[column].dt.tz_convert("UTC").dt.tz_localize(None)
                df_exp.to_excel(file)
            QMessageBox.information(self, "成功", f"结果已保存至:\n{file}")

    def on_export_images(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存图像文件夹")
        if folder:
            self.fig_wf.savefig(os.path.join(folder, "1_waveform_qc.png"), dpi=300, bbox_inches='tight')
            self.fig_river.savefig(os.path.join(folder, "2_river_depth.png"), dpi=300, bbox_inches='tight')
            self.fig_psd.savefig(os.path.join(folder, "3_psd_spectrogram.png"), dpi=300, bbox_inches='tight')
            self.fig_inv.savefig(os.path.join(folder, "4_inversion_results.png"), dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "成功", f"高分辨率图像包已导出至:\n{folder}")

    def on_worker_error(self, err_msg):
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("运行异常！")
        QMessageBox.critical(self, "运行异常", err_msg)

    def show_about(self):
        QMessageBox.about(
            self, "关于 OBS Analyzer",
            f"<h3>OBS Analyzer v{APP_VERSION}</h3>"
            "<p>河流推移质地震信号物理与经验反演系统。</p>"
            "<p>算法参考: Tsai et al. (2012)、Luong et al. (2024)、"
            "Luong et al. (2026) 与 "
            "seismic_bedload v0.2.0（GitHub）。</p>"
        )


# =============================================================================
# 6. v3.1.0 融合扩展：项目、实测级配、断面水力与动态反演
# =============================================================================


def read_tabular_file(path: str | Path) -> pd.DataFrame:
    """读取 CSV/TXT/Excel 表格，并尽量兼容中文工程数据。"""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".dat"}:
        attempts = [
            {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
            {"sep": None, "engine": "python", "encoding": "gb18030"},
            {"sep": "\\s+", "engine": "python", "encoding": "utf-8-sig"},
        ]
        errors = []
        for kwargs in attempts:
            try:
                frame = pd.read_csv(file_path, **kwargs)
                if not frame.empty or len(frame.columns) >= 1:
                    return frame
            except Exception as exc:
                errors.append(str(exc))
        raise ValueError("无法读取文本表格：" + "；".join(errors[-2:]))
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(file_path, engine="openpyxl")
    if suffix == ".xls":
        return pd.read_excel(file_path)
    raise ValueError(f"暂不支持该表格格式：{suffix}")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """清理千分位、空格和常见全角字符后转为数字。"""

    text = series.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("，", "", regex=False)
    text = text.str.replace("％", "%", regex=False)
    text = text.str.replace("%", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def detect_table_column(
    columns: Iterable[Any], candidates: Iterable[str], fallback_index: int
) -> Any:
    """按完全匹配、包含匹配、最后按位置回退识别列。"""

    column_list = list(columns)
    if not column_list:
        raise ValueError("表格没有可用列。")
    normalized = {str(column).strip().lower(): column for column in column_list}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    for column in column_list:
        text = str(column).strip().lower()
        if any(str(candidate).strip().lower() in text for candidate in candidates):
            return column
    if fallback_index < len(column_list):
        return column_list[fallback_index]
    raise ValueError("无法识别所需数据列。")


def parse_datetime_values(values: pd.Series) -> pd.Series:
    """兼容 Excel 日期、中文年月日和常规时间字符串。"""

    parsed = pd.to_datetime(values, errors="coerce")
    text = values.astype(str).str.strip()
    for source, target in (("年", "-"), ("月", "-"), ("日", " "), ("/", "-")):
        text = text.str.replace(source, target, regex=False)
    parsed_text = pd.to_datetime(text, errors="coerce")
    if parsed_text.notna().sum() > parsed.notna().sum():
        parsed = parsed_text
    numeric = pd.to_numeric(values, errors="coerce")
    parsed_excel = pd.to_datetime(
        numeric, unit="D", origin="1899-12-30", errors="coerce"
    )
    if parsed_excel.notna().sum() > parsed.notna().sum():
        parsed = parsed_excel
    return parsed


def ensure_utc_index(index_like: Any, assumed_timezone: str = "UTC") -> pd.DatetimeIndex:
    """将任意时间轴统一为带时区 UTC DatetimeIndex。"""

    parsed = pd.DatetimeIndex(pd.to_datetime(index_like, errors="coerce"))
    if parsed.tz is None:
        parsed = parsed.tz_localize(
            assumed_timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    return parsed.tz_convert("UTC")


def standardize_water_dataframe(
    frame: pd.DataFrame,
    assumed_timezone: str,
    depth_only: bool,
) -> pd.DataFrame:
    """标准化水位或水深文件，输出 UTC 索引和 water_value_m。"""

    if frame is None or frame.empty:
        raise ValueError("水位/水深表为空。")
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]
    time_column = detect_table_column(
        df.columns,
        ["time", "datetime", "date", "timestamp", "时间", "日期", "观测时间"],
        0,
    )
    candidates = (
        ["water_depth", "flowdepth", "depth", "h", "水深", "流深", "有效水深"]
        if depth_only
        else ["water_level", "stage", "level", "水位", "水位高程", "水面高程"]
    )
    value_column = detect_table_column(df.columns, candidates, 1)
    output = pd.DataFrame(
        {
            "time": parse_datetime_values(df[time_column]),
            "water_value_m": clean_numeric_series(df[value_column]),
        }
    ).dropna(subset=["time", "water_value_m"])

    # 无表头数据被 pandas 误作列名时，尝试补回第一行。
    if len(df.columns) >= 2:
        header_time = parse_datetime_values(pd.Series([df.columns[0]])).iloc[0]
        header_value = clean_numeric_series(pd.Series([df.columns[1]])).iloc[0]
        if pd.notna(header_time) and pd.notna(header_value):
            output = pd.concat(
                [
                    pd.DataFrame(
                        {"time": [header_time], "water_value_m": [header_value]}
                    ),
                    output,
                ],
                ignore_index=True,
            )

    if output.empty:
        raise ValueError(
            "水位/水深数据标准化后为空。请确认第一列为时间、第二列为数值，"
            "并检查是否存在说明行或合并单元格。"
        )
    utc = ensure_utc_index(output["time"], assumed_timezone)
    output.index = utc
    output.index.name = "time_utc"
    output = output.loc[~output.index.isna(), ["water_value_m"]]
    output = output.groupby(level=0).mean().sort_index()
    return output


def standardize_cross_section_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """标准化河道横断面为 x_m 与 bed_elevation_m。"""

    if frame is None or frame.empty:
        raise ValueError("河道断面表为空。")
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]
    x_column = detect_table_column(
        df.columns,
        ["x", "distance", "station", "offset", "起点距", "横向距离", "断面距离"],
        0,
    )
    z_column = detect_table_column(
        df.columns,
        ["bed_elevation", "elevation", "bed", "z", "河床高程", "河底高程", "床面高程"],
        1,
    )
    output = pd.DataFrame(
        {
            "x_m": clean_numeric_series(df[x_column]),
            "bed_elevation_m": clean_numeric_series(df[z_column]),
        }
    ).dropna()
    if len(df.columns) >= 2:
        header_x = clean_numeric_series(pd.Series([df.columns[0]])).iloc[0]
        header_z = clean_numeric_series(pd.Series([df.columns[1]])).iloc[0]
        if pd.notna(header_x) and pd.notna(header_z):
            output = pd.concat(
                [
                    pd.DataFrame(
                        {"x_m": [header_x], "bed_elevation_m": [header_z]}
                    ),
                    output,
                ],
                ignore_index=True,
            )
    output = (
        output.sort_values("x_m")
        .drop_duplicates(subset=["x_m"], keep="first")
        .reset_index(drop=True)
    )
    if len(output) < 2 or np.any(np.diff(output["x_m"].to_numpy()) <= 0.0):
        raise ValueError("河道断面至少需要两个横坐标严格递增的有效点。")
    return output


def mass_fraction_to_pdf(
    diameter_m: np.ndarray, fraction_mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把离散粒级概率质量转换为物理粒径空间 PDF。"""

    grains = np.asarray(diameter_m, dtype=float)
    mass = np.asarray(fraction_mass, dtype=float)
    valid = (
        np.isfinite(grains)
        & (grains > 0.0)
        & np.isfinite(mass)
        & (mass >= 0.0)
    )
    grains, mass = grains[valid], mass[valid]
    if grains.size < 2:
        raise ValueError("实测粒径分布至少需要两个有效粒径点。")
    order = np.argsort(grains)
    grains, mass = grains[order], mass[order]
    unique, inverse = np.unique(grains, return_inverse=True)
    if unique.size != grains.size:
        merged = np.zeros(unique.size, dtype=float)
        np.add.at(merged, inverse, mass)
        grains, mass = unique, merged
    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("粒径比例总和必须大于 0。")
    mass = mass / total
    edges = np.empty(grains.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (grains[:-1] + grains[1:])
    edges[0] = max(
        np.finfo(float).tiny,
        grains[0] - 0.5 * (grains[1] - grains[0]),
    )
    edges[-1] = grains[-1] + 0.5 * (grains[-1] - grains[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("粒径点必须严格递增。")
    density = mass / widths
    return grains, mass, normalize_pdf(grains, density, "实测粒径 PDF")


def weighted_quantile_value(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """计算离散质量权重分位粒径。"""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    values, weights = values[valid], weights[valid]
    if values.size == 0 or np.sum(weights) <= 0.0:
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    index = min(int(np.searchsorted(cumulative, quantile)), values.size - 1)
    return float(values[index])


def standardize_grain_dataframe(
    frame: pd.DataFrame, unit_mode: str = "auto"
) -> pd.DataFrame:
    """导入粒级质量比例或累计通过率，输出统一实测级配。"""

    if frame is None or frame.empty:
        raise ValueError("粒径分布表为空。")
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]
    diameter_column = detect_table_column(
        df.columns,
        [
            "d_mm", "diameter", "grain_size", "grain size", "particle_size",
            "size", "粒径", "粒径(mm)", "粒径（mm）", "粒径/m", "d",
        ],
        0,
    )
    fraction_column = detect_table_column(
        df.columns,
        [
            "fraction", "mass_fraction", "volume_fraction", "percentage",
            "percent", "含量", "质量百分比", "体积百分比", "频率", "累计", "通过率",
        ],
        1,
    )
    diameter_raw = clean_numeric_series(df[diameter_column])
    fraction_raw = clean_numeric_series(df[fraction_column])
    output = pd.DataFrame(
        {"diameter_raw": diameter_raw, "fraction_raw": fraction_raw}
    ).dropna()
    output = output[(output["diameter_raw"] > 0.0) & (output["fraction_raw"] >= 0.0)]
    if output.empty:
        raise ValueError("粒径分布标准化后为空。")

    diameter_name = str(diameter_column).lower()
    if unit_mode == "mm" or (unit_mode == "auto" and "mm" in diameter_name):
        diameter_m = output["diameter_raw"].to_numpy(dtype=float) / 1000.0
    elif unit_mode == "m":
        diameter_m = output["diameter_raw"].to_numpy(dtype=float)
    else:
        # 工程级配通常以 mm 输入；无单位且中位值大于 1 时按 mm 解释。
        raw = output["diameter_raw"].to_numpy(dtype=float)
        diameter_m = raw / 1000.0 if float(np.nanmedian(raw)) > 1.0 else raw

    fraction_text = str(fraction_column).lower()
    values = output["fraction_raw"].to_numpy(dtype=float)
    cumulative_hint = any(
        token in fraction_text
        for token in ("cum", "cumulative", "passing", "累计", "通过")
    )
    monotonic = values.size > 2 and np.all(np.diff(values) >= -1.0e-10)
    near_full = np.nanmax(values) > 0.90
    if cumulative_hint or (monotonic and near_full):
        cumulative = values.copy()
        if np.nanmax(cumulative) > 1.5:
            cumulative = cumulative / 100.0
        cumulative = np.clip(cumulative, 0.0, 1.0)
        order = np.argsort(diameter_m)
        diameter_m = diameter_m[order]
        cumulative = np.maximum.accumulate(cumulative[order])
        fractions = np.diff(np.concatenate([[0.0], cumulative]))
    else:
        fractions = values.copy()
        if np.nansum(fractions) > 1.5:
            fractions = fractions / 100.0

    grains, mass, density = mass_fraction_to_pdf(diameter_m, fractions)
    result = pd.DataFrame(
        {
            "D_m": grains,
            "D_mm": grains * 1000.0,
            "fraction": mass,
            "pdf_density_1_m": density,
        }
    )
    result["fraction_percent"] = result["fraction"] * 100.0
    result["cum_fraction"] = result["fraction"].cumsum()
    result["cum_percent"] = result["cum_fraction"] * 100.0
    return result


def build_grain_from_percentiles(
    d16_mm: float, d50_mm: float, d84_mm: float, bins: int = 48
) -> pd.DataFrame:
    """由 D16、D50、D84 构造对数正态离散级配。"""

    d16 = _require_positive("D16", d16_mm)
    d50 = _require_positive("D50", d50_mm)
    d84 = _require_positive("D84", d84_mm)
    if not d16 < d50 < d84:
        raise ValueError("代表粒径必须满足 D16 < D50 < D84。")
    sigma = 0.5 * (np.log(d84) - np.log(d16))
    mu = np.log(d50)
    edges = np.geomspace(max(d16 / 8.0, 1.0e-5), d84 * 8.0, max(16, int(bins)) + 1)
    distribution = lognorm(s=sigma, scale=np.exp(mu))
    fractions = np.diff(distribution.cdf(edges))
    centers_mm = np.sqrt(edges[:-1] * edges[1:])
    valid = fractions > 1.0e-10
    base = pd.DataFrame(
        {"D_mm": centers_mm[valid], "fraction": fractions[valid]}
    )
    base["fraction"] /= base["fraction"].sum()
    grains, mass, density = mass_fraction_to_pdf(
        base["D_mm"].to_numpy(dtype=float) / 1000.0,
        base["fraction"].to_numpy(dtype=float),
    )
    result = pd.DataFrame(
        {
            "D_m": grains,
            "D_mm": grains * 1000.0,
            "fraction": mass,
            "pdf_density_1_m": density,
        }
    )
    result["fraction_percent"] = result["fraction"] * 100.0
    result["cum_fraction"] = result["fraction"].cumsum()
    result["cum_percent"] = result["cum_fraction"] * 100.0
    return result


def compute_node_widths(x: np.ndarray) -> np.ndarray:
    """计算不规则断面节点代表宽度。"""

    coordinates = np.asarray(x, dtype=float)
    if coordinates.size < 2:
        return np.zeros_like(coordinates)
    widths = np.zeros_like(coordinates)
    widths[0] = 0.5 * (coordinates[1] - coordinates[0])
    widths[-1] = 0.5 * (coordinates[-1] - coordinates[-2])
    if coordinates.size > 2:
        widths[1:-1] = 0.5 * (coordinates[2:] - coordinates[:-2])
    widths[widths < 0.0] = 0.0
    return widths


def compute_segment_geometry(
    x: np.ndarray, z: np.ndarray, water_level: float
) -> tuple[float, float, float, float, float]:
    """按线性断面段计算水面宽、面积、湿周、水力半径和平均水深。"""

    wetted_width = wetted_area = wetted_perimeter = 0.0
    for index in range(len(x) - 1):
        x1, x2 = float(x[index]), float(x[index + 1])
        z1, z2 = float(z[index]), float(z[index + 1])
        dx, dz = x2 - x1, z2 - z1
        if dx <= 0.0:
            continue
        h1, h2 = water_level - z1, water_level - z2
        if h1 <= 0.0 and h2 <= 0.0:
            continue
        if h1 > 0.0 and h2 > 0.0:
            wet_dx, wet_dz = dx, dz
            area = 0.5 * (h1 + h2) * dx
        elif h1 > 0.0:
            ratio = h1 / (h1 - h2)
            wet_dx, wet_dz = ratio * dx, ratio * dz
            area = 0.5 * h1 * wet_dx
        else:
            ratio = (-h1) / (h2 - h1)
            wet_dx, wet_dz = (1.0 - ratio) * dx, (1.0 - ratio) * dz
            area = 0.5 * h2 * wet_dx
        wetted_width += wet_dx
        wetted_area += area
        wetted_perimeter += float(np.hypot(wet_dx, wet_dz))
    hydraulic_radius = wetted_area / wetted_perimeter if wetted_perimeter > 0.0 else 0.0
    mean_depth = wetted_area / wetted_width if wetted_width > 0.0 else 0.0
    return wetted_width, wetted_area, wetted_perimeter, hydraulic_radius, mean_depth


class SettlingVelocityLookup:
    """用 Ferguson–Church 型关系建立沉降速度反算粒径查找表。"""

    def __init__(self, relative_density: float, g: float, viscosity: float):
        self.relative_density = _require_positive("相对水下密度 Rs", relative_density)
        self.g = _require_positive("重力加速度 g", g)
        self.viscosity = _require_positive("运动黏度 ν", viscosity)
        self.diameter_grid = np.geomspace(1.0e-7, 2.0, 5000)
        self.velocity_grid = self.velocity(self.diameter_grid)
        order = np.argsort(self.velocity_grid)
        velocity = self.velocity_grid[order]
        diameter = self.diameter_grid[order]
        keep = np.concatenate([[True], np.diff(velocity) > 0.0])
        self.velocity_grid = velocity[keep]
        self.diameter_grid = diameter[keep]

    def velocity(self, diameter_m: np.ndarray) -> np.ndarray:
        diameter = np.asarray(diameter_m, dtype=float)
        safe = np.maximum(diameter, 1.0e-12)
        denominator = 18.0 * self.viscosity + np.sqrt(
            0.75 * self.relative_density * self.g * safe**3
        )
        velocity = self.relative_density * self.g * safe**2 / denominator
        return np.where(diameter > 0.0, velocity, 0.0)

    def inverse(self, target_velocity: np.ndarray | float) -> np.ndarray | float:
        target = np.maximum(np.asarray(target_velocity, dtype=float), 0.0)
        diameter = np.interp(
            target,
            self.velocity_grid,
            self.diameter_grid,
            left=0.0,
            right=self.diameter_grid[-1],
        )
        return float(diameter) if np.isscalar(target_velocity) else diameter


def local_activity_weights(
    depth: np.ndarray,
    settings: Mapping[str, float],
    settling: SettlingVelocityLookup,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算断面节点活动权重及局部有效粒径范围。"""

    h = np.asarray(depth, dtype=float)
    weights = np.zeros_like(h)
    d_min = np.full_like(h, np.nan)
    d_max = np.full_like(h, np.nan)
    wet = h > 0.0
    if not np.any(wet):
        return weights, d_min, d_max
    slope = float(settings["slope"])
    relative_density = float(settings["relative_density"])
    theta_c = float(settings["theta_c"])
    kappa = float(settings["kappa"])
    rouse_threshold = float(settings["rouse_threshold"])
    beta = float(settings["transport_exponent"])
    g = float(settings["g"])
    h_wet = h[wet]
    d_max_wet = h_wet * slope / (relative_density * theta_c)
    u_star = np.sqrt(g * h_wet * slope)
    d_min_wet = settling.inverse(rouse_threshold * kappa * u_star)
    d_min[wet], d_max[wet] = d_min_wet, d_max_wet
    valid = (
        np.isfinite(d_min_wet)
        & np.isfinite(d_max_wet)
        & (d_min_wet > 0.0)
        & (d_max_wet > d_min_wet)
    )
    if not np.any(valid):
        return weights, d_min, d_max
    h_valid = h_wet[valid]
    min_valid, max_valid = d_min_wet[valid], d_max_wet[valid]
    samples = 24
    q = np.linspace(0.0, 1.0, samples)
    diameter = min_valid[:, None] * (max_valid / min_valid)[:, None] ** q[None, :]
    shields = h_valid[:, None] * slope / (relative_density * diameter)
    integrand = np.maximum(shields - theta_c, 0.0) ** beta
    dlog = (np.log(max_valid) - np.log(min_valid)) / (samples - 1)
    integrated = (
        0.5 * integrand[:, 0]
        + np.sum(integrand[:, 1:-1], axis=1)
        + 0.5 * integrand[:, -1]
    ) * dlog
    wet_indices = np.flatnonzero(wet)
    weights[wet_indices[valid]] = integrated
    return weights, d_min, d_max


def representative_effective_grain_range(
    h_eff: float,
    settings: Mapping[str, float],
    settling: SettlingVelocityLookup,
) -> tuple[float, float]:
    """由代表性 H_eff 计算悬移下限和起动上限粒径。"""

    if not np.isfinite(h_eff) or h_eff <= 0.0:
        return float("nan"), float("nan")
    slope = float(settings["slope"])
    relative_density = float(settings["relative_density"])
    theta_c = float(settings["theta_c"])
    d_max = h_eff * slope / (relative_density * theta_c)
    u_star = np.sqrt(float(settings["g"]) * h_eff * slope)
    d_min = settling.inverse(
        float(settings["rouse_threshold"])
        * float(settings["kappa"])
        * u_star
    )
    if d_max <= d_min:
        return float("nan"), float("nan")
    return float(d_min), float(d_max)


def compute_hydraulic_timeseries(
    water: pd.DataFrame,
    cross_section: pd.DataFrame | None,
    settings: Mapping[str, Any],
    progress_callback=None,
) -> pd.DataFrame:
    """
    计算 H_eff、W_eff、断面几何量和有效运动粒径范围。

    输入方式：
    ``depth``       水深序列 + 固定宽度；
    ``stage_fixed`` 水位高程 - 参考河床高程 + 固定宽度；
    ``cross``       水位高程 + 实测横断面。

    在 ``depth`` 模式下可启用水位高程自动识别：当输入值明显超过合理水深，
    且减去参考河床高程后落入合理范围时，自动按 ``stage_fixed`` 处理。
    横断面活动权重无解时回退到几何平均水深和湿水宽度，避免水力模块把
    整段数据错误置为零；输移阈值仍由后续 Tsai/Luong 模型独立判断。
    """

    if water is None or water.empty:
        raise ValueError("水位/水深数据为空。")
    requested_mode = str(
        settings.get(
            "input_mode",
            "depth" if bool(settings.get("depth_only", False)) else "cross",
        )
    )
    if requested_mode not in {"depth", "stage_fixed", "cross"}:
        raise ValueError(f"未知水文输入模式：{requested_mode}")

    slope = _require_positive("河道坡降 S", settings["slope"])
    rho_s = _require_positive("颗粒密度 rho_s", settings["rho_s"])
    rho_f = _require_positive("流体密度 rho_f", settings["rho_f"])
    if rho_s <= rho_f:
        raise ValueError("颗粒密度 rho_s 必须大于流体密度 rho_f。")
    default_width = _require_positive("固定/回退有效宽度 W", settings["default_width"])
    theta_c = _require_positive("临界 Shields 参数", settings["theta_c"])
    active_ratio = float(settings["active_threshold_percent"]) / 100.0
    if not 0.0 <= active_ratio <= 1.0:
        raise ValueError("活动宽度阈值百分比必须在 0–100% 内。")
    bed_elevation = float(settings.get("bed_elevation_m", 0.0))
    if not np.isfinite(bed_elevation):
        raise ValueError("参考河床高程必须为有限数。")
    plausible_depth = _require_positive(
        "最大合理水深", settings.get("max_plausible_depth_m", 50.0)
    )
    auto_detect = bool(settings.get("auto_detect_stage", True))

    calculation = {
        "slope": slope,
        "relative_density": (rho_s - rho_f) / rho_f,
        "theta_c": theta_c,
        "kappa": _require_positive("von Kármán 常数", settings["kappa"]),
        "rouse_threshold": _require_positive("Rouse 阈值", settings["rouse_threshold"]),
        "transport_exponent": _require_positive("活动权重指数", settings["transport_exponent"]),
        "g": _require_positive("重力加速度", settings.get("g", 9.81)),
    }
    settling = SettlingVelocityLookup(
        calculation["relative_density"],
        calculation["g"],
        _require_positive("运动黏度", settings.get("kinematic_viscosity", 1.0e-6)),
    )

    effective_mode = requested_mode
    auto_converted = False
    values_all = water["water_value_m"].to_numpy(dtype=float)
    finite_values = values_all[np.isfinite(values_all)]
    if requested_mode == "depth" and auto_detect and finite_values.size:
        median_value = float(np.nanmedian(finite_values))
        converted = finite_values - bed_elevation
        positive_converted = converted[np.isfinite(converted) & (converted >= 0.0)]
        converted_median = (
            float(np.nanmedian(positive_converted)) if positive_converted.size else np.nan
        )
        if (
            median_value > plausible_depth
            and np.isfinite(converted_median)
            and 0.0 <= converted_median <= plausible_depth
        ):
            effective_mode = "stage_fixed"
            auto_converted = True

    if effective_mode == "cross":
        if cross_section is None or cross_section.empty:
            raise ValueError("水位高程 + 横断面模式需要导入河道横断面。")
        x = cross_section["x_m"].to_numpy(dtype=float)
        z = cross_section["bed_elevation_m"].to_numpy(dtype=float)
        node_widths = compute_node_widths(x)

    rows = []
    total = len(water)
    for number, (timestamp, row) in enumerate(water.iterrows(), start=1):
        raw_value = float(row["water_value_m"])
        activity_fallback = False
        if effective_mode == "depth":
            h_eff = max(raw_value, 0.0)
            w_eff = default_width if h_eff > 0.0 else 0.0
            wet_width = w_eff
            area = h_eff * w_eff
            wetted_perimeter = w_eff + 2.0 * h_eff if h_eff > 0.0 else 0.0
            hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0.0 else 0.0
            mean_depth = h_eff
            water_level = np.nan
        elif effective_mode == "stage_fixed":
            water_level = raw_value
            h_eff = max(water_level - bed_elevation, 0.0)
            w_eff = default_width if h_eff > 0.0 else 0.0
            wet_width = w_eff
            area = h_eff * w_eff
            wetted_perimeter = w_eff + 2.0 * h_eff if h_eff > 0.0 else 0.0
            hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0.0 else 0.0
            mean_depth = h_eff
        else:
            water_level = raw_value
            depth = np.maximum(water_level - z, 0.0)
            wet_width, area, wetted_perimeter, hydraulic_radius, mean_depth = (
                compute_segment_geometry(x, z, water_level)
            )
            weights, _, _ = local_activity_weights(depth, calculation, settling)
            weighted_width = float(np.sum(weights * node_widths))
            if weighted_width > 0.0:
                h_eff = float(np.sum(depth * weights * node_widths) / weighted_width)
                maximum_weight = float(np.nanmax(weights)) if weights.size else 0.0
                active = (
                    (weights >= active_ratio * maximum_weight)
                    & (weights > 0.0)
                    & (depth > 0.0)
                )
                w_eff = float(np.sum(node_widths[active])) if np.any(active) else 0.0
                if w_eff <= 0.0:
                    w_eff = wet_width
                    activity_fallback = wet_width > 0.0
            else:
                # 活动粒径积分没有正权重时，水力几何本身仍然有效。
                h_eff = mean_depth
                w_eff = wet_width
                activity_fallback = wet_width > 0.0

        d_min, d_max = representative_effective_grain_range(
            h_eff, calculation, settling
        )
        rows.append(
            {
                "time_utc": timestamp,
                "input_water_value_m": raw_value,
                "water_level_m": water_level,
                "mean_depth_m": mean_depth,
                "hydraulic_radius_m": hydraulic_radius,
                "wetted_width_m": wet_width,
                "wetted_area_m2": area,
                "wetted_perimeter_m": wetted_perimeter,
                "H_eff_m": h_eff,
                "W_eff_m": w_eff,
                "D_effective_min_m": d_min,
                "D_effective_max_m": d_max,
                "D_effective_min_mm": d_min * 1000.0 if np.isfinite(d_min) else np.nan,
                "D_effective_max_mm": d_max * 1000.0 if np.isfinite(d_max) else np.nan,
                "auto_stage_conversion_flag": bool(auto_converted),
                "activity_weight_fallback_flag": bool(activity_fallback),
            }
        )
        if progress_callback is not None and (
            number == total or number == 1 or number % max(1, total // 100) == 0
        ):
            progress_callback(int(100 * number / max(total, 1)))

    result = pd.DataFrame(rows).set_index("time_utc")
    result.index = ensure_utc_index(result.index, "UTC")
    result.index.name = "time_utc"
    result = result.sort_index()
    result.attrs["requested_input_mode"] = requested_mode
    result.attrs["effective_input_mode"] = effective_mode
    result.attrs["auto_stage_conversion"] = auto_converted
    result.attrs["bed_elevation_m"] = bed_elevation
    return result

def align_hydraulic_dataframe(
    hydraulic: pd.DataFrame, target_times: pd.DatetimeIndex
) -> pd.DataFrame:
    """只在观测范围内部将动态水力参数插值到 PSD 时间。"""

    if hydraulic is None or hydraulic.empty:
        raise ValueError("动态水力参数为空。")
    source = hydraulic.copy()
    source.index = ensure_utc_index(source.index, "UTC")
    source = source.groupby(level=0).mean(numeric_only=True).sort_index()
    target = ensure_utc_index(target_times, "UTC")
    union = source.index.union(target).sort_values()
    numeric = source.select_dtypes(include=[np.number])
    aligned = (
        numeric.reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(target)
    )
    aligned.index = target
    aligned.index.name = "time_utc"
    return aligned


def condition_grain_distribution(
    grains: np.ndarray,
    density: np.ndarray,
    mass: np.ndarray | None,
    lower: float | None,
    upper: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """按动态粒径范围截断并返回条件 PDF 及原始有效质量比例。"""

    diameter = np.asarray(grains, dtype=float)
    pdf = np.asarray(density, dtype=float)
    mask = np.isfinite(diameter) & np.isfinite(pdf) & (diameter > 0.0) & (pdf >= 0.0)
    if lower is not None and np.isfinite(lower) and lower > 0.0:
        mask &= diameter >= lower
    if upper is not None and np.isfinite(upper) and upper > 0.0:
        mask &= diameter <= upper
    if np.count_nonzero(mask) < 2:
        raise ValueError("当前有效运动粒径范围内不足两个级配点。")
    selected_d = diameter[mask]
    selected_pdf = normalize_pdf(selected_d, pdf[mask], "动态有效粒径 PDF")
    if mass is not None:
        mass_array = np.asarray(mass, dtype=float)
        active_fraction = float(np.sum(mass_array[mask]) / np.sum(mass_array))
    else:
        active_fraction = float(np.trapezoid(pdf[mask], diameter[mask]))
    return selected_d, selected_pdf, float(np.clip(active_fraction, 0.0, 1.0))


def clean_flux_series(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    window_minutes: float,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """保留原始值，同时生成局部中值尖峰清洗序列。"""

    index = ensure_utc_index(times, "UTC")
    series = pd.Series(np.asarray(values, dtype=float), index=index)
    window = max(1.0, float(window_minutes))
    factor = max(1.0, float(multiplier))
    local = series.rolling(
        f"{window:g}min", center=True, min_periods=3
    ).median()
    flags = (
        series.notna()
        & local.notna()
        & (local > 0.0)
        & (series > factor * local)
    )
    cleaned = series.mask(flags)
    interpolated = cleaned.interpolate(method="time", limit_area="inside")
    cleaned = series.where(~flags, interpolated)
    return (
        cleaned.to_numpy(dtype=float),
        flags.to_numpy(dtype=bool),
        local.to_numpy(dtype=float),
    )


def cumulative_flux_series(
    times: pd.DatetimeIndex, flux_kg_s: np.ndarray
) -> np.ndarray:
    """按真实时间间隔计算逐时累积质量，并跳过长缺口。"""

    index = ensure_utc_index(times, "UTC")
    values = np.asarray(flux_kg_s, dtype=float)
    cumulative = np.zeros(values.size, dtype=float)
    if values.size < 2:
        return cumulative
    seconds = index.asi8.astype(float) / 1.0e9
    delta = np.diff(seconds)
    positive = delta[np.isfinite(delta) & (delta > 0.0)]
    if positive.size == 0:
        return cumulative
    typical = float(np.median(positive))
    for idx in range(1, values.size):
        cumulative[idx] = cumulative[idx - 1]
        if (
            np.isfinite(values[idx - 1])
            and np.isfinite(values[idx])
            and 0.0 < delta[idx - 1] <= 3.0 * typical
        ):
            cumulative[idx] += (
                0.5 * (values[idx - 1] + values[idx]) * delta[idx - 1]
            )
    return cumulative


def _inversion_resolution_minutes(value: Any) -> int:
    """0 使用 PSD 原始时间，1–60 对应 UTC 反演时间窗。"""
    minutes = float(value)
    if not np.isfinite(minutes) or not minutes.is_integer() or not 0 <= minutes <= 60:
        raise ValueError("反演时间分辨率须为 1–60 分钟的整数，或 0（PSD 原始时间）。")
    return int(minutes)


def aggregate_inversion_inputs(
    spectra: np.ndarray,
    frequency_indices: np.ndarray,
    hydraulic: pd.DataFrame,
    minutes: int,
    psd_step_s: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """按固定 UTC 时间窗汇总配对有效的线性谱和水力参数。"""
    times = hydraulic.index
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("选择固定反演分辨率时，PSD 时间戳须有效、唯一且递增。")
    if psd_step_s is None:
        if times.size < 2:
            raise ValueError("至少需要两个 PSD 时间点确定原始采样间隔。")
        psd_step_s = float(np.median((times[1:] - times[:-1]).total_seconds()))
    psd_step_s = _require_positive("原始 PSD 间隔", psd_step_s)
    seconds = minutes * 60
    if psd_step_s > seconds + 1e-6:
        raise ValueError(
            f"当前 PSD 输出间隔为 {psd_step_s:.2f} 秒，不能支持 {minutes} 分钟反演。"
            "请选择更长的反演间隔，或用更密的 PSD 重新计算。"
        )

    interval = pd.Timedelta(seconds=seconds)
    starts = pd.date_range(times[0].floor(interval), times[-1].floor(interval), freq=interval)
    centers = starts + interval / 2
    edges = times.searchsorted(starts.append(pd.DatetimeIndex([starts[-1] + interval])))
    spectra_out = np.full((len(starts), frequency_indices.size), np.nan)
    hydraulic_values = hydraulic.to_numpy(dtype=float)
    hydraulic_out = np.full((len(starts), hydraulic_values.shape[1]), np.nan)
    depth_width = hydraulic[["H_eff_m", "W_eff_m"]].to_numpy(dtype=float)
    valid_rows = (np.isfinite(depth_width) & (depth_width > 0.0)).all(axis=1)
    valid_count = np.zeros(len(starts), dtype=int)
    # 不足半窗的配对观测保留为缺测，避免稀疏数据代表一整窗。
    expected = seconds / psd_step_s
    min_count = max(1, int(np.ceil(0.5 * expected - 1e-9)))
    for row, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if left == right:
            continue
        block = np.asarray(spectra[left:right, frequency_indices], dtype=float)
        valid_power = np.isfinite(block) & (block >= 0.0)
        usable = valid_rows[left:right] & (valid_power.sum(axis=1) >= 2)
        usable &= (valid_power & (block > 0.0)).any(axis=1)
        valid_count[row] = int(usable.sum())
        if valid_count[row] < min_count:
            continue
        power = np.where(valid_power[usable], block[usable], np.nan)
        present = np.isfinite(power).any(axis=0)
        spectra_out[row, present] = np.nanmedian(power[:, present], axis=0)

        values = hydraulic_values[left:right][usable]
        present = np.isfinite(values).any(axis=0)
        hydraulic_out[row, present] = np.nanmedian(
            np.where(np.isfinite(values[:, present]), values[:, present], np.nan), axis=0
        )

    metadata = pd.DataFrame(
        {
            "inversion_window_start_utc": starts,
            "inversion_window_end_utc": starts + interval,
            "inversion_resolution_minutes": minutes,
            "source_psd_interval_s": psd_step_s,
            "source_psd_count": np.diff(edges),
            "paired_psd_count": valid_count,
            "paired_coverage_fraction": np.minimum(valid_count / expected, 1.0),
            "sufficient_paired_coverage": valid_count >= min_count,
        },
        index=centers,
    )
    return (
        spectra_out,
        pd.DataFrame(hydraulic_out, index=centers, columns=hydraulic.columns),
        metadata,
    )


def invert_bedload_timeseries_enhanced(
    df_psd: pd.DataFrame,
    hydraulic_input: pd.DataFrame | pd.Series,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    """融合动态 H_eff/W_eff、实测级配和三种论文模型的统一反演。"""

    if df_psd is None or df_psd.empty:
        raise ValueError("PSD 数据为空。")
    frequencies = np.asarray(df_psd.columns, dtype=float)
    if not np.all(np.isfinite(frequencies)) or np.unique(frequencies).size != frequencies.size:
        raise ValueError("PSD 频率轴必须由互不重复的有限值组成。")
    fmin = _require_positive("反演频带下限", params["f_inv_min"])
    fmax = _require_positive("反演频带上限", params["f_inv_max"])
    if fmin >= fmax:
        raise ValueError("反演频带下限必须小于上限。")
    band_mask = (frequencies >= fmin) & (frequencies <= fmax)
    if np.count_nonzero(band_mask) < 2:
        raise ValueError("反演频带内至少需要两个 PSD 频率点。")
    statistic = params.get("band_statistic", "median")
    if statistic not in {"median", "mean"}:
        raise ValueError("反演频带统计量须为 median 或 mean。")
    reduce_band = np.median if statistic == "median" else np.mean
    frequency_grid = params.get("frequency_grid", "native")
    if frequency_grid not in {"native", "interpolated"}:
        raise ValueError("反演频率轴须为 native 或 interpolated。")
    order = np.argsort(frequencies[band_mask])
    band_frequency = frequencies[band_mask][order]
    resolution = _inversion_resolution_minutes(params.get("resolution_minutes", 0))
    frequency_indices = np.flatnonzero(band_mask)[order]
    psd_times = ensure_utc_index(df_psd.index, "UTC")
    if isinstance(hydraulic_input, pd.Series):
        source = hydraulic_input.to_frame("H_eff_m")
        source["W_eff_m"] = float(params["W"])
    else:
        source = hydraulic_input
    hydraulic = align_hydraulic_dataframe(source, psd_times)
    if "H_eff_m" not in hydraulic.columns:
        if "river_depth_m" not in hydraulic.columns:
            raise ValueError("动态水力数据缺少 H_eff_m。")
        hydraulic["H_eff_m"] = hydraulic["river_depth_m"]
    if "W_eff_m" not in hydraulic.columns:
        hydraulic["W_eff_m"] = float(params["W"])

    time_metadata = None
    if resolution:
        observed_grid, hydraulic, time_metadata = aggregate_inversion_inputs(
            df_psd.to_numpy(copy=False), frequency_indices, hydraulic,
            resolution, params.get("source_psd_interval_s"),
        )
        psd_times = hydraulic.index
    else:
        observed_grid = df_psd.iloc[:, frequency_indices].to_numpy(dtype=float)
    n_rows = len(psd_times)
    model_frequency = band_frequency
    if frequency_grid == "interpolated":
        model_frequency = np.linspace(
            band_frequency[0], band_frequency[-1],
            max(2, int(params.get("n_frequency_bins", 60))),
        )
        interpolated = np.full((n_rows, model_frequency.size), np.nan)
        for row_index, row in enumerate(observed_grid):
            valid = np.isfinite(row) & (row >= 0.0)
            if np.count_nonzero(valid) >= 2:
                interpolated[row_index] = np.interp(
                    model_frequency, band_frequency[valid], row[valid],
                    left=np.nan, right=np.nan,
                )
        observed_grid = interpolated
    observed_band = np.full(n_rows, np.nan)
    valid_frequency = np.isfinite(observed_grid) & (observed_grid >= 0.0)
    frequency_count = np.count_nonzero(valid_frequency, axis=1)
    for row_index in np.flatnonzero(frequency_count >= 2):
        row = observed_grid[row_index]
        observed_band[row_index] = reduce_band(row[valid_frequency[row_index]])
    depths = hydraulic["H_eff_m"].to_numpy(dtype=float)
    widths = hydraulic["W_eff_m"].to_numpy(dtype=float)
    measured_radius = hydraulic.get(
        "hydraulic_radius_m", pd.Series(np.nan, index=psd_times)
    ).to_numpy(dtype=float)
    measured_radius_valid = np.isfinite(measured_radius) & (measured_radius > 0.0)
    shear_radius = np.where(measured_radius_valid, measured_radius, depths)
    sediment = _dataclass_from_mapping(
        SedimentParams, params.get("sediment_params", {})
    )
    model_type = str(params["model_type"])
    if model_type not in MODEL_TYPES:
        raise ValueError(f"未知模型类型：{model_type}")

    reference = np.full(n_rows, np.nan)
    qb_raw = np.full(n_rows, np.nan)
    active_fraction = np.ones(n_rows, dtype=float)
    range_fallback = np.zeros(n_rows, dtype=bool)
    status = np.full(n_rows, "not_evaluated", dtype=object)
    diagnostics: dict[str, np.ndarray] = {}

    hydraulic_valid = (
        np.isfinite(depths) & (depths > 0.0)
        & np.isfinite(widths) & (widths > 0.0)
    )
    psd_valid = np.isfinite(observed_band) & (observed_band >= 0.0)
    status[~hydraulic_valid] = "invalid_or_unaligned_hydraulic"
    status[hydraulic_valid & ~psd_valid] = "invalid_observed_psd"
    if time_metadata is not None:
        status[~time_metadata["sufficient_paired_coverage"].to_numpy()] = (
            "insufficient_paired_coverage"
        )

    if model_type == "Luong2026":
        empirical = SeismicHydraulicModel(sediment)
        empirical_params = dict(params.get("empirical_params", {}))
        outputs = empirical.inverse_bedload(
            observed_band_psd_linear=observed_band,
            depth_m=depths,
            slope=params["slope"],
            D50=params["D50"],
            hydraulic_radius_m=measured_radius,
            **empirical_params,
        )
        qb_raw = outputs["qb_vol_m2_s"]
        qb_raw[~(hydraulic_valid & psd_valid)] = np.nan
        shear_radius = outputs["shear_hydraulic_radius_m"]
        reference = 10.0 ** (outputs["reference_psd_db"] / 10.0)
        diagnostics = {
            "dimensionless_shields": outputs["shields"],
            "reference_psd_db": outputs["reference_psd_db"],
            "observed_reference_power_ratio": outputs["power_ratio"],
            "dimensionless_bedload_flux": outputs["dimensionless_flux"],
        }
        ok = np.isfinite(qb_raw) & hydraulic_valid & psd_valid
        status[ok] = "ok"
        below = ok & (outputs["shields"] <= empirical_params.get("tau_cr", 0.045))
        status[below] = "below_transport_threshold"
        status[hydraulic_valid & psd_valid & ~ok] = "invalid_empirical_reference"
    else:
        seismic = _dataclass_from_mapping(
            SeismicParams, params.get("seismic_params", {})
        )
        grain_config = params.get("grain_distribution")
        if grain_config is not None:
            base_d, base_mass, base_pdf = mass_fraction_to_pdf(
                np.asarray(grain_config["D_m"], dtype=float),
                np.asarray(grain_config["fraction"], dtype=float),
            )
        else:
            base_d, base_pdf = build_grain_distribution(
                D50=params["D50"],
                log_std=params["grain_log_std"],
                n_points=params.get("grain_points", 80),
                D_min=params.get("D_min"),
                D_max=params.get("D_max"),
            )
            base_mass = None
        if model_type == "Tsai2012":
            model: SaltationModel | MultimodeModel = SaltationModel(
                sediment, seismic
            )
            hop_times = pdf_t = None
        else:
            model_class = IMHEModel2026 if model_type == "IMHE2026" else MultimodeModel
            model = model_class(sediment, seismic)
            hop_times, pdf_t = build_hop_time_distribution(
                t_min=params.get("hop_t_min", 0.01),
                t_max=params.get("hop_t_max", 0.50),
                log_median=params.get("hop_log_median", 0.08),
                log_sigma=params.get("hop_log_sigma", 1.0),
                n_points=params.get("hop_points", 70),
            )
        use_range = bool(params.get("use_effective_grain_range", False))
        correct_fraction = bool(params.get("correct_effective_fraction", False))
        dmin_values = (
            hydraulic["D_effective_min_m"].to_numpy(dtype=float)
            if "D_effective_min_m" in hydraulic.columns
            else np.full(n_rows, np.nan)
        )
        dmax_values = (
            hydraulic["D_effective_max_m"].to_numpy(dtype=float)
            if "D_effective_max_m" in hydraulic.columns
            else np.full(n_rows, np.nan)
        )
        valid_rows = hydraulic_valid & psd_valid
        stage_form = params.get("transport_stage_form", "excess")
        ks = params.get("ks")
        roughness = 3.0 * params["D50"] if ks is None else _require_positive("粗糙度 ks", ks)
        clip_tau_c = bool(params.get("clip_tau_c", model_type in MULTIMODE_TYPES))
        for index in np.flatnonzero(valid_rows):
            fraction = 1.0
            if use_range:
                try:
                    grains, grain_pdf, fraction = condition_grain_distribution(
                        base_d,
                        base_pdf,
                        base_mass,
                        dmin_values[index],
                        dmax_values[index],
                    )
                except ValueError:
                    # 动态范围与离散实测级配无交集时，回退完整级配，避免整段
                    # 反演被跳过；诊断列明确记录该回退。
                    grains, grain_pdf = base_d, base_pdf
                    fraction = 0.0
                    range_fallback[index] = True
            else:
                grains, grain_pdf = base_d, base_pdf
            active_fraction[index] = fraction
            common = dict(
                f=model_frequency[valid_frequency[index]],
                D=grains,
                H=float(depths[index]),
                W=float(widths[index]),
                slope=params["slope"],
                r0=params["r0"],
                qb=1.0,
                D50=params["D50"],
                pdf_D=grain_pdf,
                tau_c50=params.get("tau_c50", 0.045),
                clip_tau_c=clip_tau_c,
                hydraulic_radius_m=float(shear_radius[index]),
                ks=roughness,
            )
            if model_type == "Tsai2012":
                modeled = model.forward_psd(**common, transport_stage_form=stage_form)
            else:
                modeled = model.forward_psd(
                    **common, hop_times=hop_times, pdf_t=pdf_t
                )
            model_reference = float(reduce_band(modeled))
            # 只有在真正截断成功时才应用活动质量比例。回退完整级配时
            # fraction=0 仅作为诊断，不能再把理论 PSD 乘为 0。
            if correct_fraction and not range_fallback[index]:
                model_reference *= fraction
            reference[index] = model_reference
            if np.isfinite(model_reference) and model_reference > EPS_PSD:
                qb_raw[index] = observed_band[index] / model_reference
                status[index] = (
                    "ok_full_grain_fallback" if range_fallback[index] else "ok"
                )
            else:
                status[index] = "below_transport_threshold_or_zero_reference"

    filter_params = params.get("outlier_filter", {})
    filter_enabled = bool(filter_params.get("enabled", False))
    if filter_enabled:
        qb_clean, outlier_flag, local_median = clean_flux_series(
            psd_times,
            qb_raw,
            filter_params.get("window_minutes", 60.0),
            filter_params.get("multiplier", 5.0),
        )
    else:
        qb_clean = qb_raw.copy()
        outlier_flag = np.zeros(qb_raw.size, dtype=bool)
        local_median = np.full(qb_raw.size, np.nan)

    result = pd.DataFrame(index=psd_times)
    result.index.name = "time_utc"
    if time_metadata is not None:
        for column in time_metadata:
            result[column] = time_metadata[column].to_numpy()
    else:
        result["inversion_resolution_minutes"] = 0
    result["river_depth_m"] = depths
    result["H_eff_m"] = depths
    result["W_eff_m"] = widths
    result["shear_hydraulic_radius_m"] = shear_radius
    result["hydraulic_radius_from_section_flag"] = measured_radius_valid
    for column in (
        "water_level_m", "mean_depth_m", "hydraulic_radius_m",
        "wetted_width_m", "D_effective_min_m", "D_effective_max_m",
        "D_effective_min_mm", "D_effective_max_mm",
        "auto_stage_conversion_flag", "activity_weight_fallback_flag",
    ):
        if column in hydraulic.columns:
            result[column] = hydraulic[column].to_numpy()
    result["observed_band_psd_linear"] = observed_band
    result["observed_band_psd_db"] = 10.0 * np.log10(
        np.maximum(observed_band, EPS_PSD)
    )
    result["band_statistic"] = statistic
    result["frequency_grid"] = frequency_grid
    result["band_valid_frequency_count"] = frequency_count
    result["rho_s_kg_m3"] = sediment.rho_s
    result["rho_f_kg_m3"] = sediment.rho_f
    result["D50_m"] = params["D50"]
    if model_type != "Luong2026":
        result["roughness_ks_m"] = roughness
        result["clip_tau_c"] = clip_tau_c
    if model_type == "Tsai2012":
        result["tsai_transport_stage"] = stage_form
    if model_type in MULTIMODE_TYPES:
        result["restitution"] = seismic.restitution
        result["impact_azimuth_rad"] = seismic.impact_azimuth_rad
    if model_type == "Luong2026":
        result["reference_A_db"] = empirical_params.get("A_db", 29.0)
        result["reference_B_db"] = empirical_params.get("B_db", -127.0)
        result["tau_cr"] = empirical_params.get("tau_cr", 0.045)
    if model_type == "Luong2026":
        result["reference_psd_linear"] = reference
    else:
        result["reference_psd_per_unit_qb"] = reference
    for name, values in diagnostics.items():
        result[name] = values
    result["hydraulic_valid_flag"] = hydraulic_valid
    result["observed_psd_valid_flag"] = psd_valid
    result["effective_grain_mass_fraction"] = active_fraction
    result["grain_range_fallback_flag"] = range_fallback
    result["inversion_status"] = status
    result["qb_vol_m2_s_raw"] = qb_raw
    result["qb_mass_kg_m_s_raw"] = qb_raw * sediment.rho_s
    result["Qb_mass_kg_s_raw"] = qb_raw * sediment.rho_s * widths
    result["local_median_qb_vol_m2_s"] = local_median
    result["outlier_flag"] = outlier_flag
    result["qb_vol_m2_s_clean"] = qb_clean
    result["qb_mass_kg_m_s_clean"] = qb_clean * sediment.rho_s
    result["Qb_mass_kg_s_clean"] = qb_clean * sediment.rho_s * widths
    result["qb_vol_m2_s"] = qb_clean
    result["qb_mass_kg_m_s"] = qb_clean * sediment.rho_s
    result["Qb_mass_kg_s"] = qb_clean * sediment.rho_s * widths
    result["cumulative_mass_kg_raw"] = cumulative_flux_series(
        psd_times, result["Qb_mass_kg_s_raw"].to_numpy(dtype=float)
    )
    result["cumulative_mass_kg"] = cumulative_flux_series(
        psd_times, result["Qb_mass_kg_s"].to_numpy(dtype=float)
    )
    result["model_name"] = model_type
    return result

class HydraulicWorker(QThread):
    """后台计算断面水力参数，避免长水位序列冻结界面。"""

    finished_sig = pyqtSignal(object)
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)

    def __init__(self, water, cross_section, settings):
        super().__init__()
        self.water = water
        self.cross_section = cross_section
        self.settings = settings

    def run(self):
        try:
            self.progress_sig.emit(2, "正在准备断面几何和沉降速度查找表...")
            result = compute_hydraulic_timeseries(
                self.water,
                self.cross_section,
                self.settings,
                lambda value: self.progress_sig.emit(
                    value, f"正在计算 H_eff/W_eff：{value}%"
                ),
            )
            self.progress_sig.emit(100, "断面水力参数计算完成。")
            self.finished_sig.emit(result)
        except Exception as exc:
            self.error_sig.emit(
                f"水力参数计算异常：{exc}\n{traceback.format_exc()}"
            )


class EnhancedInversionWorker(QThread):
    """融合动态水力和实测粒径的三模型反演线程。"""

    finished_sig = pyqtSignal(object)
    progress_sig = pyqtSignal(int, str)
    error_sig = pyqtSignal(str)

    def __init__(self, df_psd, hydraulic_input, model_params):
        super().__init__()
        self.df_psd = df_psd
        self.hydraulic_input = hydraulic_input
        self.params = model_params

    def run(self):
        try:
            self.progress_sig.emit(
                5, "正在对齐 UTC 水力参数并准备实测/理论粒径分布..."
            )
            result = invert_bedload_timeseries_enhanced(
                self.df_psd, self.hydraulic_input, self.params
            )
            self.progress_sig.emit(100, "动态推移质通量反演完成。")
            self.finished_sig.emit(result)
        except Exception as exc:
            self.error_sig.emit(
                f"融合反演异常：{exc}\n{traceback.format_exc()}"
            )


# 保存旧版 PyQt6 主窗体并在其上扩展，继续复用程序1的全部专业绘图与线程逻辑。
_BaseOBSAnalyzerGUI = OBSAnalyzerGUI


class OBSAnalyzerGUI(_BaseOBSAnalyzerGUI):
    """程序1界面风格下的 v3.1.0 融合主窗体。"""

    def __init__(self):
        self.project_file_path: Path | None = None
        self.project_directory: Path | None = None
        self.water_file_path = ""
        self.cross_section_file_path = ""
        self.grain_file_path = ""
        self.water_input_df: pd.DataFrame | None = None
        self.cross_section_df: pd.DataFrame | None = None
        self.hydraulic_df: pd.DataFrame | None = None
        self.grain_distribution_df: pd.DataFrame | None = None
        super().__init__()
        self.setWindowTitle(
            f"OBS Bedload Analyzer v{APP_VERSION} - PyQt6 Professional Edition"
        )

    def init_ui(self):
        """复用程序1布局，再插入项目和实测粒径标签页。"""

        super().init_ui()
        project_tab = self.build_project_tab()
        grain_tab = self.build_grain_tab()
        self.tabs.insertTab(0, project_tab, create_scientific_icon("settings"), "项目管理")
        self.tabs.insertTab(3, grain_tab, create_scientific_icon("river"), "实测粒径")
        labels = [
            "0. 项目管理",
            "1. 波形预处理",
            "2. 断面水力 Hₑff/Wₑff",
            "3. 实测粒径与级配",
            "4. PSD 时频谱图",
            "5. 模型参数",
            "6. 反演与结果",
        ]
        for index, label in enumerate(labels):
            if index < self.tabs.count():
                self.tabs.setTabText(index, label)
        self.tabs.setMovable(False)
        self.tabs.setDocumentMode(True)
        self.install_project_menu_actions()

    def install_project_menu_actions(self):
        file_menu = self.menuBar().actions()[0].menu()
        first_action = file_menu.actions()[0] if file_menu.actions() else None
        self.act_new_project = QAction("新建项目...", self)
        self.act_new_project.triggered.connect(self.create_project_directory)
        self.act_open_project = QAction("打开项目配置...", self)
        self.act_open_project.triggered.connect(self.load_project_file)
        self.act_save_project = QAction("保存项目配置", self)
        self.act_save_project.triggered.connect(self.save_project_file)
        for action in reversed(
            [self.act_new_project, self.act_open_project, self.act_save_project]
        ):
            if first_action is None:
                file_menu.addAction(action)
            else:
                file_menu.insertAction(first_action, action)
        if first_action is not None:
            file_menu.insertSeparator(first_action)

    # ------------------------------------------------------------------
    # 项目管理
    # ------------------------------------------------------------------
    def build_project_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_basic = QGroupBox("1. 项目基本信息")
        form = QFormLayout(group_basic)
        self.edit_project_name = QLineEdit("OBS_Bedload_Project")
        self.edit_station_id = QLineEdit()
        self.edit_project_dir = QLineEdit()
        self.edit_output_dir = QLineEdit()
        self.edit_project_notes = QTextEdit()
        self.edit_project_notes.setMaximumHeight(90)
        form.addRow("项目名称：", self.edit_project_name)
        form.addRow("台站 / 测点：", self.edit_station_id)
        form.addRow("项目目录：", self.path_row(self.edit_project_dir, self.create_project_directory, "选择/新建"))
        form.addRow("输出目录：", self.path_row(self.edit_output_dir, self.choose_project_output_dir, "选择"))
        form.addRow("备注：", self.edit_project_notes)
        panel_layout.addWidget(group_basic)

        group_files = QGroupBox("2. 当前工程数据路径")
        files_form = QFormLayout(group_files)
        self.edit_project_waveform = QLineEdit()
        self.edit_project_waveform.setReadOnly(True)
        self.edit_project_water = QLineEdit()
        self.edit_project_water.setReadOnly(True)
        self.edit_project_cross = QLineEdit()
        self.edit_project_cross.setReadOnly(True)
        self.edit_project_grain = QLineEdit()
        self.edit_project_grain.setReadOnly(True)
        files_form.addRow("波形文件：", self.edit_project_waveform)
        files_form.addRow("水位/水深：", self.edit_project_water)
        files_form.addRow("河道断面：", self.edit_project_cross)
        files_form.addRow("实测级配：", self.edit_project_grain)
        panel_layout.addWidget(group_files)

        group_actions = QGroupBox("3. 项目操作")
        actions = QGridLayout(group_actions)
        button_new = QPushButton("新建项目目录")
        button_open = QPushButton("打开 project_config.json")
        button_sync = QPushButton("同步当前界面状态")
        button_save = QPushButton("保存项目配置")
        button_new.clicked.connect(self.create_project_directory)
        button_open.clicked.connect(self.load_project_file)
        button_sync.clicked.connect(self.sync_project_summary)
        button_save.clicked.connect(self.save_project_file)
        actions.addWidget(button_new, 0, 0)
        actions.addWidget(button_open, 0, 1)
        actions.addWidget(button_sync, 1, 0)
        actions.addWidget(button_save, 1, 1)
        panel_layout.addWidget(group_actions)
        panel_layout.addStretch()
        layout.addWidget(self.wrap_in_scroll(panel, 520))

        right = QWidget()
        right_layout = QVBoxLayout(right)
        title = QLabel("OBS 推移质监测项目概览")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Weight.Bold))
        self.txt_project_summary = QPlainTextEdit()
        self.txt_project_summary.setReadOnly(True)
        self.txt_project_summary.setFont(QFont("Consolas", 10))
        self.txt_project_summary.setPlainText(
            "尚未建立项目。\n\n建议工作流：\n"
            "1. 建立项目目录；2. 导入波形；3. 计算断面水力；"
            "4. 导入实测级配；5. 计算 PSD；6. 执行模型反演。"
        )
        right_layout.addWidget(title)
        right_layout.addWidget(self.txt_project_summary)
        layout.addWidget(right)
        return tab

    def path_row(self, edit: QLineEdit, callback, button_text: str) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        button = QPushButton(button_text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return widget

    def create_project_directory(self):
        selected = QFileDialog.getExistingDirectory(self, "选择项目根目录")
        if not selected:
            return
        name = self.edit_project_name.text().strip() or "OBS_Bedload_Project"
        root = Path(selected)
        project = root if root.name == name else root / name
        for folder in (
            project,
            project / "01_waveform",
            project / "02_hydraulic",
            project / "03_sediment",
            project / "04_psd",
            project / "05_results",
            project / "figures",
        ):
            folder.mkdir(parents=True, exist_ok=True)
        self.project_directory = project
        self.project_file_path = project / "project_config.json"
        self.edit_project_dir.setText(str(project))
        self.edit_output_dir.setText(str(project / "05_results"))
        self.sync_project_summary()
        self.status_bar.showMessage(f"项目目录已建立：{project}", 4000)

    def choose_project_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择结果输出目录")
        if directory:
            self.edit_output_dir.setText(directory)
            self.sync_project_summary()

    def collect_project_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "project_version": APP_VERSION,
            "project": {
                "name": self.edit_project_name.text().strip(),
                "station_id": self.edit_station_id.text().strip(),
                "project_dir": self.edit_project_dir.text().strip(),
                "output_dir": self.edit_output_dir.text().strip(),
                "notes": self.edit_project_notes.toPlainText().strip(),
            },
            "files": {
                "waveform": self.edit_project_waveform.text().strip(),
                "water": self.water_file_path,
                "cross_section": self.cross_section_file_path,
                "grain_distribution": self.grain_file_path,
            },
            "hydraulic": self.current_hydraulic_settings(allow_missing=True),
            "grain": {
                "mode": self.combo_grain_mode.currentData(),
                "d16_mm": self.spin_d16.value(),
                "d50_mm": self.spin_d50_percentile.value(),
                "d84_mm": self.spin_d84.value(),
                "use_measured": self.chk_use_measured_grain.isChecked(),
                "use_effective_range": self.chk_use_effective_grain_range.isChecked(),
                "correct_effective_fraction": self.chk_correct_effective_fraction.isChecked(),
            },
        }
        if hasattr(self, "combo_model"):
            model_key, values = self.current_model_values()
            data["model"] = {"model_type": model_key, "values": values}
        return data

    def sync_project_summary(self):
        if hasattr(self, "waveform_headers") and self.waveform_headers:
            paths = sorted({record.path for record in self.waveform_headers})
            self.edit_project_waveform.setText("; ".join(paths[:5]))
        self.edit_project_water.setText(self.water_file_path)
        self.edit_project_cross.setText(self.cross_section_file_path)
        self.edit_project_grain.setText(self.grain_file_path)
        data = self.collect_project_data()
        project = data["project"]
        lines = [
            "OBS BEDLOAD PROJECT",
            "=" * 78,
            f"项目名称：{project['name'] or '未命名'}",
            f"台站/测点：{project['station_id'] or '未填写'}",
            f"项目目录：{project['project_dir'] or '未设置'}",
            f"输出目录：{project['output_dir'] or '未设置'}",
            "",
            "数据状态",
            "-" * 78,
            f"波形：{'已扫描' if self.waveform_headers else '未导入'}",
            f"水力参数：{'已计算 ' + str(len(self.hydraulic_df)) + ' 个时刻' if self.hydraulic_df is not None else '未计算'}",
            f"实测级配：{'已建立 ' + str(len(self.grain_distribution_df)) + ' 个粒级' if self.grain_distribution_df is not None else '未建立'}",
            f"PSD：{'已计算 ' + str(len(self.df_psd)) + ' 个时间块' if self.df_psd is not None else '未计算'}",
            f"反演：{'已完成 ' + str(len(self.df_result)) + ' 个时间块' if self.df_result is not None else '未执行'}",
            "",
            "理论依据",
            "-" * 78,
            "Tsai et al. (2012), GRL 39(2)",
            "Luong et al. (2024), JGR Earth Surface 129(9)",
            "Luong et al. (2026), Water Resources Research 62(2)",
            f"seismic_bedload {UPSTREAM_VERSION}, commit {UPSTREAM_COMMIT}",
        ]
        self.txt_project_summary.setPlainText("\n".join(lines))

    def save_project_file(self):
        directory_text = self.edit_project_dir.text().strip()
        if not directory_text:
            self.create_project_directory()
            directory_text = self.edit_project_dir.text().strip()
        if not directory_text:
            return
        project_dir = Path(directory_text)
        project_dir.mkdir(parents=True, exist_ok=True)
        path = self.project_file_path or project_dir / "project_config.json"
        data = self.collect_project_data()
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        self.project_file_path = Path(path)
        self.project_directory = project_dir
        self.sync_project_summary()
        QMessageBox.information(self, "项目已保存", f"配置已保存至：\n{path}")

    def load_project_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "打开项目配置",
            self.edit_project_dir.text().strip(),
            "Project JSON (project_config.json *.json)",
        )
        if not file_name:
            return
        try:
            with Path(file_name).open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            self.apply_project_data(data)
            self.project_file_path = Path(file_name)
            self.project_directory = Path(file_name).parent
            self.sync_project_summary()
            QMessageBox.information(self, "项目已载入", f"已载入：\n{file_name}")
        except Exception as exc:
            QMessageBox.critical(self, "载入失败", str(exc))

    def apply_project_data(self, data: Mapping[str, Any]):
        project = data.get("project", {})
        self.edit_project_name.setText(str(project.get("name", "")))
        self.edit_station_id.setText(str(project.get("station_id", "")))
        self.edit_project_dir.setText(str(project.get("project_dir", "")))
        self.edit_output_dir.setText(str(project.get("output_dir", "")))
        self.edit_project_notes.setPlainText(str(project.get("notes", "")))
        files = data.get("files", {})
        self.water_file_path = str(files.get("water", ""))
        self.cross_section_file_path = str(files.get("cross_section", ""))
        self.grain_file_path = str(files.get("grain_distribution", ""))
        self.edit_water_path.setText(self.water_file_path)
        self.edit_cross_path.setText(self.cross_section_file_path)
        self.edit_grain_path.setText(self.grain_file_path)
        grain = data.get("grain", {})
        mode = grain.get("mode")
        if mode is not None:
            index = self.combo_grain_mode.findData(mode)
            if index >= 0:
                self.combo_grain_mode.setCurrentIndex(index)
        for control, key in (
            (self.spin_d16, "d16_mm"),
            (self.spin_d50_percentile, "d50_mm"),
            (self.spin_d84, "d84_mm"),
        ):
            if key in grain:
                control.setValue(float(grain[key]))
        self.chk_use_measured_grain.setChecked(bool(grain.get("use_measured", True)))
        self.chk_use_effective_grain_range.setChecked(bool(grain.get("use_effective_range", True)))
        self.chk_correct_effective_fraction.setChecked(bool(grain.get("correct_effective_fraction", False)))
        hydraulic = data.get("hydraulic", {})
        self.apply_hydraulic_settings(hydraulic)
        model = data.get("model", {})
        model_key = model.get("model_type")
        if model_key:
            model_index = self.combo_model.findData(model_key)
            if model_index >= 0:
                self.combo_model.setCurrentIndex(model_index)
            controls = self.model_controls.get(model_key, {})
            for key, value in model.get("values", {}).items():
                control = controls.get(key)
                if control is not None and hasattr(control, "setValue"):
                    try:
                        control.setValue(float(value))
                    except Exception:
                        pass
        # 只自动载入轻量表格；波形仍由用户选择目标时间段。
        if self.grain_file_path and Path(self.grain_file_path).exists():
            self.load_grain_file(show_message=False)
        if self.water_file_path and Path(self.water_file_path).exists():
            self.load_water_input(show_message=False)
        if self.cross_section_file_path and Path(self.cross_section_file_path).exists():
            self.load_cross_section(show_message=False)

    # ------------------------------------------------------------------
    # 断面水力标签页（覆盖程序1原水位页）
    # ------------------------------------------------------------------
    def setup_tab_river(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_mode = QGroupBox("1. 水力输入方式与文件")
        mode_layout = QGridLayout(group_mode)
        self.combo_hydraulic_mode = QComboBox()
        self.combo_hydraulic_mode.addItem("水深序列 + 固定有效宽度", "depth")
        self.combo_hydraulic_mode.addItem("水位序列 + 河道横断面", "cross")
        self.combo_hydraulic_mode.currentIndexChanged.connect(self.update_hydraulic_mode_ui)
        mode_layout.addWidget(QLabel("计算模式："), 0, 0)
        mode_layout.addWidget(self.combo_hydraulic_mode, 0, 1, 1, 2)
        self.edit_water_path = QLineEdit()
        button_water = QPushButton("选择水位/水深...")
        button_water.clicked.connect(self.on_load_river)
        mode_layout.addWidget(QLabel("时间序列："), 1, 0)
        mode_layout.addWidget(self.edit_water_path, 1, 1)
        mode_layout.addWidget(button_water, 1, 2)
        self.edit_cross_path = QLineEdit()
        self.btn_cross_path = QPushButton("选择断面...")
        self.btn_cross_path.clicked.connect(self.choose_cross_section_file)
        mode_layout.addWidget(QLabel("河道断面："), 2, 0)
        mode_layout.addWidget(self.edit_cross_path, 2, 1)
        mode_layout.addWidget(self.btn_cross_path, 2, 2)
        self.combo_tz = QComboBox()
        for label, value in (
            ("Asia/Shanghai（北京时间）", "Asia/Shanghai"),
            ("UTC", "UTC"),
            ("Asia/Singapore", "Asia/Singapore"),
            ("US/Mountain", "US/Mountain"),
            ("US/Pacific", "US/Pacific"),
        ):
            self.combo_tz.addItem(label, value)
        mode_layout.addWidget(QLabel("输入时间时区："), 3, 0)
        mode_layout.addWidget(self.combo_tz, 3, 1, 1, 2)
        panel_layout.addWidget(group_mode)

        group_parameters = QGroupBox("2. 断面输移活动参数")
        form = QFormLayout(group_parameters)
        self.spin_hyd_slope = QDoubleSpinBox()
        self.spin_hyd_slope.setRange(1.0e-8, 0.999)
        self.spin_hyd_slope.setDecimals(9)
        self.spin_hyd_slope.setValue(self.config["site"]["slope"])
        self.spin_hyd_rho_s = QDoubleSpinBox()
        self.spin_hyd_rho_s.setRange(1001.0, 6000.0)
        self.spin_hyd_rho_s.setValue(self.config["sediment"]["rho_s_kg_m3"])
        self.spin_hyd_rho_f = QDoubleSpinBox()
        self.spin_hyd_rho_f.setRange(100.0, 2000.0)
        self.spin_hyd_rho_f.setValue(self.config["sediment"]["rho_f_kg_m3"])
        self.spin_hyd_tau_c = QDoubleSpinBox()
        self.spin_hyd_tau_c.setRange(0.0001, 1.0)
        self.spin_hyd_tau_c.setDecimals(6)
        self.spin_hyd_tau_c.setValue(self.config["sediment"]["tau_c50"])
        self.spin_hyd_width = QDoubleSpinBox()
        self.spin_hyd_width.setRange(0.01, 100000.0)
        self.spin_hyd_width.setValue(self.config["site"]["channel_width_m"])
        self.spin_hyd_kappa = QDoubleSpinBox()
        self.spin_hyd_kappa.setRange(0.01, 2.0)
        self.spin_hyd_kappa.setValue(0.4)
        self.spin_hyd_rouse = QDoubleSpinBox()
        self.spin_hyd_rouse.setRange(0.01, 20.0)
        self.spin_hyd_rouse.setValue(2.5)
        self.spin_hyd_active = QDoubleSpinBox()
        self.spin_hyd_active.setRange(0.0, 100.0)
        self.spin_hyd_active.setValue(5.0)
        self.spin_hyd_beta = QDoubleSpinBox()
        self.spin_hyd_beta.setRange(0.01, 10.0)
        self.spin_hyd_beta.setValue(1.5)
        form.addRow("河道坡降 S：", self.spin_hyd_slope)
        form.addRow("颗粒密度 ρs (kg/m³)：", self.spin_hyd_rho_s)
        form.addRow("流体密度 ρf (kg/m³)：", self.spin_hyd_rho_f)
        form.addRow("临界 Shields θc：", self.spin_hyd_tau_c)
        form.addRow("固定/回退宽度 W (m)：", self.spin_hyd_width)
        form.addRow("von Kármán 常数 κ：", self.spin_hyd_kappa)
        form.addRow("Rouse 阈值：", self.spin_hyd_rouse)
        form.addRow("活动宽度阈值 (%)：", self.spin_hyd_active)
        form.addRow("输移活动指数 β：", self.spin_hyd_beta)
        panel_layout.addWidget(group_parameters)

        group_execute = QGroupBox("3. 计算与使用")
        execute = QVBoxLayout(group_execute)
        self.chk_use_dynamic_hydraulic = QCheckBox("反演使用动态 H_eff(t) 与 W_eff(t)")
        self.chk_use_dynamic_hydraulic.setChecked(True)
        self.btn_apply_river = QPushButton("计算断面 H_eff / W_eff 与有效粒径范围")
        self.btn_apply_river.clicked.connect(self.on_apply_river)
        execute.addWidget(self.chk_use_dynamic_hydraulic)
        execute.addWidget(self.btn_apply_river)
        panel_layout.addWidget(group_execute)

        group_table = QGroupBox("结果预览")
        table_layout = QVBoxLayout(group_table)
        self.table_river = QTableWidget()
        self.table_river.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_river.setAlternatingRowColors(True)
        table_layout.addWidget(self.table_river)
        panel_layout.addWidget(group_table)
        layout.addWidget(self.wrap_in_scroll(panel, 530))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_river = Figure(figsize=(8, 6))
        self.canvas_river = FigureCanvas(self.fig_river)
        plot_layout.addWidget(NavigationToolbar(self.canvas_river, self))
        plot_layout.addWidget(self.canvas_river)
        layout.addWidget(plot_panel)
        self.tabs.addTab(tab, create_scientific_icon("river"), "2. 断面水力")
        self.update_hydraulic_mode_ui()

    def update_hydraulic_mode_ui(self):
        cross_mode = self.combo_hydraulic_mode.currentData() == "cross"
        self.edit_cross_path.setEnabled(cross_mode)
        self.btn_cross_path.setEnabled(cross_mode)
        self.spin_hyd_active.setEnabled(cross_mode)

    def choose_cross_section_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择河道横断面文件",
            self.edit_project_dir.text().strip(),
            "Table (*.xlsx *.xls *.csv *.txt *.dat)",
        )
        if file_name:
            self.cross_section_file_path = file_name
            self.edit_cross_path.setText(file_name)
            self.load_cross_section(show_message=True)
            self.sync_project_summary()

    def load_cross_section(self, show_message=True) -> bool:
        path = self.edit_cross_path.text().strip() or self.cross_section_file_path
        if not path:
            return False
        try:
            self.cross_section_df = standardize_cross_section_dataframe(
                read_tabular_file(path)
            )
            self.cross_section_file_path = path
            if show_message:
                QMessageBox.information(
                    self,
                    "断面已读取",
                    f"有效断面点：{len(self.cross_section_df)}\n"
                    f"横向范围：{self.cross_section_df['x_m'].min():.3f}–"
                    f"{self.cross_section_df['x_m'].max():.3f} m",
                )
            return True
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "断面读取失败", str(exc))
            return False

    def on_load_river(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择水位或水深时间序列",
            self.edit_project_dir.text().strip(),
            "Table (*.xlsx *.xls *.csv *.txt *.dat)",
        )
        if file_name:
            self.water_file_path = file_name
            self.edit_water_path.setText(file_name)
            self.load_water_input(show_message=True)
            self.sync_project_summary()

    def load_water_input(self, show_message=True) -> bool:
        path = self.edit_water_path.text().strip() or self.water_file_path
        if not path:
            return False
        try:
            depth_only = self.combo_hydraulic_mode.currentData() == "depth"
            timezone = str(self.combo_tz.currentData() or "UTC")
            self.water_input_df = standardize_water_dataframe(
                read_tabular_file(path), timezone, depth_only
            )
            self.water_file_path = path
            if show_message:
                QMessageBox.information(
                    self,
                    "时间序列已读取",
                    f"有效记录：{len(self.water_input_df)}\n"
                    f"UTC 覆盖：{self.water_input_df.index[0]} 至 "
                    f"{self.water_input_df.index[-1]}",
                )
            return True
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "时间序列读取失败", str(exc))
            return False

    def current_hydraulic_settings(self, allow_missing=False) -> dict[str, Any]:
        if not hasattr(self, "combo_hydraulic_mode"):
            return {}
        settings = {
            "depth_only": self.combo_hydraulic_mode.currentData() == "depth",
            "slope": self.spin_hyd_slope.value(),
            "rho_s": self.spin_hyd_rho_s.value(),
            "rho_f": self.spin_hyd_rho_f.value(),
            "theta_c": self.spin_hyd_tau_c.value(),
            "default_width": self.spin_hyd_width.value(),
            "kappa": self.spin_hyd_kappa.value(),
            "rouse_threshold": self.spin_hyd_rouse.value(),
            "active_threshold_percent": self.spin_hyd_active.value(),
            "transport_exponent": self.spin_hyd_beta.value(),
            "g": 9.81,
            "kinematic_viscosity": 1.0e-6,
            "input_timezone": self.combo_tz.currentData(),
        }
        if not allow_missing:
            if settings["rho_s"] <= settings["rho_f"]:
                raise ValueError("颗粒密度必须大于流体密度。")
        return settings

    def apply_hydraulic_settings(self, settings: Mapping[str, Any]):
        if not settings or not hasattr(self, "combo_hydraulic_mode"):
            return
        mode = "depth" if bool(settings.get("depth_only", False)) else "cross"
        index = self.combo_hydraulic_mode.findData(mode)
        if index >= 0:
            self.combo_hydraulic_mode.setCurrentIndex(index)
        mapping = {
            "slope": self.spin_hyd_slope,
            "rho_s": self.spin_hyd_rho_s,
            "rho_f": self.spin_hyd_rho_f,
            "theta_c": self.spin_hyd_tau_c,
            "default_width": self.spin_hyd_width,
            "kappa": self.spin_hyd_kappa,
            "rouse_threshold": self.spin_hyd_rouse,
            "active_threshold_percent": self.spin_hyd_active,
            "transport_exponent": self.spin_hyd_beta,
        }
        for key, control in mapping.items():
            if key in settings:
                control.setValue(float(settings[key]))
        timezone = settings.get("input_timezone")
        if timezone:
            tz_index = self.combo_tz.findData(timezone)
            if tz_index >= 0:
                self.combo_tz.setCurrentIndex(tz_index)

    def on_apply_river(self):
        if self.water_input_df is None:
            if not self.load_water_input(show_message=True):
                return
        settings = self.current_hydraulic_settings()
        if not settings["depth_only"] and self.cross_section_df is None:
            if not self.load_cross_section(show_message=True):
                return
        self.hydraulic_worker = HydraulicWorker(
            self.water_input_df,
            None if settings["depth_only"] else self.cross_section_df,
            settings,
        )
        self.hydraulic_worker.progress_sig.connect(self.on_progress_update)
        self.hydraulic_worker.finished_sig.connect(self.on_hydraulic_completed)
        self.hydraulic_worker.error_sig.connect(self.on_worker_error)
        self.hydraulic_worker.start()

    def on_hydraulic_completed(self, result: pd.DataFrame):
        self.hydraulic_df = result
        self.river_series_utc = result["H_eff_m"].copy()
        self.df_river = result.copy()
        self.populate_dataframe_table(self.table_river, result.reset_index(), 120)
        self.plot_river_canvas()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(
            f"水力参数完成：{len(result)} 个 UTC 时刻，"
            f"H_eff={result['H_eff_m'].min():.3g}–{result['H_eff_m'].max():.3g} m，"
            f"W_eff={result['W_eff_m'].min():.3g}–{result['W_eff_m'].max():.3g} m",
            6000,
        )
        self.sync_hydraulic_to_model_controls()
        self.sync_project_summary()

    def sync_hydraulic_to_model_controls(self):
        if self.hydraulic_df is None or self.hydraulic_df.empty:
            return
        valid_width = self.hydraulic_df["W_eff_m"].replace(0.0, np.nan).dropna()
        representative_width = (
            float(valid_width.median())
            if not valid_width.empty
            else self.spin_hyd_width.value()
        )
        for model_key in ("Tsai2012", "Luong2024"):
            controls = self.model_controls.get(model_key, {})
            if "W" in controls:
                controls["W"].setValue(representative_width)
            if "slope" in controls:
                controls["slope"].setValue(self.spin_hyd_slope.value())
            if "rho_s" in controls:
                controls["rho_s"].setValue(self.spin_hyd_rho_s.value())
            if "rho_f" in controls:
                controls["rho_f"].setValue(self.spin_hyd_rho_f.value())
        controls = self.model_controls.get("Luong2026", {})
        if "W" in controls:
            controls["W"].setValue(representative_width)
        if "slope" in controls:
            controls["slope"].setValue(self.spin_hyd_slope.value())

    def populate_dataframe_table(
        self, table: QTableWidget, frame: pd.DataFrame, max_rows: int = 100
    ):
        preview = frame.head(max_rows).copy()
        table.setRowCount(len(preview))
        table.setColumnCount(len(preview.columns))
        table.setHorizontalHeaderLabels([str(column) for column in preview.columns])
        for row_index, (_, row) in enumerate(preview.iterrows()):
            for column_index, value in enumerate(row):
                if isinstance(value, float):
                    text = "" if not np.isfinite(value) else f"{value:.7g}"
                else:
                    text = str(value)
                table.setItem(row_index, column_index, QTableWidgetItem(text))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if table.columnCount() > 0:
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    def plot_river_canvas(self):
        if self.hydraulic_df is None or self.hydraulic_df.empty:
            return super().plot_river_canvas()
        result = self.hydraulic_df
        self.fig_river.clear()
        ax_depth = self.fig_river.add_subplot(311)
        ax_width = self.fig_river.add_subplot(312, sharex=ax_depth)
        ax_grain = self.fig_river.add_subplot(313, sharex=ax_depth)
        self.apply_canvas_style(self.fig_river, [ax_depth, ax_width, ax_grain])
        ax_depth.plot(result.index, result["H_eff_m"], lw=1.3, label="H_eff")
        if "mean_depth_m" in result:
            ax_depth.plot(result.index, result["mean_depth_m"], lw=0.9, alpha=0.75, label="Mean depth")
        ax_depth.set_ylabel("Depth (m)")
        ax_depth.set_title("断面水力与有效运动粒径（UTC）")
        ax_depth.legend(loc="upper right")
        ax_width.plot(result.index, result["W_eff_m"], lw=1.2, label="W_eff")
        if "wetted_width_m" in result:
            ax_width.plot(result.index, result["wetted_width_m"], lw=0.8, alpha=0.70, label="Wetted width")
        ax_width.set_ylabel("Width (m)")
        ax_width.legend(loc="upper right")
        valid_min = result["D_effective_min_mm"].to_numpy(dtype=float)
        valid_max = result["D_effective_max_mm"].to_numpy(dtype=float)
        ax_grain.plot(result.index, valid_min, lw=1.0, label="D_eff,min")
        ax_grain.plot(result.index, valid_max, lw=1.0, label="D_eff,max")
        both = np.isfinite(valid_min) & np.isfinite(valid_max)
        if np.any(both):
            ax_grain.fill_between(
                result.index,
                valid_min,
                valid_max,
                where=both,
                alpha=0.18,
            )
        ax_grain.set_ylabel("Grain (mm)")
        ax_grain.set_xlabel("Time (UTC)")
        ax_grain.set_yscale("log")
        ax_grain.legend(loc="upper right")
        ax_grain.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        self.fig_river.tight_layout()
        self.canvas_river.draw_idle()

    # ------------------------------------------------------------------
    # 实测粒径标签页
    # ------------------------------------------------------------------
    def build_grain_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_input = QGroupBox("1. 实测级配输入")
        form = QFormLayout(group_input)
        self.combo_grain_mode = QComboBox()
        self.combo_grain_mode.addItem("导入粒径—质量比例/累计通过率", "file")
        self.combo_grain_mode.addItem("由 D16 / D50 / D84 构造", "percentiles")
        self.combo_grain_mode.currentIndexChanged.connect(self.update_grain_mode_ui)
        self.edit_grain_path = QLineEdit()
        self.btn_grain_file = QPushButton("选择")
        self.btn_grain_file.clicked.connect(self.choose_grain_file)
        grain_path_widget = QWidget()
        grain_path_layout = QHBoxLayout(grain_path_widget)
        grain_path_layout.setContentsMargins(0, 0, 0, 0)
        grain_path_layout.addWidget(self.edit_grain_path)
        grain_path_layout.addWidget(self.btn_grain_file)
        form.addRow("输入方式：", self.combo_grain_mode)
        form.addRow("级配文件：", grain_path_widget)
        self.combo_grain_unit = QComboBox()
        self.combo_grain_unit.addItem("自动识别", "auto")
        self.combo_grain_unit.addItem("毫米 mm", "mm")
        self.combo_grain_unit.addItem("米 m", "m")
        form.addRow("粒径单位：", self.combo_grain_unit)
        self.spin_d16 = QDoubleSpinBox()
        self.spin_d16.setRange(1.0e-5, 10000.0)
        self.spin_d16.setDecimals(5)
        self.spin_d16.setValue(8.0)
        self.spin_d50_percentile = QDoubleSpinBox()
        self.spin_d50_percentile.setRange(1.0e-5, 10000.0)
        self.spin_d50_percentile.setDecimals(5)
        self.spin_d50_percentile.setValue(20.0)
        self.spin_d84 = QDoubleSpinBox()
        self.spin_d84.setRange(1.0e-5, 10000.0)
        self.spin_d84.setDecimals(5)
        self.spin_d84.setValue(50.0)
        form.addRow("D16 (mm)：", self.spin_d16)
        form.addRow("D50 (mm)：", self.spin_d50_percentile)
        form.addRow("D84 (mm)：", self.spin_d84)
        panel_layout.addWidget(group_input)

        group_use = QGroupBox("2. 反演使用方式")
        use_layout = QVBoxLayout(group_use)
        self.chk_use_measured_grain = QCheckBox("Tsai/Luong 2024 使用当前实测级配")
        self.chk_use_measured_grain.setChecked(True)
        self.chk_use_effective_grain_range = QCheckBox("按每时刻 D_eff,min–D_eff,max 截断级配")
        self.chk_use_effective_grain_range.setChecked(True)
        self.chk_correct_effective_fraction = QCheckBox("用活动粒径原始质量比例修正理论 PSD")
        self.chk_correct_effective_fraction.setChecked(False)
        use_layout.addWidget(self.chk_use_measured_grain)
        use_layout.addWidget(self.chk_use_effective_grain_range)
        use_layout.addWidget(self.chk_correct_effective_fraction)
        panel_layout.addWidget(group_use)

        group_actions = QGroupBox("3. 生成、预览与保存")
        actions = QGridLayout(group_actions)
        button_build = QPushButton("读取/生成标准级配")
        button_build.clicked.connect(self.build_current_grain_distribution)
        button_save = QPushButton("保存标准级配...")
        button_save.clicked.connect(self.save_grain_distribution)
        actions.addWidget(button_build, 0, 0)
        actions.addWidget(button_save, 0, 1)
        panel_layout.addWidget(group_actions)

        self.table_grain = QTableWidget()
        self.table_grain.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_grain.setAlternatingRowColors(True)
        group_table = QGroupBox("级配数据预览")
        table_layout = QVBoxLayout(group_table)
        table_layout.addWidget(self.table_grain)
        panel_layout.addWidget(group_table)
        layout.addWidget(self.wrap_in_scroll(panel, 500))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_grain = Figure(figsize=(8, 6))
        self.canvas_grain = FigureCanvas(self.fig_grain)
        plot_layout.addWidget(NavigationToolbar(self.canvas_grain, self))
        plot_layout.addWidget(self.canvas_grain)
        self.lbl_grain_stats = QLabel("尚未建立标准级配。")
        self.lbl_grain_stats.setWordWrap(True)
        plot_layout.addWidget(self.lbl_grain_stats)
        layout.addWidget(plot_panel)
        self.update_grain_mode_ui()
        return tab

    def update_grain_mode_ui(self):
        file_mode = self.combo_grain_mode.currentData() == "file"
        self.edit_grain_path.setEnabled(file_mode)
        self.btn_grain_file.setEnabled(file_mode)
        self.combo_grain_unit.setEnabled(file_mode)
        for control in (self.spin_d16, self.spin_d50_percentile, self.spin_d84):
            control.setEnabled(not file_mode)

    def choose_grain_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择实测粒径分布",
            self.edit_project_dir.text().strip(),
            "Table (*.xlsx *.xls *.csv *.txt *.dat)",
        )
        if file_name:
            self.grain_file_path = file_name
            self.edit_grain_path.setText(file_name)
            self.load_grain_file(show_message=True)
            self.sync_project_summary()

    def load_grain_file(self, show_message=True) -> bool:
        path = self.edit_grain_path.text().strip() or self.grain_file_path
        if not path:
            return False
        try:
            unit = str(self.combo_grain_unit.currentData() or "auto")
            self.grain_distribution_df = standardize_grain_dataframe(
                read_tabular_file(path), unit
            )
            self.grain_file_path = path
            self.on_grain_distribution_ready()
            if show_message:
                QMessageBox.information(
                    self,
                    "级配已读取",
                    f"标准化粒级数：{len(self.grain_distribution_df)}",
                )
            return True
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "级配读取失败", str(exc))
            return False

    def build_current_grain_distribution(self):
        try:
            if self.combo_grain_mode.currentData() == "file":
                if not self.load_grain_file(show_message=False):
                    raise ValueError("请先选择有效的实测级配文件。")
            else:
                self.grain_distribution_df = build_grain_from_percentiles(
                    self.spin_d16.value(),
                    self.spin_d50_percentile.value(),
                    self.spin_d84.value(),
                )
                self.grain_file_path = ""
                self.on_grain_distribution_ready()
            self.sync_project_summary()
        except Exception as exc:
            QMessageBox.critical(self, "级配生成失败", str(exc))

    def on_grain_distribution_ready(self):
        distribution = self.grain_distribution_df
        if distribution is None or distribution.empty:
            return
        self.populate_dataframe_table(self.table_grain, distribution, 150)
        self.plot_grain_distribution()
        diameter = distribution["D_m"].to_numpy(dtype=float)
        mass = distribution["fraction"].to_numpy(dtype=float)
        d16 = weighted_quantile_value(diameter, mass, 0.16)
        d50 = weighted_quantile_value(diameter, mass, 0.50)
        d84 = weighted_quantile_value(diameter, mass, 0.84)
        self.lbl_grain_stats.setText(
            f"D16={d16 * 1000.0:.4g} mm；D50={d50 * 1000.0:.4g} mm；"
            f"D84={d84 * 1000.0:.4g} mm；粒级数={len(distribution)}。"
        )
        for model_key in ("Tsai2012", "Luong2024", "Luong2026"):
            controls = self.model_controls.get(model_key, {})
            if "D50" in controls:
                controls["D50"].setValue(d50)
            if "D_min" in controls:
                controls["D_min"].setValue(float(diameter.min()))
            if "D_max" in controls:
                controls["D_max"].setValue(float(diameter.max()))

    def plot_grain_distribution(self):
        if self.grain_distribution_df is None:
            return
        distribution = self.grain_distribution_df
        self.fig_grain.clear()
        ax_mass = self.fig_grain.add_subplot(211)
        ax_cum = self.fig_grain.add_subplot(212, sharex=ax_mass)
        self.apply_canvas_style(self.fig_grain, [ax_mass, ax_cum])
        ax_mass.plot(
            distribution["D_mm"],
            distribution["fraction_percent"],
            marker="o",
            markersize=3,
            lw=1.0,
        )
        ax_mass.set_ylabel("Fraction (%)")
        ax_mass.set_title("实测/构造粒径分布")
        ax_cum.plot(
            distribution["D_mm"],
            distribution["cum_percent"],
            marker="o",
            markersize=3,
            lw=1.1,
        )
        ax_cum.axhline(50.0, lw=0.8, linestyle="--", alpha=0.7)
        ax_cum.set_xscale("log")
        ax_cum.set_ylim(0.0, 102.0)
        ax_cum.set_xlabel("Grain diameter D (mm)")
        ax_cum.set_ylabel("Cumulative (%)")
        self.fig_grain.tight_layout()
        self.canvas_grain.draw_idle()

    def save_grain_distribution(self):
        if self.grain_distribution_df is None:
            QMessageBox.warning(self, "提示", "尚未建立标准级配。")
            return
        default = self.edit_output_dir.text().strip() or "grain_distribution_standard.csv"
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存标准级配",
            str(Path(default) / "grain_distribution_standard.csv") if Path(default).is_dir() else default,
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not file_name:
            return
        if file_name.lower().endswith(".xlsx"):
            self.grain_distribution_df.to_excel(file_name, index=False)
        else:
            self.grain_distribution_df.to_csv(file_name, index=False, encoding="utf-8-sig")
        QMessageBox.information(self, "保存成功", file_name)

    # ------------------------------------------------------------------
    # 反演页与融合反演
    # ------------------------------------------------------------------
    def setup_tab_inversion(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_exec = QGroupBox("1. 床载反演计算")
        execute = QVBoxLayout(group_exec)
        self.btn_run_inv = QPushButton("启动动态推移质通量 qb 反演")
        self.btn_run_inv.clicked.connect(self.on_run_inversion)
        execute.addWidget(self.btn_run_inv)
        panel_layout.addWidget(group_exec)

        group_filter = QGroupBox("2. 瞬态异常处理（原始结果始终保留）")
        filter_form = QFormLayout(group_filter)
        self.chk_filter_flux = QCheckBox("生成局部中值尖峰清洗结果")
        self.chk_filter_flux.setChecked(False)
        self.spin_filter_window = QDoubleSpinBox()
        self.spin_filter_window.setRange(1.0, 1440.0)
        self.spin_filter_window.setValue(60.0)
        self.spin_filter_multiplier = QDoubleSpinBox()
        self.spin_filter_multiplier.setRange(1.01, 1000.0)
        self.spin_filter_multiplier.setValue(5.0)
        filter_form.addRow(self.chk_filter_flux)
        filter_form.addRow("滚动窗口 (min)：", self.spin_filter_window)
        filter_form.addRow("尖峰倍数阈值：", self.spin_filter_multiplier)
        panel_layout.addWidget(group_filter)

        group_summary = QGroupBox("3. 推移质反演结果汇总")
        summary_layout = QVBoxLayout(group_summary)
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        summary_layout.addWidget(self.txt_summary)
        panel_layout.addWidget(group_summary)

        group_export = QGroupBox("4. 数据与图像导出")
        export_layout = QVBoxLayout(group_export)
        self.btn_export_excel = QPushButton("导出 Excel / CSV 数据...")
        self.btn_export_excel.clicked.connect(self.on_export_excel)
        self.btn_export_img = QPushButton("导出高分辨率图像包 (PNG)...")
        self.btn_export_img.clicked.connect(self.on_export_images)
        export_layout.addWidget(self.btn_export_excel)
        export_layout.addWidget(self.btn_export_img)
        panel_layout.addWidget(group_export)
        layout.addWidget(self.wrap_in_scroll(panel, 420))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_inv = Figure(figsize=(8, 6))
        self.canvas_inv = FigureCanvas(self.fig_inv)
        plot_layout.addWidget(NavigationToolbar(self.canvas_inv, self))
        plot_layout.addWidget(self.canvas_inv)
        layout.addWidget(plot_panel)
        self.tabs.addTab(tab, create_scientific_icon("inversion"), "6. 反演与结果")

    def on_run_inversion(self):
        if self.df_psd is None:
            QMessageBox.warning(self, "警告", "请先计算 PSD 功率谱。")
            return
        if self.chk_use_dynamic_hydraulic.isChecked():
            if self.hydraulic_df is None:
                QMessageBox.warning(self, "警告", "请先计算断面水力 H_eff/W_eff。")
                return
            hydraulic_input: pd.DataFrame | pd.Series = self.hydraulic_df
        else:
            if self.river_series_utc is None:
                QMessageBox.warning(self, "警告", "请先准备水深数据。")
                return
            hydraulic_input = self.river_series_utc

        if getattr(self, "inv_worker", None) is not None and self.inv_worker.isRunning():
            return
        resolution_control = getattr(self, "spin_inversion_resolution", None)
        resolution = resolution_control.value() if resolution_control is not None else 0
        psd_step = self.psd_metadata.nominal_block_seconds if self.psd_metadata is not None else None
        if resolution and psd_step is not None and psd_step > resolution * 60 + 1e-6:
            QMessageBox.warning(
                self, "反演分辨率过细",
                f"当前 PSD 每 {psd_step:.2f} 秒输出一条谱。"
                "请选择更长的反演间隔，或重新计算时间间隔更短的 PSD。",
            )
            return
        model_key, values = self.current_model_values()
        sediment_cfg = self.config["sediment"]
        params: dict[str, Any] = {
            "model_type": model_key,
            "D50": values["D50"],
            "D_min": None if values.get("D_min", 0.0) <= 0.0 else values["D_min"],
            "D_max": values.get("D_max"),
            "grain_log_std": values.get("grain_log_std", sediment_cfg["grain_log_std"]),
            "grain_points": int(values.get("grain_points", sediment_cfg["grain_points"])),
            "W": values["W"],
            "slope": values["slope"],
            "r0": values.get("r0", self.config["site"]["source_distance_m"]),
            "tau_c50": values.get("tau_c50", sediment_cfg["tau_c50"]),
            "clip_tau_c": values.get("clip_tau_c", model_key == "Luong2024"),
            "ks": values.get("ks"),
            "transport_stage_form": values.get("transport_stage_form", "excess"),
            "f_inv_min": values["f_inv_min"],
            "f_inv_max": values["f_inv_max"],
            "n_frequency_bins": int(values["n_frequency_bins"]),
            "band_statistic": values.get("band_statistic", "median"),
            "frequency_grid": values.get("frequency_grid", "native"),
            "resolution_minutes": resolution,
            "source_psd_interval_s": psd_step,
            "sediment_params": {
                "rho_s": values["rho_s"],
                "rho_f": values["rho_f"],
            },
            "use_effective_grain_range": self.chk_use_effective_grain_range.isChecked(),
            "correct_effective_fraction": self.chk_correct_effective_fraction.isChecked(),
            "outlier_filter": {
                "enabled": self.chk_filter_flux.isChecked(),
                "window_minutes": self.spin_filter_window.value(),
                "multiplier": self.spin_filter_multiplier.value(),
            },
        }
        if (
            model_key in {"Tsai2012", "Luong2024"}
            and self.chk_use_measured_grain.isChecked()
        ):
            if self.grain_distribution_df is None:
                QMessageBox.warning(
                    self,
                    "警告",
                    "已选择使用实测级配，但尚未读取或生成标准级配。",
                )
                return
            params["grain_distribution"] = {
                "D_m": self.grain_distribution_df["D_m"].to_numpy(dtype=float),
                "fraction": self.grain_distribution_df["fraction"].to_numpy(dtype=float),
            }
        if model_key in {"Tsai2012", "Luong2024"}:
            params["seismic_params"] = {
                "v0": values["v0"],
                "z0": values["z0"],
                "f0": values["f0"],
                "a": values["a"],
                "Q0": values["Q0"],
                "eta": values["eta"],
                "vc0_override": None if values["vc0_override"] <= 0.0 else values["vc0_override"],
                "zeta_override": None if values["zeta_override"] < 0.0 else values["zeta_override"],
                "restitution": values.get("restitution", 0.0),
                "impact_azimuth_rad": values.get("impact_azimuth_rad"),
                "fz": values.get("fz", self.config["seismic"]["fz"]),
                "Nzz": values.get("Nzz", self.config["seismic"]["Nzz"]),
            }
        if model_key == "Luong2024":
            params.update(
                {
                    "hop_t_min": values["hop_t_min"],
                    "hop_t_max": values["hop_t_max"],
                    "hop_log_median": values["hop_log_median"],
                    "hop_log_sigma": values["hop_log_sigma"],
                    "hop_points": int(values["hop_points"]),
                }
            )
        if model_key == "Luong2026":
            params["empirical_params"] = {
                key: values[key]
                for key in (
                    "tau_cr", "k", "m", "n", "A_db", "B_db",
                    "hydraulic_radius_depth_ratio",
                )
            }
        self.inv_worker = EnhancedInversionWorker(
            self.df_psd, hydraulic_input, params
        )
        self.inv_worker.progress_sig.connect(self.on_progress_update)
        self.inv_worker.finished_sig.connect(self.on_inversion_completed)
        self.inv_worker.error_sig.connect(self.on_worker_error)
        self.inv_worker.start()

    def on_inversion_completed(self, df_res):
        self.df_result = df_res
        valid = df_res["Qb_mass_kg_s"].replace([np.inf, -np.inf], np.nan).dropna()
        raw = df_res["Qb_mass_kg_s_raw"].replace([np.inf, -np.inf], np.nan).dropna()
        total_mass = float(df_res["cumulative_mass_kg"].iloc[-1]) if len(df_res) else 0.0
        raw_total = float(df_res["cumulative_mass_kg_raw"].iloc[-1]) if len(df_res) else 0.0
        outliers = int(df_res["outlier_flag"].sum())
        model_key, model_values = self.current_model_values()
        width_valid = df_res["W_eff_m"].replace(0.0, np.nan).dropna()
        active_fraction = df_res["effective_grain_mass_fraction"].dropna()
        stats_text = (
            f"模型：{model_key}\n"
            f"有效反演块：{len(valid)} / {len(df_res)}\n"
            f"动态 H_eff：{df_res['H_eff_m'].min():.4g}–{df_res['H_eff_m'].max():.4g} m\n"
            f"动态 W_eff：{width_valid.min() if len(width_valid) else np.nan:.4g}–"
            f"{width_valid.max() if len(width_valid) else np.nan:.4g} m\n\n"
            "清洗/主结果 Qb\n"
            f"  平均：{valid.mean() if len(valid) else np.nan:.5g} kg/s\n"
            f"  峰值：{valid.max() if len(valid) else np.nan:.5g} kg/s\n"
            f"  累积质量：{total_mass:.6g} kg\n\n"
            "原始结果 Qb_raw\n"
            f"  峰值：{raw.max() if len(raw) else np.nan:.5g} kg/s\n"
            f"  累积质量：{raw_total:.6g} kg\n"
            f"  标记尖峰：{outliers} 个\n\n"
            "级配与核心公式\n"
            f"  D50={model_values['D50']:.6g} m；"
            f"活动粒径平均质量比例={active_fraction.mean() if len(active_fraction) else np.nan:.3f}\n"
            "  U=8.1u*[H/(3D50)]^(1/6)\n"
            "  Ub,Hb ∝ max(transport_stage-1,0)^exponent\n"
        )
        self.txt_summary.setPlainText(stats_text)
        self.plot_inv_canvas()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("融合反演完成。", 5000)
        self.sync_project_summary()

    def plot_inv_canvas(self):
        if self.df_result is None:
            return
        result = self.df_result
        self.fig_inv.clear()
        ax_h = self.fig_inv.add_subplot(311)
        ax_q = self.fig_inv.add_subplot(312, sharex=ax_h)
        ax_c = self.fig_inv.add_subplot(313, sharex=ax_h)
        self.apply_canvas_style(self.fig_inv, [ax_h, ax_q, ax_c])
        ax_h.plot(result.index, result["H_eff_m"], lw=1.1, label="H_eff")
        ax_h_w = ax_h.twinx()
        self.apply_canvas_style(self.fig_inv, ax_h_w)
        ax_h_w.grid(False)
        ax_h_w.plot(result.index, result["W_eff_m"], lw=1.0, alpha=0.75, label="W_eff")
        ax_h.set_ylabel("H_eff (m)")
        ax_h_w.set_ylabel("W_eff (m)")
        ax_h.set_title("动态水力—推移质反演结果")
        ax_q.plot(result.index, result["Qb_mass_kg_s_raw"], lw=0.8, alpha=0.55, label="Raw Qb")
        ax_q.plot(result.index, result["Qb_mass_kg_s"], lw=1.2, label="Main/Clean Qb")
        flags = result["outlier_flag"].to_numpy(dtype=bool)
        if np.any(flags):
            ax_q.scatter(
                result.index[flags],
                result.loc[flags, "Qb_mass_kg_s_raw"],
                s=12,
                marker="x",
                label="Flagged",
            )
        ax_q.set_ylabel("Qb (kg/s)")
        ax_q.legend(loc="upper right")
        ax_c.plot(result.index, result["cumulative_mass_kg_raw"], lw=0.8, alpha=0.55, label="Raw cumulative")
        ax_c.plot(result.index, result["cumulative_mass_kg"], lw=1.2, label="Main cumulative")
        ax_c.set_ylabel("Mass (kg)")
        ax_c.set_xlabel("Time (UTC)")
        ax_c.legend(loc="upper left")
        ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        self.fig_inv.tight_layout()
        self.canvas_inv.draw_idle()

    # ------------------------------------------------------------------
    # 通用绘图、导出和文件状态同步
    # ------------------------------------------------------------------
    def on_waveform_headers_scanned(self, records):
        super().on_waveform_headers_scanned(records)
        paths = sorted({record.path for record in records})
        self.edit_project_waveform.setText("; ".join(paths[:5]))
        self.sync_project_summary()

    def on_psd_computed(self, df_psd, metadata):
        super().on_psd_computed(df_psd, metadata)
        self.sync_project_summary()

    def style_existing_canvases(self):
        super().style_existing_canvases()
        for figure_name, canvas_name in (
            ("fig_grain", "canvas_grain"),
        ):
            figure = getattr(self, figure_name, None)
            canvas = getattr(self, canvas_name, None)
            if figure is not None:
                self.apply_canvas_style(figure, list(figure.axes))
                if canvas is not None:
                    canvas.draw_idle()

    def on_export_images(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择保存图像文件夹",
            self.edit_output_dir.text().strip(),
        )
        if not folder:
            return
        targets = [
            ("fig_wf", "1_waveform_qc.png"),
            ("fig_river", "2_hydraulic_heff_weff.png"),
            ("fig_grain", "3_grain_distribution.png"),
            ("fig_psd", "4_psd_spectrogram.png"),
            ("fig_inv", "5_inversion_results.png"),
        ]
        saved = []
        for attribute, file_name in targets:
            figure = getattr(self, attribute, None)
            if figure is not None and figure.axes:
                path = Path(folder) / file_name
                figure.savefig(path, dpi=300, bbox_inches="tight")
                saved.append(file_name)
        QMessageBox.information(
            self,
            "图像导出完成",
            f"已导出 {len(saved)} 幅图至：\n{folder}",
        )

    def show_about(self):
        QMessageBox.about(
            self,
            "关于 OBS Bedload Analyzer",
            f"<h3>OBS Bedload Analyzer v{APP_VERSION}</h3>"
            "<p>PyQt6 专业融合版：连续 PSD、Tsai/Luong 三模型、UTC、"
            "实测级配、断面 H<sub>eff</sub>/W<sub>eff</sub> 与项目管理。</p>"
            "<p><b>确认公式：</b><br>"
            "U=8.1u<sub>*</sub>[H/(3D<sub>50</sub>)]<sup>1/6</sup><br>"
            "U<sub>b</sub>, H<sub>b</sub> 使用 max(T−1,0)<sup>exponent</sup></p>"
            "<p>参考：Tsai et al. (2012)；Luong et al. (2024, 2026)；"
            f"seismic_bedload {UPSTREAM_VERSION}, commit {UPSTREAM_COMMIT}。</p>"
        )



# =============================================================================
# 7. 六步工作流界面
# =============================================================================

class OBSAnalyzerGUIV322(OBSAnalyzerGUI):
    """六步工作流、可伸缩分栏和单一参数源的 PyQt6 主界面。"""

    def __init__(self):
        self.last_data_directory = ""
        super().__init__()
        self.setWindowTitle(
            f"OBS Bedload Analyzer v{APP_VERSION} - PyQt6 Professional Edition"
        )

    def init_ui(self):
        # 直接调用程序1基础界面初始化，不创建 v3.1 的项目管理页与菜单。
        _BaseOBSAnalyzerGUI.init_ui(self)
        grain_tab = self.build_grain_tab_v320()
        self.tabs.insertTab(2, grain_tab, create_scientific_icon("river"), "3. 颗粒粒径")
        labels = [
            "1. 地震波形",
            "2. 水文数据",
            "3. 颗粒粒径",
            "4. PSD 时频谱图",
            "5. 反演模型",
            "6. 计算结果",
        ]
        for index, label in enumerate(labels):
            self.tabs.setTabText(index, label)
        self.tabs.setMovable(False)
        self.tabs.setDocumentMode(True)
        file_menu = self.menuBar().actions()[0].menu()
        for action in file_menu.actions():
            if "水位" in action.text():
                action.setText("打开水文数据...")
        # 默认建立一套可用的 D16/D50/D84 级配，避免未加载粒径时反演无源。
        self.combo_grain_mode.setCurrentIndex(
            max(0, self.combo_grain_mode.findData("percentiles"))
        )
        self.grain_distribution_df = build_grain_from_percentiles(
            self.spin_d16.value(),
            self.spin_d50_percentile.value(),
            self.spin_d84.value(),
        )
        self.on_grain_distribution_ready()

    # --------------------------- 可伸缩左右分栏 ---------------------------
    def _flex_scroll(self, widget: QWidget, minimum_width: int = 280) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(minimum_width)
        scroll.setMaximumWidth(16777215)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return scroll

    def _convert_tab_layout_to_splitter(self, tab: QWidget, left_size=430, right_size=980):
        layout = tab.layout()
        if layout is None or isinstance(layout, QSplitter):
            return
        widgets = []
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widgets.append(widget)
        if len(widgets) < 2:
            for widget in widgets:
                layout.addWidget(widget)
            return
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        for widget in widgets:
            if isinstance(widget, QScrollArea):
                widget.setMinimumWidth(260)
                widget.setMaximumWidth(16777215)
                widget.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
                )
            splitter.addWidget(widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([left_size, right_size])
        layout.addWidget(splitter)
        tab._main_splitter = splitter

    def setup_tab_waveform(self):
        _BaseOBSAnalyzerGUI.setup_tab_waveform(self)
        self._convert_tab_layout_to_splitter(self.tabs.widget(self.tabs.count() - 1), 460, 980)

    def setup_tab_psd(self):
        _BaseOBSAnalyzerGUI.setup_tab_psd(self)
        self._convert_tab_layout_to_splitter(self.tabs.widget(self.tabs.count() - 1), 420, 1020)

    # --------------------------- 2. 水文数据 ---------------------------
    def setup_tab_river(self):
        tab = QWidget()
        root = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        root.addWidget(splitter)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        group_mode = QGroupBox("1. 水文输入与时间体系")
        mode_layout = QGridLayout(group_mode)
        self.combo_hydraulic_mode = QComboBox()
        self.combo_hydraulic_mode.addItem("水深序列 + 固定有效宽度", "depth")
        self.combo_hydraulic_mode.addItem("水位高程 + 参考河床高程 + 固定宽度", "stage_fixed")
        self.combo_hydraulic_mode.addItem("水位高程 + 河道横断面", "cross")
        self.combo_hydraulic_mode.currentIndexChanged.connect(self.update_hydraulic_mode_ui)
        mode_layout.addWidget(QLabel("输入类型："), 0, 0)
        mode_layout.addWidget(self.combo_hydraulic_mode, 0, 1, 1, 2)

        self.edit_water_path = QLineEdit()
        button_water = QPushButton("选择水文文件...")
        button_water.clicked.connect(self.on_load_river)
        mode_layout.addWidget(QLabel("时间序列："), 1, 0)
        mode_layout.addWidget(self.edit_water_path, 1, 1)
        mode_layout.addWidget(button_water, 1, 2)

        self.edit_cross_path = QLineEdit()
        self.btn_cross_path = QPushButton("选择断面...")
        self.btn_cross_path.clicked.connect(self.choose_cross_section_file)
        mode_layout.addWidget(QLabel("河道断面："), 2, 0)
        mode_layout.addWidget(self.edit_cross_path, 2, 1)
        mode_layout.addWidget(self.btn_cross_path, 2, 2)

        self.combo_tz = QComboBox()
        for label, value in (
            ("Asia/Shanghai（北京时间）", "Asia/Shanghai"),
            ("UTC", "UTC"),
            ("Asia/Singapore", "Asia/Singapore"),
            ("US/Mountain", "US/Mountain"),
            ("US/Pacific", "US/Pacific"),
        ):
            self.combo_tz.addItem(label, value)
        mode_layout.addWidget(QLabel("输入时间时区："), 3, 0)
        mode_layout.addWidget(self.combo_tz, 3, 1, 1, 2)
        panel_layout.addWidget(group_mode)

        group_convert = QGroupBox("2. 水位—水深换算与合理性检查")
        form = QFormLayout(group_convert)
        self.spin_bed_elevation = QDoubleSpinBox()
        self.spin_bed_elevation.setRange(-10000.0, 10000.0)
        self.spin_bed_elevation.setDecimals(4)
        self.spin_bed_elevation.setValue(self.config["site"].get("bed_elevation_m", 0.0))
        self.chk_auto_stage = QCheckBox("水深值明显异常时，自动按水位高程减河床高程")
        self.chk_auto_stage.setChecked(True)
        self.chk_auto_stage.toggled.connect(self.update_hydraulic_mode_ui)
        self.spin_max_plausible_depth = QDoubleSpinBox()
        self.spin_max_plausible_depth.setRange(0.1, 2000.0)
        self.spin_max_plausible_depth.setDecimals(2)
        self.spin_max_plausible_depth.setValue(50.0)
        form.addRow("参考河床高程 (m)：", self.spin_bed_elevation)
        form.addRow("最大合理水深 (m)：", self.spin_max_plausible_depth)
        form.addRow(self.chk_auto_stage)
        note = QLabel(
            "坡降、密度、临界 Shields、固定宽度及活动参数仅在“5. 反演模型”中设置，"
            "此处直接调用同一参数源，避免重复与不一致。"
        )
        note.setWordWrap(True)
        form.addRow(note)
        panel_layout.addWidget(group_convert)

        group_execute = QGroupBox("3. 计算 H_eff / W_eff")
        execute = QVBoxLayout(group_execute)
        self.btn_apply_river = QPushButton("计算水力参数与有效粒径范围")
        self.btn_apply_river.clicked.connect(self.on_apply_river)
        self.lbl_hydraulic_diagnostic = QLabel("尚未计算水力参数。")
        self.lbl_hydraulic_diagnostic.setWordWrap(True)
        execute.addWidget(self.btn_apply_river)
        execute.addWidget(self.lbl_hydraulic_diagnostic)
        panel_layout.addWidget(group_execute)

        group_table = QGroupBox("结果预览")
        table_layout = QVBoxLayout(group_table)
        self.table_river = QTableWidget()
        self.table_river.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_river.setAlternatingRowColors(True)
        table_layout.addWidget(self.table_river)
        panel_layout.addWidget(group_table)
        splitter.addWidget(self._flex_scroll(panel, 300))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_river = Figure(figsize=(8, 6))
        self.canvas_river = FigureCanvas(self.fig_river)
        plot_layout.addWidget(NavigationToolbar(self.canvas_river, self))
        plot_layout.addWidget(self.canvas_river)
        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 970])
        self.tabs.addTab(tab, create_scientific_icon("river"), "2. 水文数据")
        self.update_hydraulic_mode_ui()

    def update_hydraulic_mode_ui(self):
        mode = self.combo_hydraulic_mode.currentData()
        cross_mode = mode == "cross"
        self.edit_cross_path.setEnabled(cross_mode)
        self.btn_cross_path.setEnabled(cross_mode)
        self.spin_bed_elevation.setEnabled(mode == "stage_fixed" or self.chk_auto_stage.isChecked())

    def current_hydraulic_settings(self, allow_missing=False) -> dict[str, Any]:
        mode = str(self.combo_hydraulic_mode.currentData())
        values = self.current_model_values()[1] if hasattr(self, "combo_model") else {}
        settings = {
            "input_mode": mode,
            "depth_only": mode == "depth",
            "slope": float(values.get("slope", self.config["site"]["slope"])),
            "rho_s": float(values.get("rho_s", self.config["sediment"]["rho_s_kg_m3"])),
            "rho_f": float(values.get("rho_f", self.config["sediment"]["rho_f_kg_m3"])),
            "theta_c": float(values.get("tau_c50", self.config["sediment"]["tau_c50"])),
            "default_width": float(values.get("W", self.config["site"]["channel_width_m"])),
            "kappa": self.spin_activity_kappa.value(),
            "rouse_threshold": self.spin_activity_rouse.value(),
            "active_threshold_percent": self.spin_activity_width_threshold.value(),
            "transport_exponent": self.spin_activity_beta.value(),
            "g": 9.81,
            "kinematic_viscosity": 1.0e-6,
            "input_timezone": self.combo_tz.currentData(),
            "bed_elevation_m": self.spin_bed_elevation.value(),
            "auto_detect_stage": self.chk_auto_stage.isChecked(),
            "max_plausible_depth_m": self.spin_max_plausible_depth.value(),
        }
        if not allow_missing and settings["rho_s"] <= settings["rho_f"]:
            raise ValueError("颗粒密度必须大于流体密度。")
        return settings

    def apply_hydraulic_settings(self, settings: Mapping[str, Any]):
        if not settings:
            return
        mode = str(settings.get("input_mode", "depth" if settings.get("depth_only") else "cross"))
        index = self.combo_hydraulic_mode.findData(mode)
        if index >= 0:
            self.combo_hydraulic_mode.setCurrentIndex(index)
        if "bed_elevation_m" in settings:
            self.spin_bed_elevation.setValue(float(settings["bed_elevation_m"]))
        if "max_plausible_depth_m" in settings:
            self.spin_max_plausible_depth.setValue(float(settings["max_plausible_depth_m"]))
        self.chk_auto_stage.setChecked(bool(settings.get("auto_detect_stage", True)))

    def choose_cross_section_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择河道横断面文件", self.last_data_directory,
            "Table (*.xlsx *.xls *.csv *.txt *.dat)"
        )
        if file_name:
            self.last_data_directory = str(Path(file_name).parent)
            self.cross_section_file_path = file_name
            self.edit_cross_path.setText(file_name)
            self.load_cross_section(show_message=True)

    def on_load_river(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择水位或水深时间序列", self.last_data_directory,
            "Table (*.xlsx *.xls *.csv *.txt *.dat)"
        )
        if file_name:
            self.last_data_directory = str(Path(file_name).parent)
            self.water_file_path = file_name
            self.edit_water_path.setText(file_name)
            self.load_water_input(show_message=True)

    def load_water_input(self, show_message=True) -> bool:
        path = self.edit_water_path.text().strip() or self.water_file_path
        if not path:
            return False
        try:
            mode = str(self.combo_hydraulic_mode.currentData())
            timezone = str(self.combo_tz.currentData() or "UTC")
            self.water_input_df = standardize_water_dataframe(
                read_tabular_file(path), timezone, mode == "depth"
            )
            self.water_file_path = path
            values = self.water_input_df["water_value_m"]
            if show_message:
                QMessageBox.information(
                    self, "时间序列已读取",
                    f"有效记录：{len(self.water_input_df)}\n"
                    f"数值范围：{values.min():.6g}–{values.max():.6g} m\n"
                    f"UTC 覆盖：{self.water_input_df.index[0]} 至 {self.water_input_df.index[-1]}"
                )
            return True
        except Exception as exc:
            if show_message:
                QMessageBox.critical(self, "时间序列读取失败", str(exc))
            return False

    def on_apply_river(self):
        if self.water_input_df is None and not self.load_water_input(show_message=True):
            return
        try:
            settings = self.current_hydraulic_settings()
        except Exception as exc:
            QMessageBox.critical(self, "参数错误", str(exc))
            return
        if settings["input_mode"] == "cross" and self.cross_section_df is None:
            if not self.load_cross_section(show_message=True):
                return
        self.hydraulic_worker = HydraulicWorker(
            self.water_input_df,
            self.cross_section_df if settings["input_mode"] == "cross" else None,
            settings,
        )
        self.hydraulic_worker.progress_sig.connect(self.on_progress_update)
        self.hydraulic_worker.finished_sig.connect(self.on_hydraulic_completed)
        self.hydraulic_worker.error_sig.connect(self.on_worker_error)
        self.hydraulic_worker.start()

    def on_hydraulic_completed(self, result: pd.DataFrame):
        self.hydraulic_df = result
        self.river_series_utc = result["H_eff_m"].copy()
        self.df_river = result.copy()
        self.populate_dataframe_table(self.table_river, result.reset_index(), 150)
        self.plot_river_canvas()
        self.progress_bar.setVisible(False)
        auto_count = int(result.get("auto_stage_conversion_flag", pd.Series(dtype=bool)).sum())
        fallback_count = int(result.get("activity_weight_fallback_flag", pd.Series(dtype=bool)).sum())
        mode = result.attrs.get("effective_input_mode", "unknown")
        valid_h = result["H_eff_m"].replace([np.inf, -np.inf], np.nan).dropna()
        valid_w = result["W_eff_m"].replace(0.0, np.nan).dropna()
        message = (
            f"有效模式：{mode}；H_eff={valid_h.min():.4g}–{valid_h.max():.4g} m；"
            f"W_eff={valid_w.min() if len(valid_w) else np.nan:.4g}–"
            f"{valid_w.max() if len(valid_w) else np.nan:.4g} m；"
            f"水位自动换算={auto_count} 条；活动权重回退={fallback_count} 条。"
        )
        self.lbl_hydraulic_diagnostic.setText(message)
        self.status_bar.showMessage("水力参数计算完成：" + message, 8000)

    def sync_hydraulic_to_model_controls(self):
        # v3.2 使用单一共享参数源，不再把水文页参数复制到模型页。
        return

    # --------------------------- 3. 颗粒粒径 ---------------------------
    def build_grain_tab_v320(self) -> QWidget:
        tab = QWidget()
        root = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        root.addWidget(splitter)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        group_input = QGroupBox("1. 粒径级配输入")
        form = QFormLayout(group_input)
        self.combo_grain_mode = QComboBox()
        self.combo_grain_mode.addItem("导入粒径—质量比例/累计通过率", "file")
        self.combo_grain_mode.addItem("由 D16 / D50 / D84 构造", "percentiles")
        self.combo_grain_mode.currentIndexChanged.connect(self.update_grain_mode_ui)
        self.edit_grain_path = QLineEdit()
        self.btn_grain_file = QPushButton("选择")
        self.btn_grain_file.clicked.connect(self.choose_grain_file)
        path_widget = QWidget()
        path_layout = QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.edit_grain_path)
        path_layout.addWidget(self.btn_grain_file)
        form.addRow("输入方式：", self.combo_grain_mode)
        form.addRow("级配文件：", path_widget)
        self.combo_grain_unit = QComboBox()
        self.combo_grain_unit.addItem("自动识别", "auto")
        self.combo_grain_unit.addItem("毫米 mm", "mm")
        self.combo_grain_unit.addItem("米 m", "m")
        form.addRow("粒径单位：", self.combo_grain_unit)
        self.spin_d16 = QDoubleSpinBox(); self.spin_d16.setRange(1e-5, 10000); self.spin_d16.setDecimals(5); self.spin_d16.setValue(8.0)
        self.spin_d50_percentile = QDoubleSpinBox(); self.spin_d50_percentile.setRange(1e-5, 10000); self.spin_d50_percentile.setDecimals(5); self.spin_d50_percentile.setValue(20.0)
        self.spin_d84 = QDoubleSpinBox(); self.spin_d84.setRange(1e-5, 10000); self.spin_d84.setDecimals(5); self.spin_d84.setValue(50.0)
        form.addRow("D16 (mm)：", self.spin_d16)
        form.addRow("D50 (mm)：", self.spin_d50_percentile)
        form.addRow("D84 (mm)：", self.spin_d84)
        panel_layout.addWidget(group_input)

        group_actions = QGroupBox("2. 标准化与导出")
        actions = QGridLayout(group_actions)
        button_build = QPushButton("读取/生成标准级配")
        button_build.clicked.connect(self.build_current_grain_distribution)
        button_save = QPushButton("保存标准级配...")
        button_save.clicked.connect(self.save_grain_distribution)
        actions.addWidget(button_build, 0, 0)
        actions.addWidget(button_save, 0, 1)
        note = QLabel("是否使用实测级配及动态有效粒径截断，统一在“5. 反演模型”中设置。")
        note.setWordWrap(True)
        actions.addWidget(note, 1, 0, 1, 2)
        panel_layout.addWidget(group_actions)

        self.table_grain = QTableWidget()
        self.table_grain.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_grain.setAlternatingRowColors(True)
        group_table = QGroupBox("级配数据预览")
        table_layout = QVBoxLayout(group_table)
        table_layout.addWidget(self.table_grain)
        panel_layout.addWidget(group_table)
        splitter.addWidget(self._flex_scroll(panel, 300))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_grain = Figure(figsize=(8, 6))
        self.canvas_grain = FigureCanvas(self.fig_grain)
        plot_layout.addWidget(NavigationToolbar(self.canvas_grain, self))
        plot_layout.addWidget(self.canvas_grain)
        self.lbl_grain_stats = QLabel("尚未建立标准级配。")
        self.lbl_grain_stats.setWordWrap(True)
        plot_layout.addWidget(self.lbl_grain_stats)
        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 990])
        self.update_grain_mode_ui()
        return tab

    def choose_grain_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择实测粒径分布", self.last_data_directory,
            "Table (*.xlsx *.xls *.csv *.txt *.dat)"
        )
        if file_name:
            self.last_data_directory = str(Path(file_name).parent)
            self.grain_file_path = file_name
            self.edit_grain_path.setText(file_name)
            self.load_grain_file(show_message=True)

    def save_grain_distribution(self):
        if self.grain_distribution_df is None:
            QMessageBox.warning(self, "提示", "尚未建立标准级配。")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存标准级配",
            str(Path(self.last_data_directory or ".") / "grain_distribution_standard.csv"),
            "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not file_name:
            return
        self.last_data_directory = str(Path(file_name).parent)
        if file_name.lower().endswith(".xlsx"):
            self.grain_distribution_df.to_excel(file_name, index=False)
        else:
            self.grain_distribution_df.to_csv(file_name, index=False, encoding="utf-8-sig")
        QMessageBox.information(self, "保存成功", file_name)

    # --------------------------- 5. 反演模型 ---------------------------
    def setup_tab_model(self):
        tab = QWidget()
        root = QVBoxLayout(tab)
        group_sel = QGroupBox("1. 反演模型选择")
        select_layout = QHBoxLayout(group_sel)
        self.combo_model = QComboBox()
        for label, key in (
            ("Tsai et al. (2012) 单一跃移物理模型", "Tsai2012"),
            ("Luong et al. (2024) 多输移模式物理模型", "Luong2024"),
            ("Luong et al. (2026) 地震—水力经验模型", "Luong2026"),
        ):
            self.combo_model.addItem(label, key)
        configured = self.config["inversion"]["model_type"]
        idx = self.combo_model.findData(configured)
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)
        select_layout.addWidget(QLabel("模型算法："))
        select_layout.addWidget(self.combo_model)
        select_layout.addStretch()
        root.addWidget(group_sel)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        root.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)

        def dspin(value, lo, hi, decimals=6):
            control = QDoubleSpinBox()
            control.setRange(lo, hi); control.setDecimals(decimals); control.setValue(float(value))
            return control
        def ispin(value, lo, hi):
            control = QSpinBox(); control.setRange(lo, hi); control.setValue(int(value)); return control

        sediment = self.config["sediment"]
        site = self.config["site"]
        seismic = self.config["seismic"]
        inversion = self.config["inversion"]
        multimode = self.config["multimode"]
        empirical = self.config["empirical2026"]

        shared_group = QGroupBox("2. 共享场地、泥沙与反演参数（唯一参数源）")
        shared_form = QFormLayout(shared_group)
        self.shared_model_controls = {
            "W": dspin(site["channel_width_m"], 0.01, 100000, 3),
            "slope": dspin(site["slope"], 1e-8, 0.999, 9),
            "r0": dspin(site["source_distance_m"], 0.01, 1000000, 3),
            "rho_s": dspin(sediment["rho_s_kg_m3"], 1001, 6000, 2),
            "rho_f": dspin(sediment["rho_f_kg_m3"], 100, 2000, 2),
            "tau_c50": dspin(sediment["tau_c50"], 0.0001, 1.0, 6),
            "f_inv_min": dspin(inversion["f_inv_min_hz"], 0.001, 10000, 3),
            "f_inv_max": dspin(inversion["f_inv_max_hz"], 0.002, 10000, 3),
            "n_frequency_bins": ispin(inversion["n_frequency_bins"], 2, 2000),
            "grain_log_std": dspin(sediment["grain_log_std"], 0.01, 5.0, 5),
            "grain_points": ispin(sediment["grain_points"], 8, 5000),
        }
        shared_labels = {
            "W": "固定/回退有效宽度 W (m)",
            "slope": "无量纲坡降 S=tan(θ)",
            "r0": "台站距离 r0 (m)",
            "rho_s": "颗粒密度 ρs (kg/m³)",
            "rho_f": "流体密度 ρf (kg/m³)",
            "tau_c50": "临界 Shields τc,50*",
            "f_inv_min": "反演频带下限 (Hz)",
            "f_inv_max": "反演频带上限 (Hz)",
            "n_frequency_bins": "模型频率点数",
            "grain_log_std": "无实测级配时对数标准差 σ",
            "grain_points": "无实测级配时积分点数",
        }
        for key, control in self.shared_model_controls.items():
            shared_form.addRow(shared_labels[key], control)
        self.combo_band_statistic = QComboBox()
        self.combo_band_statistic.addItem("线性 PSD 中位数", "median")
        self.combo_band_statistic.addItem("线性 PSD 均值", "mean")
        self.combo_band_statistic.setCurrentIndex(
            self.combo_band_statistic.findData(inversion.get("band_statistic", "median"))
        )
        shared_form.addRow("反演频带统计量", self.combo_band_statistic)
        self.combo_frequency_grid = QComboBox()
        self.combo_frequency_grid.addItem("观测原始有效频率点", "native")
        self.combo_frequency_grid.addItem("线性插值到等间隔频率轴", "interpolated")
        self.combo_frequency_grid.setCurrentIndex(
            self.combo_frequency_grid.findData(inversion.get("frequency_grid", "native"))
        )
        shared_form.addRow("反演频率轴", self.combo_frequency_grid)
        count_control = self.shared_model_controls["n_frequency_bins"]
        count_control.setEnabled(self.combo_frequency_grid.currentData() == "interpolated")
        self.combo_frequency_grid.currentIndexChanged.connect(
            lambda _: count_control.setEnabled(
                self.combo_frequency_grid.currentData() == "interpolated"
            )
        )
        left_layout.addWidget(shared_group)

        activity_group = QGroupBox("3. 动态水力与粒径耦合")
        activity_form = QFormLayout(activity_group)
        self.spin_activity_kappa = dspin(0.4, 0.01, 2.0, 4)
        self.spin_activity_rouse = dspin(2.5, 0.01, 20.0, 4)
        self.spin_activity_width_threshold = dspin(5.0, 0.0, 100.0, 2)
        self.spin_activity_beta = dspin(1.5, 0.01, 10.0, 4)
        activity_form.addRow("von Kármán 常数 κ", self.spin_activity_kappa)
        activity_form.addRow("Rouse 阈值", self.spin_activity_rouse)
        activity_form.addRow("活动宽度阈值 (%)", self.spin_activity_width_threshold)
        activity_form.addRow("输移活动指数 β", self.spin_activity_beta)
        self.chk_use_dynamic_hydraulic = QCheckBox("反演使用动态 H_eff(t) 与 W_eff(t)")
        self.chk_use_dynamic_hydraulic.setChecked(True)
        self.chk_use_measured_grain = QCheckBox("Tsai/Luong 2024 使用“3.颗粒粒径”的标准级配")
        self.chk_use_measured_grain.setChecked(True)
        self.chk_use_effective_grain_range = QCheckBox("按每时刻 D_eff,min–D_eff,max 截断级配（实验选项）")
        self.chk_use_effective_grain_range.setChecked(False)
        self.chk_correct_effective_fraction = QCheckBox("用活动粒径原始质量比例修正理论 PSD")
        self.chk_correct_effective_fraction.setChecked(False)
        activity_form.addRow(self.chk_use_dynamic_hydraulic)
        activity_form.addRow(self.chk_use_measured_grain)
        activity_form.addRow(self.chk_use_effective_grain_range)
        activity_form.addRow(self.chk_correct_effective_fraction)
        left_layout.addWidget(activity_group)

        propagation_group = QGroupBox("4. 瑞利波传播参数（Tsai / Luong 2024）")
        self.propagation_group = propagation_group
        propagation_form = QFormLayout(propagation_group)
        self.seismic_shared_controls = {
            "v0": dspin(seismic["v0_m_s"], 1, 20000, 3),
            "z0": dspin(seismic["z0_m"], 0.01, 100000, 3),
            "f0": dspin(seismic["f0_hz"], 0.001, 1000, 3),
            "a": dspin(seismic["a"], 0, 0.999, 6),
            "Q0": dspin(seismic["Q0"], 0.01, 10000, 3),
            "eta": dspin(seismic["eta"], -5, 5, 6),
            "vc0_override": dspin(seismic["vc0_override_m_s"] or 0.0, 0, 20000, 3),
            "zeta_override": dspin(seismic["zeta_override"] if seismic["zeta_override"] is not None else -1.0, -1, 10, 6),
        }
        for key, label in (
            ("v0", "介质速度 v0 (m/s)"), ("z0", "参考深度 z0 (m)"),
            ("f0", "参考频率 f0 (Hz)"), ("a", "速度幂指数 a"),
            ("Q0", "品质因子 Q0"), ("eta", "Q 频率指数 η"),
            ("vc0_override", "vc0 覆盖值 (0=自动)"),
            ("zeta_override", "ζ 覆盖值 (-1=自动)"),
        ):
            propagation_form.addRow(label, self.seismic_shared_controls[key])
        left_layout.addWidget(propagation_group)

        self.model_param_stack = QStackedWidget()
        tsai_page = QWidget(); tsai_form = QFormLayout(tsai_page)
        self.spin_roughness = dspin(sediment.get("roughness_m") or 0.0, 0, 100, 6)
        propagation_form.addRow("河床粗糙度 ks (m；0=3D50)", self.spin_roughness)
        self.chk_tsai_clip = QCheckBox("将粒径临界 Shields 限制在 [0.03, 0.06]")
        self.chk_tsai_clip.setChecked(sediment.get("clip_tau_c_tsai", False))
        tsai_form.addRow(self.chk_tsai_clip)
        self.combo_tsai_stage = QComboBox()
        self.combo_tsai_stage.addItem("OBS 超额输移阶段 T−1", "excess")
        self.combo_tsai_stage.addItem("Tsai 2012 原式 T", "published")
        self.combo_tsai_stage.setCurrentIndex(
            self.combo_tsai_stage.findData(sediment.get("tsai_transport_stage", "excess"))
        )
        tsai_form.addRow("计算跃移速度和层厚的输移阶段", self.combo_tsai_stage)
        tsai_note = QLabel("T≤1 时两个选项均将理论撞击 PSD 置零。")
        tsai_note.setWordWrap(True); tsai_form.addRow(tsai_note)

        luong_page = QWidget(); luong_form = QFormLayout(luong_page)
        self.luong_controls = {
            "restitution": dspin(seismic["restitution_gamma"], 0, 1, 5),
            "fz": dspin(seismic["fz"], 0, 10, 5),
            "Nzz": dspin(seismic["Nzz"], 0, 10, 5),
            "hop_t_min": dspin(multimode["hop_t_min_s"], 1e-5, 100, 6),
            "hop_t_max": dspin(multimode["hop_t_max_s"], 1e-4, 1000, 6),
            "hop_log_median": dspin(multimode["hop_log_median_s"], 1e-5, 100, 6),
            "hop_log_sigma": dspin(multimode["hop_log_sigma"], 0.001, 10, 6),
            "hop_points": ispin(multimode["hop_points"], 8, 5000),
        }
        for key, label in (
            ("restitution", "恢复系数 γ"), ("fz", "法向速度比 fz"),
            ("Nzz", "垂向方向系数 Nzz"), ("hop_t_min", "冲击间隔下限 (s)"),
            ("hop_t_max", "冲击间隔上限 (s)"),
            ("hop_log_median", "对数正态中位数 (s)"),
            ("hop_log_sigma", "对数正态 σ"), ("hop_points", "冲击间隔积分点数"),
        ):
            luong_form.addRow(label, self.luong_controls[key])
        self.chk_multimode_clip = QCheckBox("将粒径临界 Shields 限制在 [0.03, 0.06]")
        self.chk_multimode_clip.setChecked(sediment.get("clip_tau_c_multimode", True))
        luong_form.addRow(self.chk_multimode_clip)
        self.chk_impact_azimuth = QCheckBox("启用冲击的水平耦合项")
        self.spin_impact_azimuth = dspin(0.0, -360.0, 360.0, 3)
        self.spin_impact_azimuth.setEnabled(False)
        self.chk_impact_azimuth.toggled.connect(self.spin_impact_azimuth.setEnabled)
        luong_form.addRow(self.chk_impact_azimuth)
        luong_form.addRow("冲击方位角 (°)", self.spin_impact_azimuth)
        restitution_note = QLabel("恢复系数默认假设为 0（完全非弹性），可用现场标定值覆盖。")
        restitution_note.setWordWrap(True)
        luong_form.addRow(restitution_note)

        empirical_page = QWidget(); empirical_form = QFormLayout(empirical_page)
        self.empirical_controls = {
            "k": dspin(empirical["k"], 1e-8, 1e8, 6),
            "m": dspin(empirical["m"], 1e-6, 20, 6),
            "n": dspin(empirical["n"], 1e-6, 20, 6),
            "A_db": dspin(empirical["A_db"], -1000, 1000, 5),
            "B_db": dspin(empirical["B_db"], -1000, 1000, 5),
            "tau_cr": dspin(empirical["tau_cr"], 0, 1, 6),
            "hydraulic_radius_depth_ratio": dspin(empirical["hydraulic_radius_depth_ratio"], 0.001, 10, 6),
        }
        for key, label in (
            ("k", "尺度系数 k"), ("m", "PSD 修正指数 m"),
            ("n", "超额剪应力指数 n"), ("A_db", "参考 PSD 斜率 A (dB)"),
            ("B_db", "参考 PSD 截距 B (dB)"), ("tau_cr", "临界 Shields τcr*"),
            ("hydraulic_radius_depth_ratio", "缺少断面 Rh 时的 Rh/H"),
        ):
            empirical_form.addRow(label, self.empirical_controls[key])
        empirical_note = QLabel("Luong 2026 参数为 ADLP 标定值，迁移场地应结合实测输沙重新标定。")
        empirical_note.setWordWrap(True); empirical_form.addRow(empirical_note)
        calibration_button = QPushButton("用独立数据拟合参考 PSD 的 A、B")
        calibration_button.setToolTip("表格需要 shields 和 psd_linear 两列。PSD 必须与反演使用同一频带及统计量。")
        calibration_button.clicked.connect(self.calibrate_reference_psd)
        empirical_form.addRow(calibration_button)

        self.model_param_stack.addWidget(tsai_page)
        self.model_param_stack.addWidget(luong_page)
        self.model_param_stack.addWidget(empirical_page)
        self.model_page_index = {"Tsai2012": 0, "Luong2024": 1, "Luong2026": 2}
        specific_group = QGroupBox("5. 当前模型专属参数")
        specific_layout = QVBoxLayout(specific_group)
        specific_layout.addWidget(self.model_param_stack)
        left_layout.addWidget(specific_group)

        self.model_controls = {
            "Tsai2012": {**self.shared_model_controls, **self.seismic_shared_controls},
            "Luong2024": {**self.shared_model_controls, **self.seismic_shared_controls, **self.luong_controls},
            "Luong2026": {**self.shared_model_controls, **self.empirical_controls},
        }
        self.model_defaults = {
            key: {name: control.value() for name, control in controls.items()}
            for key, controls in self.model_controls.items()
        }
        button_restore = QPushButton("恢复当前模型默认参数")
        button_restore.clicked.connect(self.restore_current_model_defaults)
        left_layout.addWidget(button_restore)
        left_layout.addStretch()
        splitter.addWidget(self._flex_scroll(left, 320))

        right = QGroupBox("物理机制分区图与关键公式")
        right_layout = QVBoxLayout(right)
        self.schematic_widget = ModelSchematicWidget()
        self.formula_canvas = FormulaCanvas(self)
        right_layout.addWidget(self.schematic_widget)
        right_layout.addWidget(self.formula_canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([510, 930])
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        self.on_model_changed()
        self.tabs.addTab(tab, create_scientific_icon("settings"), "5. 反演模型")

    def on_model_changed(self):
        _BaseOBSAnalyzerGUI.on_model_changed(self)
        model_key = str(self.combo_model.currentData())
        if hasattr(self, "propagation_group"):
            self.propagation_group.setVisible(model_key in {"Tsai2012", "Luong2024"})
        if hasattr(self, "shared_model_controls"):
            self.shared_model_controls["r0"].setEnabled(model_key in {"Tsai2012", "Luong2024"})

    def current_model_values(self):
        model_key = str(self.combo_model.currentData())
        values = {
            key: control.value()
            for key, control in self.model_controls[model_key].items()
        }
        values["clip_tau_c"] = (
            self.chk_tsai_clip.isChecked() if model_key == "Tsai2012"
            else self.chk_multimode_clip.isChecked()
        )
        values["ks"] = self.spin_roughness.value() or None
        values["transport_stage_form"] = self.combo_tsai_stage.currentData()
        values["band_statistic"] = self.combo_band_statistic.currentData()
        values["frequency_grid"] = self.combo_frequency_grid.currentData()
        values["impact_azimuth_rad"] = (
            float(np.deg2rad(self.spin_impact_azimuth.value()))
            if model_key == "Luong2024" and self.chk_impact_azimuth.isChecked()
            else None
        )
        if self.grain_distribution_df is not None and not self.grain_distribution_df.empty:
            diameter = self.grain_distribution_df["D_m"].to_numpy(dtype=float)
            mass = self.grain_distribution_df["fraction"].to_numpy(dtype=float)
            d50 = weighted_quantile_value(diameter, mass, 0.50)
            values["D50"] = d50
            values["D_min"] = float(np.nanmin(diameter))
            values["D_max"] = float(np.nanmax(diameter))
        else:
            values["D50"] = self.spin_d50_percentile.value() / 1000.0
            values["D_min"] = self.spin_d16.value() / 8000.0
            values["D_max"] = self.spin_d84.value() * 8.0 / 1000.0
        return model_key, values

    def calibrate_reference_psd(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入独立标定表（shields、psd_linear）", self.last_data_directory,
            "标定表格 (*.csv *.tsv *.txt *.xlsx *.xls)",
        )
        if not path:
            return
        try:
            frame = read_tabular_file(path)
            A, B = SeismicHydraulicModel.calibrate_reference_psd(
                frame["psd_linear"], frame["shields"]
            )
            for key, value in (("A_db", A), ("B_db", B)):
                control = self.empirical_controls[key]
                if not control.minimum() <= value <= control.maximum():
                    raise ValueError(f"{key}={value:g} 超出参数控件范围。")
            self.empirical_controls["A_db"].setValue(A)
            self.empirical_controls["B_db"].setValue(B)
        except (KeyError, ValueError, OSError, ImportError) as exc:
            QMessageBox.warning(self, "参考 PSD 标定失败", f"请检查 shields 和 psd_linear 两列。\n{exc}")
            return
        self.status_bar.showMessage(f"参考 PSD 标定完成，A={A:.5f}、B={B:.5f}", 8000)

    # --------------------------- 6. 计算结果 ---------------------------
    def setup_tab_inversion(self):
        tab = QWidget()
        root = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        root.addWidget(splitter)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        group_exec = QGroupBox("1. 推移质反演计算")
        execute = QVBoxLayout(group_exec)
        resolution_form = QFormLayout()
        self.spin_inversion_resolution = QSpinBox()
        self.spin_inversion_resolution.setRange(0, 60)
        self.spin_inversion_resolution.setSuffix(" min")
        self.spin_inversion_resolution.setSpecialValueText("PSD 原始时间")
        self.spin_inversion_resolution.setValue(
            self.config["inversion"].get("resolution_minutes", 0)
        )
        self.spin_inversion_resolution.setToolTip(
            "固定时间窗对配对有效的线性 PSD 和水力数据取中位数。"
            "窗口不得短于 PSD 输出间隔。"
        )
        resolution_form.addRow("反演时间分辨率", self.spin_inversion_resolution)
        execute.addLayout(resolution_form)
        self.lbl_inversion_psd_interval = QLabel("请先计算 PSD，以确定可用的反演分辨率。")
        self.lbl_inversion_psd_interval.setWordWrap(True)
        execute.addWidget(self.lbl_inversion_psd_interval)
        self.btn_run_inv = QPushButton("启动推移质通量 qb / Qb 反演")
        self.btn_run_inv.clicked.connect(self.on_run_inversion)
        execute.addWidget(self.btn_run_inv)
        panel_layout.addWidget(group_exec)

        group_filter = QGroupBox("2. 瞬态异常处理（原始结果始终保留）")
        filter_form = QFormLayout(group_filter)
        self.chk_filter_flux = QCheckBox("生成局部中值尖峰清洗结果")
        self.chk_filter_flux.setChecked(False)
        self.spin_filter_window = QDoubleSpinBox(); self.spin_filter_window.setRange(1, 1440); self.spin_filter_window.setValue(60)
        self.spin_filter_multiplier = QDoubleSpinBox(); self.spin_filter_multiplier.setRange(1.01, 1000); self.spin_filter_multiplier.setValue(5)
        filter_form.addRow(self.chk_filter_flux)
        filter_form.addRow("滚动窗口 (min)：", self.spin_filter_window)
        filter_form.addRow("尖峰倍数阈值：", self.spin_filter_multiplier)
        panel_layout.addWidget(group_filter)

        group_summary = QGroupBox("3. 结果与有效性诊断")
        summary_layout = QVBoxLayout(group_summary)
        self.txt_summary = QTextEdit(); self.txt_summary.setReadOnly(True)
        summary_layout.addWidget(self.txt_summary)
        panel_layout.addWidget(group_summary)

        group_export = QGroupBox("4. 数据与图像导出")
        export_layout = QVBoxLayout(group_export)
        self.btn_export_excel = QPushButton("导出 Excel / CSV 数据...")
        self.btn_export_excel.clicked.connect(self.on_export_excel)
        self.btn_export_img = QPushButton("导出高分辨率图像包 (PNG)...")
        self.btn_export_img.clicked.connect(self.on_export_images)
        export_layout.addWidget(self.btn_export_excel)
        export_layout.addWidget(self.btn_export_img)
        panel_layout.addWidget(group_export)
        splitter.addWidget(self._flex_scroll(panel, 300))

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.fig_inv = Figure(figsize=(8, 6))
        self.canvas_inv = FigureCanvas(self.fig_inv)
        plot_layout.addWidget(NavigationToolbar(self.canvas_inv, self))
        plot_layout.addWidget(self.canvas_inv)
        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 980])
        self.tabs.addTab(tab, create_scientific_icon("inversion"), "6. 计算结果")

    def on_inversion_completed(self, df_res):
        self.df_result = df_res
        valid = df_res["Qb_mass_kg_s"].replace([np.inf, -np.inf], np.nan).dropna()
        raw = df_res["Qb_mass_kg_s_raw"].replace([np.inf, -np.inf], np.nan).dropna()
        total_mass = float(df_res["cumulative_mass_kg"].iloc[-1]) if len(df_res) else 0.0
        raw_total = float(df_res["cumulative_mass_kg_raw"].iloc[-1]) if len(df_res) else 0.0
        outliers = int(df_res["outlier_flag"].sum())
        actual = df_res.iloc[0]
        model_key = actual["model_name"]
        width_valid = df_res["W_eff_m"].replace(0.0, np.nan).dropna()
        active_fraction = df_res["effective_grain_mass_fraction"].dropna()
        statuses = df_res["inversion_status"].value_counts(dropna=False)
        fallback_count = int(df_res.get("grain_range_fallback_flag", pd.Series(False, index=df_res.index)).sum())
        hydraulic_invalid = int((~df_res.get("hydraulic_valid_flag", pd.Series(False, index=df_res.index))).sum())
        psd_invalid = int((~df_res.get("observed_psd_valid_flag", pd.Series(False, index=df_res.index))).sum())
        zero_reference = int(statuses.get("below_transport_threshold_or_zero_reference", 0))
        below_threshold = int(statuses.get("below_transport_threshold", 0))
        auto_stage = int(df_res.get("auto_stage_conversion_flag", pd.Series(False, index=df_res.index)).sum())
        status_lines = "\n".join(f"  {name}: {count}" for name, count in statuses.items())
        resolution = int(actual["inversion_resolution_minutes"])
        time_description = f"{resolution} min UTC 时间窗" if resolution else "PSD 原始时间"
        settings_lines = [
            f"  反演时间分辨率：{time_description}",
            f"  D50={actual['D50_m']:.6g} m，ρs={actual['rho_s_kg_m3']:g} kg/m³",
            f"  线性 PSD 频带统计量：{actual['band_statistic']}，频率轴：{actual['frequency_grid']}",
            f"  使用断面 Rh：{int(df_res['hydraulic_radius_from_section_flag'].sum())} 块",
        ]
        if resolution:
            settings_lines.append(
                f"  原始 PSD 间隔 {actual['source_psd_interval_s']:.2f} s，"
                f"配对覆盖不足 {int(statuses.get('insufficient_paired_coverage', 0))} 块"
            )
        if model_key != "Luong2026":
            settings_lines.append(f"  ks={actual['roughness_ks_m']:.6g} m，Shields 截断：{actual['clip_tau_c']}")
        if model_key == "Tsai2012":
            stage = "T−1" if actual["tsai_transport_stage"] == "excess" else "T"
            settings_lines.append(f"  Tsai 输移阶段：{stage}")
        elif model_key == "Luong2024":
            settings_lines.append(f"  恢复系数 γ={actual['restitution']:g}")
        else:
            settings_lines.append(
                f"  参考 PSD 标定 A={actual['reference_A_db']:g}、B={actual['reference_B_db']:g}"
            )
        stats_text = (
            f"模型：{model_key}\n"
            f"有效反演块：{len(valid)} / {len(df_res)}\n"
            f"动态 H_eff：{df_res['H_eff_m'].min():.4g}–{df_res['H_eff_m'].max():.4g} m\n"
            f"动态 W_eff：{width_valid.min() if len(width_valid) else np.nan:.4g}–"
            f"{width_valid.max() if len(width_valid) else np.nan:.4g} m\n\n"
            "主结果 Qb\n"
            f"  平均：{valid.mean() if len(valid) else np.nan:.5g} kg/s\n"
            f"  峰值：{valid.max() if len(valid) else np.nan:.5g} kg/s\n"
            f"  累积质量：{total_mass:.6g} kg\n\n"
            "原始结果 Qb_raw\n"
            f"  峰值：{raw.max() if len(raw) else np.nan:.5g} kg/s\n"
            f"  累积质量：{raw_total:.6g} kg\n"
            f"  标记尖峰：{outliers} 个\n\n"
            "有效性诊断\n"
            f"  水力无效/未对齐：{hydraulic_invalid} 块\n"
            f"  PSD 无效：{psd_invalid} 块\n"
            f"  低于输移阈值或单位理论 PSD 为零：{zero_reference} 块\n"
            f"  经验模型低于起动阈值（零通量）：{below_threshold} 块\n"
            f"  动态粒径无交集并回退完整级配：{fallback_count} 块\n"
            f"  自动将水位高程换算为水深：{auto_stage} 块\n"
            f"  活动粒径平均质量比例：{active_fraction.mean() if len(active_fraction) else np.nan:.3f}\n"
            "  状态计数：\n" + status_lines + "\n\n"
            "本次反演参数\n" + "\n".join(settings_lines)
        )
        self.txt_summary.setPlainText(stats_text)
        self.plot_inv_canvas()
        self.progress_bar.setVisible(False)
        if len(valid) == 0:
            self.status_bar.showMessage("反演完成但无有效块，请查看结果页的有效性诊断。", 8000)
        else:
            self.status_bar.showMessage(f"反演完成：{len(valid)} / {len(df_res)} 个有效块。", 6000)

    # --------------------------- 无项目页的通用覆盖 ---------------------------
    def sync_project_summary(self):
        return

    def on_waveform_headers_scanned(self, records):
        _BaseOBSAnalyzerGUI.on_waveform_headers_scanned(self, records)
        if records:
            self.last_data_directory = str(Path(records[0].path).parent)

    def on_psd_computed(self, df_psd, metadata):
        _BaseOBSAnalyzerGUI.on_psd_computed(self, df_psd, metadata)
        step = metadata.nominal_block_seconds
        minimum = max(1, int(np.ceil((step - 1e-6) / 60.0)))
        self.lbl_inversion_psd_interval.setText(
            f"PSD 输出间隔 {step:.2f} s。固定反演分辨率最细为 {minimum} min。"
            if minimum <= 60 else f"PSD 输出间隔 {step:.2f} s。请选择 PSD 原始时间。"
        )

    def on_waveform_processed(self, tr_raw, tr_det, tr_vel):
        self.lbl_inversion_psd_interval.setText("请先计算 PSD，以确定可用的反演分辨率。")
        super().on_waveform_processed(tr_raw, tr_det, tr_vel)

    def on_export_images(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择保存图像文件夹", self.last_data_directory
        )
        if not folder:
            return
        self.last_data_directory = folder
        targets = [
            ("fig_wf", "1_waveform_qc.png"),
            ("fig_river", "2_hydraulic_heff_weff.png"),
            ("fig_grain", "3_grain_distribution.png"),
            ("fig_psd", "4_psd_spectrogram.png"),
            ("fig_inv", "5_inversion_results.png"),
        ]
        saved = []
        for attribute, file_name in targets:
            figure = getattr(self, attribute, None)
            if figure is not None and figure.axes:
                figure.savefig(Path(folder) / file_name, dpi=300, bbox_inches="tight")
                saved.append(file_name)
        QMessageBox.information(self, "图像导出完成", f"已导出 {len(saved)} 幅图至：\n{folder}")

    def show_about(self):
        QMessageBox.about(
            self, "关于 OBS Bedload Analyzer",
            f"<h3>OBS Bedload Analyzer v{APP_VERSION}</h3>"
            "<p>PyQt6 六步专业版：地震波形、水文数据、颗粒粒径、连续 PSD、"
            "Tsai/Luong 三模型与计算结果。</p>"
            "<p><b>确认公式：</b><br>"
            "U=8.1u<sub>*</sub>[H/(3D<sub>50</sub>)]<sup>1/6</sup><br>"
            "U<sub>b</sub>, H<sub>b</sub> 使用 max(T−1,0)<sup>exponent</sup></p>"
            "<p>参考：Tsai et al. (2012)；Luong et al. (2024, 2026)；"
            f"seismic_bedload {UPSTREAM_VERSION}, commit {UPSTREAM_COMMIT}。</p>"
        )


# =============================================================================
# 6. 程序入口
# =============================================================================

# =============================================================================
# 站点资料、流式处理及五步界面
# =============================================================================

import csv
from threading import BoundedSemaphore
from scipy.fft import set_workers as fft_workers
from scipy.signal import firwin, resample_poly
import pyqtgraph as pg
from PyQt6.QtCore import QBuffer, QIODevice, QRectF
from PyQt6.QtGui import QImage, QTransform
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QGraphicsPathItem
from bisect import bisect_left, bisect_right
from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, nullcontext
from functools import cached_property, lru_cache
from io import BytesIO
from itertools import accumulate, zip_longest
from html import escape
from struct import unpack
from tempfile import SpooledTemporaryFile, TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile
from threading import Lock
from zoneinfo import ZoneInfo
from PyQt6.QtCore import QSignalBlocker, QTimeZone


TIME_ZONES = (
    ("北京时间", "Beijing (UTC+08:00)", "Asia/Shanghai"),
    ("协调世界时 UTC", "UTC", "UTC"),
    ("日本标准时间", "Japan", "Asia/Tokyo"),
    ("香港时间", "Hong Kong", "Asia/Hong_Kong"),
    ("欧洲中部时间", "Central Europe", "Europe/Berlin"),
    ("美国山地时间", "US Mountain", "America/Denver"),
)
BEIJING = ZoneInfo("Asia/Shanghai")

OBSERVED_SPECTRUM_SYMBOLS = (
    ("Pobs", "实测地表质点速度功率谱密度 [(m/s)²/Hz]", "Observed ground-velocity PSD [(m/s)²/Hz]"),
    ("Pqᵦ₌₁", "单位单宽体积通量的理论功率谱密度", "Theoretical PSD for unit volumetric flux per unit width"),
    ("𝒮𝓑", "反演频带内的线性功率谱密度统计量", "Linear PSD statistic over the inversion band"),
)

FORMULA_SYMBOLS = {
    "IMHE2026": (
        ("τ*", "实际 Shields 参数，无量纲", "Applied Shields parameter, dimensionless"),
        ("τc*(D)", "粒径 D 对应的临界 Shields 参数，无量纲", "Critical Shields parameter for diameter D, dimensionless"),
        ("Rₕ", "水力半径 [m]", "Hydraulic radius [m]"),
        ("S", "河床坡降，无量纲", "Bed slope, dimensionless"),
        ("R", "颗粒相对浸没密度，无量纲", "Submerged relative sediment density, dimensionless"),
        ("D", "颗粒粒径 [m]", "Grain diameter [m]"),
        ("Uᵦ", "颗粒顺流速度 [m/s]", "Streamwise grain speed [m/s]"),
        ("Pₜ", "模型地表质点速度功率谱密度 [(m/s)²/Hz]", "Modelled ground-velocity PSD [(m/s)²/Hz]"),
        ("f", "频率 [Hz]", "Frequency [Hz]"),
        ("C", "冲击系数，π²G²⟨1/tᴅ⟩/4 [s⁻¹]", "Impact factor, π²G²⟨1/tᴅ⟩/4 [s⁻¹]"),
        ("G", "含恢复系数的方向耦合因子，无量纲", "Directional coupling including restitution, dimensionless"),
        ("tᴅ", "相邻冲击间隔 [s]", "Interval between impacts [s]"),
        ("W", "河宽 [m]", "Channel width [m]"),
        ("qᵦ", "单宽体积通量 [m²/s]", "Volumetric flux per unit width [m²/s]"),
        ("p(D)", "粒径概率密度 [m⁻¹]", "Grain-size probability density [m⁻¹]"),
        ("Vₚ", "单颗粒体积 [m³]", "Volume of one grain [m³]"),
        ("𝒦(f)", "瑞利波传播项，f³χ(β)/(v_c³v_u²)", "Rayleigh-wave propagation term, f³χ(β)/(v_c³v_u²)"),
        ("v_c", "瑞利波相速度 [m/s]", "Rayleigh-wave phase velocity [m/s]"),
        ("v_u", "瑞利波群速度 [m/s]", "Rayleigh-wave group velocity [m/s]"),
        ("χ(β)", "瑞利波衰减函数，无量纲", "Rayleigh-wave attenuation function, dimensionless"),
        ("β", "传播衰减函数的无量纲自变量", "Dimensionless argument of the attenuation function"),
        *OBSERVED_SPECTRUM_SYMBOLS,
    ),
    "Tsai2012": (
        ("P(f)", "模型地表质点速度功率谱密度 [(m/s)²/Hz]", "Modelled ground-velocity PSD [(m/s)²/Hz]"),
        ("f", "频率 [Hz]", "Frequency [Hz]"),
        ("D", "颗粒粒径 [m]", "Grain diameter [m]"),
        ("p(D)", "粒径概率密度 [m⁻¹]", "Grain-size probability density [m⁻¹]"),
        ("Ṅ(D)", "单位河长内该粒级的冲击率", "Impact rate per unit channel length for size D"),
        ("𝒦ₜ", "单次冲击经瑞利波传至测点的功率响应", "Single-impact Rayleigh-wave power response at the receiver"),
        ("U", "平均流速 [m/s]", "Mean flow speed [m/s]"),
        ("u*", "剪切速度 [m/s]", "Shear velocity [m/s]"),
        ("Uᵦ", "颗粒顺流速度 [m/s]", "Streamwise grain speed [m/s]"),
        ("H", "水深 [m]", "Water depth [m]"),
        ("Hᵦ", "输移层厚度 [m]", "Transport-layer thickness [m]"),
        ("kₛ", "河床粗糙度 [m]", "Bed roughness [m]"),
        ("T_eff", "有效输移阶段，无量纲", "Effective transport stage, dimensionless"),
        ("qᵦ", "单宽体积通量 [m²/s]", "Volumetric flux per unit width [m²/s]"),
        *OBSERVED_SPECTRUM_SYMBOLS,
    ),
    "Luong2024": (
        ("P(f)", "模型地表质点速度功率谱密度 [(m/s)²/Hz]", "Modelled ground-velocity PSD [(m/s)²/Hz]"),
        ("f", "频率 [Hz]", "Frequency [Hz]"),
        ("D", "颗粒粒径 [m]", "Grain diameter [m]"),
        ("W", "河宽 [m]", "Channel width [m]"),
        ("p(D)", "粒径概率密度 [m⁻¹]", "Grain-size probability density [m⁻¹]"),
        ("tᴅ", "相邻冲击间隔 [s]", "Interval between impacts [s]"),
        ("pₜ(tᴅ)", "冲击间隔概率密度 [s⁻¹]", "Impact-interval probability density [s⁻¹]"),
        ("Vₚ", "单颗粒体积 [m³]", "Volume of one grain [m³]"),
        ("Uᵦ", "颗粒顺流速度 [m/s]", "Streamwise grain speed [m/s]"),
        ("v_c", "瑞利波相速度 [m/s]", "Rayleigh-wave phase velocity [m/s]"),
        ("v_u", "瑞利波群速度 [m/s]", "Rayleigh-wave group velocity [m/s]"),
        ("I", "冲量 [N·s]", "Impulse [N·s]"),
        ("N", "方向耦合因子，无量纲", "Directional coupling, dimensionless"),
        ("𝒦ᴿ(f)", "瑞利波传播项", "Rayleigh-wave propagation term"),
        ("ρₛ", "颗粒密度 [kg/m³]", "Sediment density [kg/m³]"),
        ("χ(β)", "瑞利波衰减函数，无量纲", "Rayleigh-wave attenuation function, dimensionless"),
        ("β", "传播衰减函数的无量纲自变量", "Dimensionless argument of the attenuation function"),
        ("qᵦ", "单宽体积通量 [m²/s]", "Volumetric flux per unit width [m²/s]"),
        *OBSERVED_SPECTRUM_SYMBOLS,
    ),
    "Luong2026": (
        ("qᵦ*", "无量纲推移质通量", "Dimensionless bedload flux"),
        ("τ*", "实际 Shields 参数，无量纲", "Applied Shields parameter, dimensionless"),
        ("τcr*", "起动临界 Shields 参数，无量纲", "Critical Shields parameter for entrainment, dimensionless"),
        ("[x]₊", "取 x 与 0 的较大值", "Positive part of x, max(x, 0)"),
        ("Rₕ", "水力半径 [m]", "Hydraulic radius [m]"),
        ("D₅₀", "中值粒径 [m]", "Median grain diameter [m]"),
        ("S", "河床坡降，无量纲", "Bed slope, dimensionless"),
        ("R", "颗粒相对浸没密度，无量纲", "Submerged relative sediment density, dimensionless"),
        ("P", "实测线性功率谱密度 [(m/s)²/Hz]", "Observed linear PSD [(m/s)²/Hz]"),
        ("Pᵣₑ", "同等剪切应力下的线性参考功率谱密度 [(m/s)²/Hz]", "Reference linear PSD at the same shear stress [(m/s)²/Hz]"),
        ("P/Pᵣₑ", "观测与参考功率谱密度之比，无量纲", "Ratio of observed to reference PSD, dimensionless"),
        ("k", "场地标定的尺度系数，无量纲", "Calibrated site scale factor, dimensionless"),
        ("m", "功率比指数，无量纲", "Power-ratio exponent, dimensionless"),
        ("n", "超额剪切应力指数，无量纲", "Excess-shear exponent, dimensionless"),
        ("A", "参考谱标定斜率 [dB]", "Reference-PSD calibration slope [dB]"),
        ("B", "参考谱标定截距 [dB]", "Reference-PSD calibration intercept [dB]"),
        ("P_dB", "实测功率谱密度 [dB]", "Observed PSD [dB]"),
        ("Pᵣₑ,dB", "参考功率谱密度 [dB]", "Reference PSD [dB]"),
    ),
}

STATIONS = {
    "PZ12": {
        "bed_m": 2892.0,
        "width": {
            2892.01: 0.0, 2897.01: 49.311, 2902.01: 145.986, 2907.01: 168.342,
            2912.01: 191.814, 2917.01: 207.483, 2922.01: 216.913, 2927.01: 237.419,
        },
    },
    "MT11": {
        "bed_m": 710.6,
        "width": {
            710.6: 0.0, 715.6: 108.953, 720.6: 124.806, 725.6: 134.041,
            730.6: 145.638, 735.6: 156.47, 740.6: 168.647, 745.6: 185.113,
            750.6: 205.808,
        },
    },
}


def default_response_prefilter(fs: float) -> tuple[float, float, float, float]:
    """Scale the upper response taper with Nyquist while preserving the 100 Hz preset."""
    fs = _require_positive("采样率 / sample rate", fs)
    return min(0.2, 0.02 * fs), min(0.5, 0.05 * fs), 0.35 * fs, 0.45 * fs


def parse_hydrology(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Read stage elevation and discharge, retaining the original rows for preview."""
    if frame.empty or frame.shape[1] < 3:
        raise ValueError("水文表需要时间、水位和流量三列 / Expected time, stage and discharge.")
    columns = frame.columns.astype(str).str.strip()
    frame = frame.set_axis(columns, axis=1)
    time_col = detect_table_column(columns, ["time", "datetime", "timestamp", "时间", "日期"], 0)
    stage_col = detect_table_column(columns, ["water_level", "stage", "level", "水位", "高程"], 1)
    flow_col = detect_table_column(columns, ["discharge", "flow_rate", "flow", "q", "流量"], 2)
    if len({time_col, stage_col, flow_col}) != 3:
        raise ValueError("水文表列名有歧义 / Ambiguous hydrology columns.")
    source_time = frame[time_col]
    numeric_time = pd.to_numeric(source_time, errors="coerce")
    if pd.api.types.is_numeric_dtype(source_time) or (
        not pd.api.types.is_datetime64_any_dtype(source_time)
        and numeric_time.notna().any()
        and numeric_time.notna().sum() == source_time.notna().sum()
        and numeric_time.dropna().between(20000, 80000).all()
    ):
        times = pd.to_datetime(numeric_time, unit="D", origin="1899-12-30", errors="coerce")
    else:
        times = parse_datetime_values(source_time)
    try:
        utc = ensure_utc_index(times, timezone)
    except ValueError:
        # Parse explicit mixed offsets individually without relocalizing them.
        utc = pd.DatetimeIndex([
            pd.NaT if pd.isna(value) else ensure_utc_index([value], timezone)[0]
            for value in times
        ], tz="UTC")
    result = pd.DataFrame(
        {
            "water_level_m": clean_numeric_series(frame[stage_col]).to_numpy(),
            "discharge_m3_s": clean_numeric_series(frame[flow_col]).to_numpy(),
        },
        index=utc,
    )
    result = result.loc[~result.index.isna()]
    if not result["water_level_m"].notna().any():
        raise ValueError("没有有效的水位与时间 / No valid stage and time observations.")
    return result


def rated_hydraulics(frame: pd.DataFrame, station: str) -> pd.DataFrame:
    """Interpolate the surveyed rating curve within its measured domain."""
    setup = STATIONS[station]
    items = sorted(setup["width"].items())
    stages, widths = np.asarray(items, dtype=float).T
    result = frame.copy()
    level = result["water_level_m"].to_numpy(dtype=float)
    result["H_eff_m"] = np.maximum(level - setup["bed_m"], 0.0)
    result["W_eff_m"] = np.interp(level, stages, widths, left=np.nan, right=np.nan)
    result["W_eff_m"] = result["W_eff_m"].where(result["H_eff_m"] != 0.0, 0.0)
    return result


def paired_hydrology(
    frame: pd.DataFrame, centers: pd.DatetimeIndex, station: str, cadence=None,
) -> pd.DataFrame:
    """Align UTC observations without extrapolating or bridging long gaps."""
    valid = frame.loc[frame["water_level_m"].notna()] if cadence is None else frame
    if valid.empty:
        return rated_hydraulics(frame.iloc[:0].reindex(centers), station)
    index = valid.index
    right = index.searchsorted(centers)
    right_sample = np.minimum(right, len(index) - 1)
    exact = index[right_sample] == centers
    left_sample = np.maximum(right - 1, 0)
    span = (index[right_sample] - index[left_sample]).total_seconds().to_numpy()
    if cadence is None:
        cadence = np.median((index[1:] - index[:-1]).total_seconds()) if len(index) > 1 else 0.0
    within = exact | ((right > 0) & (right < len(index)) & (span <= 3 * cadence))
    frac = np.divide(
        (centers - index[left_sample]).total_seconds(), span,
        out=np.zeros(len(centers)), where=(span > 0),
    )
    values = valid.to_numpy(dtype=float, copy=False)
    paired = values[left_sample] + frac[:, None] * (values[right_sample] - values[left_sample])
    paired[exact] = values[right_sample[exact]]
    paired[~within] = np.nan
    return rated_hydraulics(pd.DataFrame(paired, index=centers, columns=valid.columns), station)


_FALLBACK_READ_LOCK = Lock()


def _read_binary_sac_slice(record, left, right):
    """Read only the requested samples of a standard evenly spaced SAC trace."""
    with open(record.path, "rb") as handle:
        if os.fstat(handle.fileno()).st_size != 632 + 4 * record.npts:
            return None
        handle.seek(280)
        header = handle.read(160)
        if len(header) != 160:
            return None
        for order in ("<", ">"):
            fields = unpack(order + "40i", header)
            if (fields[6], fields[9], fields[15], fields[35]) != (6, record.npts, 1, 1):
                continue
            handle.seek(632 + 4 * left)
            data = handle.read(4 * (right - left))
            if len(data) != 4 * (right - left):
                raise OSError(f"SAC data ended unexpectedly: {record.path}")
            return np.frombuffer(data, dtype=order + "f4")
    return None


def epoch_seconds(values):
    return pd.DatetimeIndex(values).to_numpy(dtype="datetime64[ns]").view("i8") / 1e9


def calendar_tick_pattern(seconds):
    if seconds >= 86399:
        return "%Y-%m-%d"
    if seconds >= 59.999:
        return "%Y-%m-%d\n%H:%M"
    if seconds >= 0.999:
        return "%Y-%m-%d\n%H:%M:%S"
    return "%Y-%m-%d\n%H:%M:%S.%f"


class CalendarDateFormatter(mdates.DateFormatter):
    def set_locs(self, locs):
        super().set_locs(locs)
        gaps = np.diff(locs) * 86400
        gaps = gaps[gaps > 0]
        self.fmt = calendar_tick_pattern(float(gaps.min())) if gaps.size else "%Y-%m-%d"


class TimeAxis(pg.AxisItem):
    def __init__(self, zone):
        super().__init__("bottom")
        self.zone = zone

    def tickStrings(self, values, scale, spacing):
        pattern = calendar_tick_pattern(spacing)
        return [pd.Timestamp(value, unit="s", tz="UTC").tz_convert(self.zone).strftime(pattern)
                for value in values]


WAVEFORM_LINE_WIDTH = 2.0
DATA_LINE_WIDTH = 2.3
RESULT_LINE_WIDTH = 1.5
HYDROLOGY_LINE_WIDTH = 1.5
AXIS_LINE_WIDTH = 1.30


def zero_missing_for_plot(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, 0.0)


def psd_display_db(power, floor):
    """Map zero linear power to the colour floor while keeping its validity mask."""
    positive = np.isfinite(power) & (power > 0)
    db = np.zeros(power.shape, dtype=np.float32)
    np.log10(power, out=db, where=positive)
    db *= 10
    db[~positive] = floor
    return db, positive


def bin_edges(centers, half_width):
    if len(centers) == 1:
        return centers[0] + np.array([-half_width, half_width])
    half_gaps = np.diff(centers) / 2
    return np.r_[centers[0] - half_gaps[0], centers[:-1] + half_gaps, centers[-1] + half_gaps[-1]]


class DataPlotPanel(pg.GraphicsLayoutWidget):
    """Reusable curves, linked axes and exportable Qt graphics scene."""

    def __init__(self):
        super().__init__()
        self.resize(820, 360)
        self.plots = []
        self.time_plot = None
        self.gap_items = {}
        self.overlays = []
        self.dark = False
        self.message = None

    def reset(self):
        for plot, view, update in self.overlays:
            plot.vb.sigResized.disconnect(update)
            view.setXLink(None)
            self.scene().removeItem(view)
        self.overlays.clear()
        for plot, (_, update) in self.gap_items.items():
            plot.vb.sigYRangeChanged.disconnect(update)
        self.clear()
        self.plots.clear()
        self.time_plot = None
        self.gap_items.clear()
        self.message = None

    def theme(self, dark):
        self.dark = dark
        self.setBackground("#0c0d0e" if dark else "#ffffff")
        for plot in self.plots:
            self.style_plot(plot)
        if self.message is not None:
            self.message.setText(self.message.text, color=self.foreground)

    @property
    def foreground(self):
        return "#f1f3f4" if self.dark else "#0f172a"

    def style_plot(self, plot):
        for side in ("top", "bottom", "left", "right"):
            self.style_axis(plot.getAxis(side))
        plot.showGrid(x=True, y=True, alpha=0.18)

    def style_axis(self, axis):
        spine = "#92989d" if self.dark else "#526171"
        axis.setPen(pg.mkPen(spine, width=AXIS_LINE_WIDTH))
        axis.setTextPen(self.foreground)
        axis.setTickFont(QFont("Microsoft YaHei", 10))
        axis.label.setDefaultTextColor(QColor(self.foreground))

    def add_chart(self, row, zone=None):
        axes = {"bottom": TimeAxis(zone)} if zone is not None else None
        plot = self.addPlot(row=row, col=0, axisItems=axes)
        self.plots.append(plot)
        if zone is not None and self.time_plot is None:
            self.time_plot = plot
        self.style_plot(plot)
        return plot

    def autoRange(self):
        for plot in self.plots:
            plot.enableAutoRange()
        for _, view, _ in self.overlays:
            view.enableAutoRange(axis="y")

    def show_message(self, text):
        self.reset()
        self.message = self.addLabel(text, color=self.foreground)

    def right_axis(self, plot, label):
        view = pg.ViewBox(enableMenu=False)
        view.setMouseEnabled(x=False, y=False)
        self.scene().addItem(view)
        plot.showAxis("right")
        plot.setLabel("right", label)
        plot.getAxis("right").linkToView(view)
        view.setXLink(plot)
        def update():
            view.setGeometry(plot.vb.sceneBoundingRect())
            view.linkedViewChanged(plot.vb, view.XAxis)
        plot.vb.sigResized.connect(update)
        self.overlays.append((plot, view, update))
        update()
        return view

    def mark_gaps(self, plot, intervals):
        if plot in self.gap_items:
            item, _ = self.gap_items[plot]
        else:
            item = QGraphicsPathItem()
            item.setPen(pg.mkPen(None))
            item.setBrush(pg.mkBrush(146, 152, 157, 82))
            item.setZValue(-8)
            plot.addItem(item, ignoreBounds=True)

            def update(*_):
                lo, hi = plot.vb.viewRange()[1]
                item.setTransform(QTransform.fromScale(1, hi - lo))
                item.setPos(0, lo)
            self.gap_items[plot] = item, update
            plot.vb.sigYRangeChanged.connect(update)
            update()
        path = QPainterPath()
        if intervals:
            bounds = epoch_seconds([value for pair in intervals for value in pair]).reshape(-1, 2)
            for start, end in bounds:
                path.addRect(start, 0, end - start, 1)
        item.setPath(path)


class DecimatedWaveform:
    """Centred half-band FIR stages on the recording's fixed UTC sample grid."""
    taps = firwin(129, 0.45, window=("kaiser", 8.6))

    def __init__(self, native, factor):
        self.native = native
        self.factor = factor
        self.fs = native.fs / factor
        self.origin, self.start, self.end = native.origin, native.start, native.end
        self.trace_id = native.trace_id

    def hours(self):
        return pd.date_range(self.start.floor("h"), self.end.ceil("h"), freq="h", inclusive="left")

    def sample(self, timestamp):
        return self.native.sample(timestamp) / self.factor

    def timestamp(self, indices):
        return self.native.timestamp(np.asarray(indices) * self.factor)

    def fingerprint(self, padding_seconds):
        return ("halfband129-v1", self.factor, self.start, self.end,
                self.native.fingerprint(padding_seconds + 64 * (self.factor - 1) / self.native.fs))

    def read_samples(self, first, stop, *, return_mask=False):
        # Whole output samples of halo keep every stage on the same decimation phase.
        halo = 64
        start = (first - halo) * self.factor
        end = (stop + halo) * self.factor
        values, valid = self.native.read_samples(start, end, return_mask=True)
        if not valid.any():
            empty = np.zeros(max(0, stop-first))
            return (empty, np.zeros(empty.size, dtype=bool)) if return_mask else empty
        factor = self.factor
        while factor > 1:
            filtered = resample_poly(values, 1, 2, window=self.taps)
            # At output j the FIR support is [2*j-64, 2*j+64].
            # Mark intervals instead of allocating full-length integer prefix sums.
            next_valid = np.zeros(len(filtered), dtype=bool)
            observed_runs = np.flatnonzero(np.diff(valid, prepend=False, append=False))
            for left, right in observed_runs.reshape(-1, 2):
                if right - left >= self.taps.size:
                    next_valid[(left + 65) // 2:(right - 63) // 2] = True
            valid = next_valid
            filtered[~valid] = 0
            values = filtered
            factor //= 2
        data, mask = values[halo:-halo], valid[halo:-halo]
        return (data, mask) if return_mask else data


class MiniSeedReadCache:
    """Decode eligible MiniSEED files once per task, with an in-memory budget."""

    def __init__(self, records):
        self.sizes = {}
        for record in records:
            if record.format_name == "MSEED":
                self.sizes[record.path] = self.sizes.get(record.path, 0) + 8 * record.npts
        available = available_physical_memory()
        self.budget = min(512 * 1024**2, available // 16) if available else 128 * 1024**2
        self.entries = OrderedDict()
        self.reserved = 0
        self.lock = Lock()

    def read(self, path, start, end):
        size = self.sizes.get(path, 0)
        owner = False
        future = None
        if 0 < size <= self.budget:
            with self.lock:
                if path in self.entries:
                    future, _ = self.entries[path]
                    self.entries.move_to_end(path)
                else:
                    for old in tuple(self.entries):
                        if self.reserved + size <= self.budget:
                            break
                        candidate, used = self.entries[old]
                        if candidate.done():
                            self.entries.pop(old)
                            self.reserved -= used
                    if self.reserved + size <= self.budget:
                        owner = True
                        future = Future()
                        self.entries[path] = future, size
                        self.reserved += size
        if future is None:
            return read(path, format="MSEED", starttime=start, endtime=end, nearest_sample=False)
        if owner:
            try:
                future.set_result(read(path, format="MSEED"))
            except Exception as exc:
                future.set_exception(exc)
                with self.lock:
                    if path in self.entries:
                        _, used = self.entries.pop(path)
                        self.reserved -= used
        return future.result()


class WaveformInputError(OSError):
    def __init__(self, path, reason):
        super().__init__(f"{path}: {reason}")
        self.issue = WaveformReadIssue(path, str(reason))


class StreamingWaveform:
    """Read bounded waveform windows on a common UTC sample grid."""

    def __init__(self, records, start, end):
        self.io_slots = BoundedSemaphore(2)
        self.mseed_cache = None
        self.prefetched = {}
        self.start = _as_utc_timestamp(start)
        self.end = _as_utc_timestamp(end)
        if self.end <= self.start:
            raise ValueError("目标区间无效 / End time must follow start time.")
        records = list(records)
        selected = records_overlapping_window(records, self.start, self.end)
        # A selection wholly inside a gap still has a time axis and sample rate.
        candidates = selected or records
        trace_ids = {r.trace_id for r in candidates if r.channel.upper().endswith("Z")}
        trace_ids = trace_ids or {r.trace_id for r in candidates}
        if len(trace_ids) != 1:
            raise ValueError("目标区间须有一个通道 / Select a window with one vertical channel.")
        self.trace_id = trace_ids.pop()
        self.records = [r for r in records if r.trace_id == self.trace_id]
        self.is_hydrophone = any(r.channel.upper() == "HYD" or Path(r.path).suffix.casefold() == ".hyd"
                                 for r in self.records)
        rates = {r.sampling_rate_hz for r in candidates if r.trace_id == self.trace_id}
        if len(rates) != 1:
            raise ValueError("目标区间采样率不一致 / Sample rate changes within the selected interval.")
        self.fs = rates.pop()
        self.origin = min(r.start_utc for r in self.records if r.sampling_rate_hz == self.fs)

    def sample(self, timestamp):
        return (timestamp - self.origin).value * self.fs / 1e9

    def timestamp(self, indices):
        return self.origin + pd.to_timedelta(np.asarray(indices, dtype=float) / self.fs, unit="s")

    @cached_property
    def _time_index(self):
        indexed = sorted(
            ((i, r) for i, r in enumerate(self.records) if r.sampling_rate_hz == self.fs),
            key=lambda item: item[1].start_utc,
        )
        starts = [r.start_utc for _, r in indexed]
        ends = list(accumulate((r.end_utc for _, r in indexed), max))
        return indexed, starts, ends

    def _records_between(self, start, end):
        indexed, starts, ends = self._time_index
        lo, hi = bisect_left(ends, start), bisect_right(starts, end)
        # Preserve the original precedence when multiple files overlap.
        return dict((r.path, r) for _, r in sorted(
            item for item in indexed[lo:hi] if item[1].end_utc >= start
        ))

    def fingerprint(self, padding_seconds):
        margin = pd.Timedelta(seconds=padding_seconds)
        records = self._records_between(self.start - margin, self.end + margin)
        files = []
        for path in records:
            try:
                files.append((path, os.stat(path)))
            except OSError as exc:
                raise WaveformInputError(path, exc) from exc
        return (self.trace_id, self.fs, self.origin, self.start, self.end,
                tuple((path, info.st_size, info.st_mtime_ns, info.st_ctime_ns) for path, info in files))

    def read_samples(self, first, stop, *, return_mask=False):
        future = self.prefetched.pop((first, stop), None) if return_mask else None
        if future is not None:
            return future.result()
        with self.io_slots:
            return self._read_window(first, stop, return_mask)

    def _read_window(self, first, stop, return_mask):
        """Snap equal-rate traces to the nearest grid point and zero-fill holes."""
        data = np.zeros(max(0, stop - first), dtype=float)
        observed = np.zeros(len(data), dtype=bool) if return_mask else None
        if stop <= first:
            return (data, observed) if return_mask else data
        # Read one extra sample on each side so an off-grid sample that snaps
        # into the requested interval is not lost by ObsPy's time trimming.
        start, end = self.timestamp(first - 1), self.timestamp(stop)
        records = self._records_between(start, end)
        a = obspy.UTCDateTime(ns=int(start.value))
        b = obspy.UTCDateTime(ns=int(end.value))

        def grid_index(start_ns):
            sample = (start_ns - self.origin.value) * self.fs / 1e9
            return int(np.floor(sample + 0.5 + 1e-7))

        def copy_samples(index, samples):
            offset = index - first
            left = max(0, offset)
            right = min(len(data), offset + len(samples))
            if right > left:
                part = samples[left - offset:right - offset]
                finite = np.isfinite(np.asarray(part))
                if np.ma.isMaskedArray(part):
                    finite &= ~np.ma.getmaskarray(part)
                np.copyto(data[left:right], np.asarray(part), where=finite)
                if observed is not None:
                    observed[left:right] |= finite

        for path, record in records.items():
            try:
                if record.format_name == "SAC" and record.start_ns is not None:
                    offset = grid_index(record.start_ns)
                    left, right = max(0, first - offset), min(record.npts, stop - offset)
                    if right <= left:
                        continue
                    samples = _read_binary_sac_slice(record, left, right)
                    if samples is not None:
                        copy_samples(offset + left, samples)
                        continue
                if record.format_name == "MSEED":
                    stream = (self.mseed_cache.read(path, a, b) if self.mseed_cache is not None
                              else read(path, format="MSEED", starttime=a, endtime=b, nearest_sample=False))
                else:
                    # The SAC warning filter changes process-wide state on Python 3.13.
                    with _FALLBACK_READ_LOCK:
                        stream = _read_obspy_waveform(read, path, starttime=a, endtime=b, nearest_sample=False)
                for trace in stream:
                    if waveform_trace_id(path, trace, record.format_name) == self.trace_id and trace.stats.sampling_rate == self.fs:
                        copy_samples(grid_index(trace.stats.starttime.ns), trace.data)
            except Exception as exc:
                raise WaveformInputError(path, exc) from exc
        return (data, observed) if return_mask else data

    def hours(self):
        return pd.date_range(self.start.floor("h"), self.end.ceil("h"), freq="h", inclusive="left")


class PAZResponseCorrection:
    """Reuse the ObsPy PAZ inverse and frequency taper on identical FFT grids."""

    def __init__(self, fs, paz, prefilter):
        self.fs = fs
        self.paz = paz
        self.prefilter = prefilter
        available = available_physical_memory()
        self.limit = min(96 * 1024**2, available // 25) if available else 0
        self.lock = Lock()
        self.response = lru_cache(maxsize=2)(self._response)

    def _response(self, nfft):
        from obspy.signal.invsim import cosine_taper, invert_spectrum, paz_to_freq_resp

        response, frequencies = paz_to_freq_resp(self.paz["poles"], self.paz["zeros"], self.paz["gain"],
                                                1 / self.fs, nfft, freq=True)
        taper = cosine_taper(frequencies.size, freqs=frequencies, flimit=self.prefilter)
        invert_spectrum(response, 600.0)
        return taper, response

    def __call__(self, values, taper_fraction):
        from obspy.signal.invsim import cosine_taper, simulate_seismometer

        nfft = None
        if self.limit and len(values) >= 600 * self.fs:
            try:
                from obspy.signal.util import _npts2nfft
                nfft = _npts2nfft(len(values))
            except ImportError:
                pass
        if nfft is None or (nfft // 2 + 1) * 24 > self.limit:
            return simulate_seismometer(values, self.fs, paz_remove=self.paz, remove_sensitivity=True,
                                       pre_filt=self.prefilter, taper_fraction=taper_fraction)

        from obspy.signal.detrend import simple

        data = values.astype(np.float64)
        data -= data.mean()
        data *= cosine_taper(len(data), taper_fraction)
        spectrum = np.fft.rfft(data, n=nfft)
        with self.lock:
            taper, response = self.response(nfft)
        spectrum *= taper
        spectrum *= response
        spectrum[-1] = abs(spectrum[-1]) + 0j
        corrected = simple(np.fft.irfft(spectrum)[:len(data)])
        corrected /= self.paz["sensitivity"]
        return corrected


def correct_stream_block(raw, fs, paz, prefilter, guard, keep_detrended=True, *, observed=None, corrector=None):
    """Correct uninterrupted runs independently and invalidate their filter edges."""
    from scipy.signal import detrend
    from obspy.signal.invsim import simulate_seismometer

    detrended = np.full_like(raw, np.nan) if keep_detrended else None
    velocity = np.full_like(raw, np.nan)
    valid_samples = np.isfinite(raw) if observed is None else observed
    bounds = np.flatnonzero(np.diff(valid_samples, prepend=False, append=False))
    for left, right in bounds.reshape(-1, 2):
        size = right - left
        if size < 3:
            continue
        data = detrend(raw[left:right], type="linear")
        if detrended is not None:
            detrended[left:right] = data
        if size <= 2 * guard:
            continue
        taper = min(0.05, guard / size)
        if corrector is None:
            corrected = simulate_seismometer(data, fs, paz_remove=paz, remove_sensitivity=True,
                                              pre_filt=prefilter, taper_fraction=taper)
        else:
            corrected = corrector(data, taper)
        velocity[left + guard:right - guard] = corrected[guard:-guard]
    return detrended, velocity


def _sample_bounds(source, start, stop):
    return tuple(int(np.ceil(source.sample(t) - 1e-6)) for t in (start, stop))


def psd_overlap_samples(nperseg, fraction):
    if fraction not in (0.05, 0.25, 0.5):
        raise ValueError("PSD 重叠率须为 5%、25% 或 50% / Select 5%, 25% or 50% overlap.")
    return int(round(nperseg * fraction))


def psd_window(source, hour, nperseg, prefilter, overlap):
    fs = source.fs
    active_start = max(hour, source.start)
    active_end = min(hour + pd.Timedelta(hours=1), source.end)
    first, stop = _sample_bounds(source, active_start, active_end)
    noverlap = psd_overlap_samples(nperseg, overlap)
    hop = nperseg - noverlap
    half = nperseg // 2
    center_samples = np.arange(((first + hop - 1) // hop) * hop, stop, hop)
    selected_a, selected_b = _sample_bounds(source, source.start, source.end)
    center_samples = center_samples[
        (center_samples - half >= selected_a) & (center_samples + half <= selected_b)
    ]
    if center_samples.size == 0:
        return None
    guard = max(1, int(np.ceil(max(30.0, 4.0 / prefilter[1]) * fs)))
    start = min(first, int(center_samples[0] - half)) - guard
    end = max(stop, int(center_samples[-1] + half)) + guard
    return active_start, active_end, noverlap, half, center_samples, guard, start, end


def stream_hour_psd(source, hour, nperseg, minutes, paz, prefilter, band, *, overlap=0.5, corrector=None):
    """Compute zero-padded short spectra, retaining independent coverage."""
    fs = source.fs
    interval = pd.Timedelta(minutes=minutes)
    starts = pd.date_range(hour, periods=60 // minutes, freq=interval)
    centers = starts + interval / 2
    full_frequency = np.fft.rfftfreq(nperseg, 1.0 / fs)
    keep = (full_frequency >= band[0]) & (full_frequency <= band[1])
    frequency = full_frequency[keep]
    if frequency.size < 2:
        raise ValueError("PSD 频带至少需要两个频点 / PSD band needs two bins.")
    psd = pd.DataFrame(0.0, index=centers, columns=frequency)
    seconds = np.zeros(len(starts))
    plan = psd_window(source, hour, nperseg, prefilter, overlap)
    if plan is None:
        return psd, seconds
    active_start, active_end, noverlap, half, center_samples, guard, start, end = plan
    raw, observed = source.read_samples(start, end, return_mask=True)
    _, velocity = correct_stream_block(
        raw, fs, paz, prefilter, guard, keep_detrended=False, observed=observed, corrector=corrector,
    )
    valid = np.isfinite(velocity)
    cumul = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    offsets = center_samples - half - start
    # A partially observed FFT window uses zeros in the missing slots. A
    # wholly missing window supplies no evidence for the PSD median.
    usable = cumul[offsets + nperseg] - cumul[offsets] > 0
    if not np.any(usable):
        return psd, seconds
    chunk = velocity[offsets[0]:offsets[-1] + nperseg]
    _, _, spectrum = spectrogram(
        np.where(np.isfinite(chunk), chunk, 0.0), fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density",
    )
    labels = starts.searchsorted(source.timestamp(center_samples), side="right") - 1
    selected_spectra = spectrum[keep]
    for i, t in enumerate(starts):
        a, b = _sample_bounds(source, max(t, active_start), min(t + interval, active_end))
        if b <= a:
            continue
        available = (cumul[b - start] - cumul[a - start]) / fs
        members = labels == i
        good = members & usable
        if available > 0 and good.any():
            psd.iloc[i] = np.median(selected_spectra[:, good], axis=1)
            seconds[i] = min(available, minutes * 60.0)
    return psd, seconds


def invert_stream_bins(psd, waveform_seconds, hydro, seconds_per_bin, params):
    """Invert PSD bins once and compute mass only over observed time."""
    result = invert_bedload_timeseries_enhanced(psd, hydro, params)
    rates = result["Qb_mass_kg_s_raw"].to_numpy(dtype=float)
    dry = (hydro["H_eff_m"].to_numpy() == 0.0) & (waveform_seconds > 0)
    rates = np.where(dry, 0.0, rates)
    good = (waveform_seconds > 0) & np.isfinite(rates) & (rates >= 0)
    rates = np.where(good, rates, np.nan)
    seconds = np.where(good, np.minimum(waveform_seconds, seconds_per_bin), 0.0)
    return pd.DataFrame({
        "water_depth_m": hydro["H_eff_m"].to_numpy(),
        "channel_width_m": hydro["W_eff_m"].to_numpy(),
        "discharge_m3_s": hydro["discharge_m3_s"].to_numpy(),
        "Qb_kg_s": rates,
        "peak_Qb_kg_s": rates,
        "bedload_mass_kg": rates * seconds,
        "valid_seconds": seconds,
        "coverage_fraction": seconds / seconds_per_bin,
        "valid_subwindows": good.astype(int),
        "waveform_valid_seconds": waveform_seconds,
        "missing_hydro_subwindows": (
            (waveform_seconds > 0) & ~np.isfinite(hydro["W_eff_m"].to_numpy())
        ).astype(int),
        "zero_reference_subwindows": (
            (waveform_seconds > 0)
            & result["inversion_status"].eq("below_transport_threshold_or_zero_reference").to_numpy()
        ).astype(int),
    }, index=psd.index - pd.Timedelta(seconds=seconds_per_bin / 2))


def summarize_stream_period(bins, period_seconds):
    """Aggregate flux by valid time, preserving mass, peaks and missing coverage."""
    covered = float(bins["valid_seconds"].sum())
    mass = float(bins["bedload_mass_kg"].sum(min_count=1))
    return {
        "water_depth_m": float(bins["water_depth_m"].mean()),
        "channel_width_m": float(bins["channel_width_m"].mean()),
        "discharge_m3_s": float(bins["discharge_m3_s"].mean()),
        "Qb_kg_s": mass / covered if covered else np.nan,
        "peak_Qb_kg_s": float(bins["peak_Qb_kg_s"].max()),
        "bedload_mass_kg": mass,
        "valid_seconds": covered,
        "coverage_fraction": covered / period_seconds,
        "valid_subwindows": int(bins["valid_subwindows"].sum()),
        "waveform_valid_seconds": float(bins["waveform_valid_seconds"].sum()),
        "missing_hydro_subwindows": int(bins["missing_hydro_subwindows"].sum()),
        "zero_reference_subwindows": int(bins["zero_reference_subwindows"].sum()),
    }


def summarize_hour(psd, waveform_seconds, hydro, seconds_per_bin, params):
    return summarize_stream_period(
        invert_stream_bins(psd, waveform_seconds, hydro, seconds_per_bin, params), 3600,
    )


def _display_extrema(values, limit):
    """Per-bucket extrema and explicit gap markers, with a bounded output size."""
    if len(values) <= limit:
        return np.arange(len(values)), values
    observed = np.isfinite(values)
    count = max(1, limit // 4) if not observed.all() else max(1, limit // 3)
    edges = np.linspace(0, len(values), count + 1, dtype=int)
    picks = []
    for left, right in zip(edges[:-1], edges[1:]):
        part = values[left:right]
        mask = observed[left:right]
        finite = np.flatnonzero(mask)
        extrema = set()
        if finite.size:
            clean = part[finite]
            extrema.update((finite[np.argmin(clean)], finite[np.argmax(clean)]))
        if finite.size < part.size:
            # Keep both edges of a missing span for the zero-level preview.
            missing = ~mask
            extrema.update((int(missing.argmax()), part.size - 1 - int(missing[::-1].argmax())))
        picks.extend(left + int(j) for j in sorted(extrema))
    indices = np.asarray(picks, dtype=int)
    return indices, values[indices]


PSD_ARCHIVE_FORMAT = "obs-bedload-psd-v2"
PSD_ARCHIVE_FILTER = "OBS PSD (*.obspsd)"
PSD_PERIODS = (1, 5, 30)


def psd_archive_metadata(source, options, station):
    native = getattr(source, "native", source)
    return {
        "format": PSD_ARCHIVE_FORMAT, "units": "(m/s)^2/Hz",
        "start_utc": source.start.isoformat(), "end_utc": source.end.isoformat(),
        "trace_id": source.trace_id, "station": station,
        "sample_rate_hz": source.fs, "input_sample_rate_hz": native.fs,
        "decimation_factor": getattr(source, "factor", 1),
        "minutes": options["minutes"], "nperseg": options["nperseg"], "overlap": options["overlap"],
        "prefilter": options["prefilter"], "band": full_psd_band(source.fs, options["nperseg"]),
        "paz": {**options["paz"], "poles": [str(z) for z in options["paz"]["poles"]],
                "zeros": [str(z) for z in options["paz"]["zeros"]]},
    }


def full_psd_band(fs, nperseg):
    return 0.0, float(np.fft.rfftfreq(nperseg, 1 / fs)[-1])


def inversion_passband(source, prefilter):
    lower, upper = prefilter[1:3]
    if getattr(source, "factor", 1) > 1:
        upper = min(upper, .4 * source.fs)
    return lower, upper


def psd_block_name(hour):
    return f"hours/{hour.value}.npy"


class PSDArchiveSource:
    """Read a portable archive of linear PSD and per-bin valid duration on demand."""

    def __init__(self, path):
        self.path = Path(path)
        self.fingerprint = self.file_fingerprint()
        with ZipFile(self.path) as archive:
            info = archive.getinfo("metadata.json")
            if info.file_size > 64 * 1024:
                raise ValueError("PSD 元数据过大 / PSD metadata is too large.")
            self.metadata = data = json.loads(archive.read(info))
            if data["format"] != PSD_ARCHIVE_FORMAT or data["units"] != "(m/s)^2/Hz":
                raise ValueError("PSD 文件格式或单位不兼容 / Unsupported PSD format or units.")
            self.start, self.end = (_as_utc_timestamp(data[key]) for key in ("start_utc", "end_utc"))
            self.trace_id = str(data["trace_id"])
            self.fs = float(data["sample_rate_hz"])
            self.input_fs = float(data["input_sample_rate_hz"])
            self.factor = int(data["decimation_factor"])
            nperseg, minutes = int(data["nperseg"]), int(data["minutes"])
            prefilter = tuple(map(float, data["prefilter"]))
            self.band = tuple(map(float, data["band"]))
            if (
                self.end <= self.start or not np.isfinite(self.fs) or not np.isfinite(self.input_fs)
                or not 0 < self.fs <= self.input_fs or self.factor not in (1, 2, 4, 8, 16)
                or not np.isclose(self.fs * self.factor, self.input_fs)
                or not 2 <= nperseg <= 262144 or minutes not in PSD_PERIODS
                or len(prefilter) != 4 or not 0 < prefilter[0] < prefilter[1] < prefilter[2] < prefilter[3] < self.fs / 2
                or self.band != full_psd_band(self.fs, nperseg)
            ):
                raise ValueError("PSD 元数据无效 / Invalid PSD metadata.")
            psd_overlap_samples(nperseg, float(data["overlap"]))
            grid = np.fft.rfftfreq(nperseg, 1 / self.fs)
            self.frequencies = grid[(grid >= self.band[0]) & (grid <= self.band[1])]
            if len(self.frequencies) < 2:
                raise ValueError("PSD 文件频点不足 / PSD archive has too few frequency bins.")
            self.rows = 60 // minutes
            hour_count = (self.end.ceil("h") - self.start.floor("h")) // pd.Timedelta(hours=1)
            if len(archive.infolist()) != hour_count + 1:
                raise ValueError("PSD 小时块数量不完整 / The PSD archive has missing or extra blocks.")
            limit = self.rows * (len(self.frequencies) + 1) * 8 + 2048
            for hour in self.hours():
                if archive.getinfo(psd_block_name(hour)).file_size > limit:
                    raise ValueError("PSD 小时块大小无效 / Invalid PSD block size.")

    def hours(self):
        return pd.date_range(self.start.floor("h"), self.end.ceil("h"), freq="h", inclusive="left")

    def file_fingerprint(self):
        info = self.path.stat()
        return info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def options(self, band, workers):
        data = self.metadata
        paz = data["paz"]
        return {
            "nperseg": int(data["nperseg"]), "minutes": int(data["minutes"]), "overlap": float(data["overlap"]),
            "prefilter": tuple(map(float, data["prefilter"])), "band": band, "workers": workers,
            "paz": {**paz, "poles": [complex(z) for z in paz["poles"]], "zeros": [complex(z) for z in paz["zeros"]]},
        }

    def read(self, archive, hour):
        with BytesIO(archive.read(psd_block_name(hour))) as chunk:
            values = self.read_array(chunk, (self.rows, len(self.frequencies)))
            coverage = self.read_array(chunk, (self.rows,))
            if chunk.read(1):
                raise ValueError("PSD 小时块包含多余数据 / Unexpected data in PSD block.")
        minutes = self.metadata["minutes"]
        if (not np.isfinite(values).all() or np.any(values < 0)
                or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > minutes * 60))):
            raise ValueError("PSD 小时块内容无效 / Invalid PSD values or coverage.")
        return values, coverage

    @staticmethod
    def read_array(chunk, expected_shape):
        if np.lib.format.read_magic(chunk) != (1, 0):
            raise ValueError("PSD 数组版本不兼容 / Unsupported PSD array version.")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(chunk)
        if shape != expected_shape or dtype.kind != "f" or dtype.itemsize != 8:
            raise ValueError("PSD 数组维度或类型无效 / Invalid PSD array shape or type.")
        count = int(np.prod(expected_shape))
        data = chunk.read(count * dtype.itemsize)
        if len(data) != count * dtype.itemsize:
            raise ValueError("PSD 数组不完整 / Incomplete PSD array.")
        return np.frombuffer(data, dtype=dtype).reshape(expected_shape, order="F" if fortran else "C")


@contextmanager
def write_psd_archive(path, metadata, interrupted):
    """Write hours to a new archive and discard an incomplete result."""
    with path.open("xb") as output:
        try:
            with ZipFile(output, "w", compression=ZIP_STORED) as archive:
                archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, allow_nan=False))
                yield archive
        except BaseException:
            output.close()
            path.unlink(missing_ok=True)
            raise
    if interrupted():
        path.unlink(missing_ok=True)


class HourlyPSDCache:
    """Temporary full-frequency spectra for one waveform and processing setup."""

    def __init__(self, max_bytes=4 * 1024**3):
        self.max_bytes = max_bytes
        self.directory = None
        self.key = None
        self.bytes_used = 0
        self.can_write = True

    def select(self, key):
        if self.key != key:
            self.close()
            self.key = key

    def close(self):
        if self.directory is not None:
            try:
                self.directory.cleanup()
            except OSError:
                # A temporary-file cleanup failure must not prevent analysis.
                pass
            self.directory = None
        self.key = None
        self.bytes_used = 0
        self.can_write = True

    def load(self, hour):
        if self.directory is None:
            return None
        path = Path(self.directory.name) / f"{hour.value}.npy"
        try:
            with path.open("rb") as handle:
                return np.load(handle, allow_pickle=False), np.load(handle, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None

    def save(self, hour, values, coverage):
        if not self.has_space(values.nbytes + coverage.nbytes):
            return
        path = None
        try:
            if self.directory is None:
                self.directory = TemporaryDirectory(prefix="obs-bedload-psd-")
            path = Path(self.directory.name) / f"{hour.value}.npy"
            previous_size = path.stat().st_size if path.exists() else 0
            with path.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
                np.save(handle, coverage, allow_pickle=False)
                self.bytes_used += handle.tell() - previous_size
        except OSError:
            self.can_write = False
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def has_space(self, data_bytes):
        return self.can_write and self.bytes_used + data_bytes + 512 <= self.max_bytes


def response_settings_key(options):
    paz = options["paz"]
    return (tuple(options["prefilter"]), tuple(paz["poles"]), tuple(paz["zeros"]),
            paz["gain"], paz["sensitivity"])


PROCESSING_THREADS = (1, 2, 4, 8, 16, 32, 64)


def logical_cpu_count():
    return getattr(os, "process_cpu_count", os.cpu_count)() or 1


def available_physical_memory():
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("size", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    "total_physical", "available_physical", "total_page_file", "available_page_file",
                    "total_virtual", "available_virtual", "available_extended_virtual",
                )
            ]

        status = MemoryStatus()
        status.size = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available_physical
    elif hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            pass
    return None


class StreamingWorker(QThread):
    progress_sig = pyqtSignal(int, str)
    finished_sig = pyqtSignal(object)
    error_sig = pyqtSignal(str)
    partial_sig = pyqtSignal(str, object)

    def __init__(self, source, options, task, settings=None, *, psd_cache=None, waveform_cache=None):
        super().__init__()
        self.source = source
        self.options = options
        self.task = task
        self.settings = settings
        self.result_cache = None
        self.psd_cache = psd_cache
        self.waveform_cache = waveform_cache
        self.max_workers = options.get("workers", 1)
        self.active_workers = 1
        self.fft_threads = 1
        self.done_hours = 0
        self.started_at = time.monotonic()
        self.rate_history = deque([(self.started_at, 0)], maxlen=24)
        self.last_preview = self.last_progress = 0.0
        self.through = None
        self.input_issue = None

    def run(self):
        try:
            result = self._waveforms() if self.task == "waveforms" else self._spectra()
            self.finished_sig.emit(result)
        except WaveformInputError as exc:
            self.input_issue = exc.issue
        except Exception as exc:
            self.error_sig.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _progress(self, step, count):
        now = time.monotonic()
        if now - self.last_progress < .35 and step < count:
            return
        self.last_progress = now
        done = self.done_hours
        before, n_before = self.rate_history[0]
        if done != self.rate_history[-1][1]:
            self.rate_history.append((now, done))
        eta = (now - before) * (count - done) / (done - n_before) if done > n_before else None
        label = "Computed" if self.options.get("english") else "已计算"
        text = f"{label} {done}/{count} h"
        if self.through is not None:
            label = "Continuous through" if self.options.get("english") else "已连续处理至"
            text += f" · {label} {self.through.tz_convert(BEIJING):%m-%d %H:%M} UTC+8"
        if done > n_before and now > before:
            text += f" · {(done - n_before) * 60 / (now - before):.1f} h/min"
        if eta is not None:
            text += f" · ETA {int(eta)//60:02d}:{int(eta)%60:02d}"
        self.progress_sig.emit(round(100 * step / count), text)

    def _ordered_hours(self, hours, compute, next_read=None):
        cpu_budget = min(self.max_workers, logical_cpu_count())
        workers = min(cpu_budget, len(hours))
        native = getattr(self.source, "native", self.source)
        available = available_physical_memory()
        if workers > 1:
            if available:
                guard_seconds = max(30.0, 4.0 / self.options["prefilter"][1])
                duration = 3600 + 2 * guard_seconds
                analysis_bytes = 192 * (duration * self.source.fs + self.options["nperseg"])
                decimation_bytes = 48 * duration * native.fs
                block_bytes = int(max(analysis_bytes, decimation_bytes)) + 32 * 1024**2
                workers = min(workers, max(1, (2 * available // 3) // block_bytes))
        self.active_workers = max(1, workers)
        if hasattr(native, "io_slots"):
            native.io_slots = BoundedSemaphore(max(1, min(workers, 4)))
        if self.task != "waveforms" and workers:
            # Every outer task may run a batch of short FFTs, within one CPU budget.
            self.fft_threads = min(8, cpu_budget // workers)
        if len(hours) and not self.isInterruptionRequested():
            self._progress(0, len(hours))
        # Keep BLAS settings identical for serial and parallel runs.
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            native_threads = nullcontext()
        else:
            native_threads = threadpool_limits(limits=1, user_api="blas")
        with native_threads:
            if workers <= 1:
                prefetch_ok = not available or available > 192 * (3600 * native.fs) + 256 * 1024**2
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="obs-read") as reader:
                    try:
                        for i, hour in enumerate(hours):
                            if self.isInterruptionRequested():
                                break
                            if prefetch_ok and next_read and hasattr(native, "prefetched") and i + 1 < len(hours):
                                span = next_read(hours[i + 1])
                                if span is not None:
                                    first, stop = span
                                    if isinstance(self.source, DecimatedWaveform):
                                        first = (first - 64) * self.source.factor
                                        stop = (stop + 64) * self.source.factor
                                    native.prefetched[first, stop] = reader.submit(native._read_window, first, stop, True)
                            value = compute(hour)
                            self.done_hours += 1
                            yield hour, value
                    finally:
                        if hasattr(native, "prefetched"):
                            native.prefetched.clear()
                return
            pending = {}
            ready = {}
            next_output = next_submit = 0
            count = len(hours)
            max_buffer = 3 * workers
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="obs-hour")
            try:
                while next_output < count and not self.isInterruptionRequested():
                    limit = min(count, next_output + max_buffer)
                    while len(pending) < workers and next_submit < limit:
                        future = pool.submit(compute, hours[next_submit])
                        pending[future] = next_submit
                        next_submit += 1
                    if next_output in ready:
                        index = next_output
                        next_output += 1
                        yield hours[index], ready.pop(index).result()
                    else:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        self.done_hours += len(done)
                        self._progress(next_output, count)
                        for future in done:
                            ready[pending.pop(future)] = future
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

    def _waveforms(self):
        source = self.source
        hours = source.hours()
        fs = source.fs
        paz, prefilter = self.options["paz"], self.options["prefilter"]
        raw_only = getattr(getattr(source, "native", source), "is_hydrophone", False)
        guard = 0 if raw_only else max(1, int(np.ceil(max(30.0, 4.0 / prefilter[1]) * fs)))
        corrector = None if raw_only else PAZResponseCorrection(fs, paz, prefilter)
        traces = [[] for _ in range(1 if raw_only else 3)]
        pixels = self.options.get("pixels", 1600)
        limit = max(4, 6 * pixels // len(hours))
        cache = self.waveform_cache
        key = (source.fingerprint((guard + 2) / fs), limit, response_settings_key(self.options))
        if cache is not None and cache.get("key") == key:
            return cache["series"]

        def snapshot():
            return [(pd.to_datetime(np.concatenate([t for t, _ in pieces]), unit="ns", utc=True),
                     np.concatenate([v for _, v in pieces])) for pieces in traces if pieces]

        def next_read(hour):
            first, stop = _sample_bounds(source, max(hour, source.start),
                                         min(hour + pd.Timedelta(hours=1), source.end))
            return first - guard, stop + guard

        def compute(hour):
            left, right = next_read(hour)
            first, stop = left + guard, right - guard
            raw, observed = source.read_samples(left, right, return_mask=True)
            signals = [np.where(observed, raw, np.nan)]
            if not raw_only:
                signals.extend(correct_stream_block(raw, fs, paz, prefilter, guard, observed=observed, corrector=corrector))
            pieces = []
            for signal in signals:
                indices, values = _display_extrema(signal[guard:guard + stop - first], limit)
                times = source.timestamp(first + indices).to_numpy(dtype="datetime64[ns]").view("i8")
                pieces.append((times, values))
            return pieces

        for i, (hour, series) in enumerate(self._ordered_hours(hours, compute, next_read)):
            for pieces, reduced in zip(traces, series):
                pieces.append(reduced)
            self.through = min(source.end, hour + pd.Timedelta(hours=1))
            if time.monotonic() - self.last_preview >= 1:
                self.partial_sig.emit("waveforms", snapshot())
                self.last_preview = time.monotonic()
            self._progress(i + 1, len(hours))
        result = snapshot()
        if cache is not None and not self.isInterruptionRequested():
            cache.update(key=key, series=result)
        return result

    def _hour_spectra(self, hours, nperseg, minutes, paz, prefilter, band, overlap, transform=None):
        source = self.source
        imported = isinstance(source, PSDArchiveSource)
        cache = None if imported else self.psd_cache
        corrector = None if imported else PAZResponseCorrection(source.fs, paz, prefilter)
        grid = np.fft.rfftfreq(nperseg, 1 / source.fs)
        full_band = float(grid[0]), float(grid[-1])
        cache_band = full_band if imported or self.task in {"psd", "psd_archive"} else tuple(map(float, band))
        if cache is not None:
            padding = max(30.0, 4.0 / prefilter[1]) + (nperseg // 2 + 3) / source.fs
            base_key = (source.fingerprint(padding), response_settings_key(self.options), nperseg, minutes, overlap)
            if cache.key == (base_key, full_band):
                cache_band = full_band
            cache.select((base_key, cache_band))
        frequencies = grid[(grid >= cache_band[0]) & (grid <= cache_band[1])]
        left, right = frequencies.searchsorted(band[0]), frequencies.searchsorted(band[1], side="right")
        selected = slice(left, right)
        if right - left < 2:
            raise ValueError("PSD 频带至少需要两个频点 / PSD band needs at least two bins.")
        cache_bytes = (60 // minutes) * (len(frequencies) + 1) * np.dtype(np.float64).itemsize

        def compute(hour):
            if imported:
                full = source.read(self.psd_reader, hour)
            else:
                full = cache.load(hour) if cache is not None else None
            newly_computed = False
            if full is None:
                retain = self.psd_writer is not None or (cache is not None and cache.has_space(cache_bytes))
                with fft_workers(self.fft_threads):
                    frame, coverage = stream_hour_psd(
                        source, hour, nperseg, minutes, paz, prefilter,
                        cache_band if retain else band, overlap=overlap, corrector=corrector,
                    )
                if retain:
                    full = frame.to_numpy(dtype=np.float64), coverage
                    newly_computed = True
            if full is not None:
                values, coverage = full
                index = (pd.date_range(hour, periods=60 // minutes, freq=pd.Timedelta(minutes=minutes))
                         + pd.Timedelta(minutes=minutes / 2))
                frame = pd.DataFrame(values[:, selected], index=index, columns=frequencies[selected])
            result = (frame, coverage) if transform is None else transform(frame, coverage)
            return result, full, newly_computed

        def next_read(hour):
            if cache is not None and cache.directory is not None:
                if (Path(cache.directory.name) / f"{hour.value}.npy").exists():
                    return None
            plan = psd_window(source, hour, nperseg, prefilter, overlap)
            return None if plan is None else plan[-2:]

        for hour, (result, full, newly_computed) in self._ordered_hours(
            hours, compute, None if imported else next_read,
        ):
            if newly_computed and cache is not None:
                cache.save(hour, *full)
            if self.psd_writer is not None:
                with BytesIO() as buffer:
                    for array in full:
                        np.save(buffer, array, allow_pickle=False)
                    self.psd_writer.writestr(psd_block_name(hour), buffer.getvalue())
            yield hour, result

    def _spectra(self):
        settings = self.settings or {}
        destination = settings.get("psd_output")
        imported = isinstance(self.source, PSDArchiveSource)
        with ExitStack() as stack:
            self.psd_reader = None
            self.psd_writer = None
            if imported:
                if self.source.file_fingerprint() != self.source.fingerprint:
                    raise ValueError("PSD 文件已变更，请重新导入 / The PSD file changed. Import it again.")
                self.psd_reader = stack.enter_context(ZipFile(self.source.path))
            if destination is not None:
                if settings.get("new_export_folder"):
                    folder = reserve_export_folder(destination)
                    stack.callback(remove_empty_export_folder, folder)
                    destination = folder / destination.name
                metadata = (self.source.metadata if imported else
                            psd_archive_metadata(self.source, self.options, settings["station"]))
                self.psd_writer = stack.enter_context(
                    write_psd_archive(destination, metadata, self.isInterruptionRequested),
                )
            if self.task == "psd_archive":
                nperseg, minutes, paz, prefilter, band = (
                    self.options[key] for key in ("nperseg", "minutes", "paz", "prefilter", "band")
                )
                hours = self.source.hours()
                for i, (hour, _) in enumerate(self._hour_spectra(
                    hours, nperseg, minutes, paz, prefilter, band, self.options.get("overlap", .5),
                )):
                    self.through = min(self.source.end, hour + pd.Timedelta(hours=1))
                    self._progress(i + 1, len(hours))
                return None if self.isInterruptionRequested() else destination
            return self._spectra_task()

    def _spectra_task(self):
        if self.task in {"hourly", "inversion"}:
            path = self.settings.get("path")
            if path:
                with open(path, "x", encoding="utf-8-sig", newline="") as handle:
                    return self._process_hours(handle)
            # Keep full results incrementally. Long runs spill to temporary
            # storage, while plotting only needs the bounded preview.
            cache = SpooledTemporaryFile(
                max_size=8 * 1024**2, mode="w+t", encoding="utf-8-sig", newline="",
            )
            try:
                result = self._process_hours(cache)
                cache.seek(0)
            except BaseException:
                cache.close()
                raise
            self.result_cache = cache
            return result
        return self._process_hours()

    def _process_hours(self, handle=None):
        source = self.source
        nperseg, minutes, paz, prefilter, band = (
            self.options[key] for key in ("nperseg", "minutes", "paz", "prefilter", "band")
        )
        overlap = self.options.get("overlap", 0.5)
        hours = source.hours()
        display_stride = max(1, int(np.ceil(len(hours) * (60 // minutes) / 1600)))
        pending_overview = None
        spectra = []
        live_offset = 0
        rows = []
        tail = deque(maxlen=200)
        day_bins = []
        current_day = None
        processed_end = source.start
        writer = None
        total_mass = 0.0
        row_count = 0
        transform = None
        if handle is not None:
            settings = self.settings
            result_minutes = settings.get("result_minutes", 60)
            if result_minutes not in {1, 60, 1440} or result_minutes % minutes:
                raise ValueError("输出分辨率与 PSD 周期不兼容 / Output and PSD intervals are incompatible.")
            # Hourly rows are the common source for all displayed and exported scales.
            result_stride = 1 if result_minutes == 60 else max(1, (len(hours) * 60 // result_minutes + 7999) // 8000)
            water = settings["water"][["water_level_m", "discharge_m3_s"]].dropna(subset=["water_level_m"])
            water = pd.DataFrame(water.to_numpy(dtype=float), index=water.index, columns=water.columns)
            cadence = (np.median((water.index[1:] - water.index[:-1]).total_seconds())
                       if len(water) > 1 else 0.0)

            def transform(frame, coverage):
                hydro = paired_hydrology(water, frame.index, settings["station"], cadence)
                bins = invert_stream_bins(frame, coverage, hydro, minutes * 60, settings["params"])
                return bins.loc[(bins.index < source.end)
                                & (bins.index + pd.Timedelta(minutes=minutes) > source.start)]

        def append_overview(frame=None, final=False):
            nonlocal pending_overview
            if frame is not None:
                pending_overview = frame if pending_overview is None else pd.concat((pending_overview, frame))
            if pending_overview is None:
                return
            count = len(pending_overview)
            stop = count if final else count - count % display_stride
            if stop == 0:
                return
            group = pending_overview.iloc[:stop]
            pending_overview = pending_overview.iloc[stop:].copy() if stop < count else None
            if display_stride == 1:
                preview = group
            else:
                buckets = group.groupby(np.arange(stop) // display_stride)
                preview = buckets.median()
                first = np.arange(0, stop, display_stride)
                last = np.minimum(first + display_stride - 1, stop - 1)
                starts = group.index[first]
                preview.index = starts + (group.index[last] - starts) / 2
            spectra.append(preview.astype(np.float32))

        def write_period(start, fields):
            nonlocal writer, total_mass, row_count
            # Preserve missingness in the scientific calculation and print zeros in the final output.
            fields = {key: value if np.isfinite(value) else 0 for key, value in fields.items()}
            total_mass += fields["bedload_mass_kg"]
            local = start.tz_convert(BEIJING)
            end = local + pd.Timedelta(minutes=result_minutes)
            row = {
                "start_beijing": local.isoformat(),
                "end_beijing": end.isoformat(),
                "result_minutes": result_minutes,
                "station": settings["station"],
                "model": settings["params"]["model_type"],
                "model_name": (
                    "IMHE model 2026" if settings["params"]["model_type"] == "IMHE2026"
                    else settings["params"]["model_type"]
                ),
                "psd_minutes": minutes,
                "sample_rate_hz": source.fs,
                "fft_window_samples": nperseg,
                "fft_overlap_fraction": overlap,
                "fft_overlap_samples": psd_overlap_samples(nperseg, overlap),
                "prefilter_hz": " ".join(f"{value:g}" for value in prefilter),
                "inversion_fmin_hz": band[0],
                "inversion_fmax_hz": band[1],
                **fields,
                "cumulative_mass_kg": total_mass,
            }
            if result_minutes == 60:
                row["hour_start_beijing"] = row["start_beijing"]
                row["hour_end_beijing"] = row["end_beijing"]
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            if row_count % result_stride == 0:
                rows.append(row)
            tail.append(row)
            row_count += 1

        for i, (hour, result) in enumerate(self._hour_spectra(
            hours, nperseg, minutes, paz, prefilter, band, overlap, transform,
        )):
            processed_end = min(source.end, hour + pd.Timedelta(hours=1))
            if handle is None:
                frame, _ = result
                frame = frame.copy()
                frame.loc[np.asarray(result[1]) <= 0] = np.nan
                freq_step = max(1, int(np.ceil(len(frame.columns) / 1024)))
                if freq_step > 1:
                    cuts = np.arange(0, len(frame.columns), freq_step)
                    reduced = np.fmax.reduceat(frame.to_numpy(), cuts, axis=1)
                    frame = pd.DataFrame(reduced, index=frame.index, columns=frame.columns[cuts])
                append_overview(frame)
            else:
                bins = result
                if result_minutes == 1:
                    for start, fields in zip(bins.index, bins.to_dict("records")):
                        write_period(start, fields)
                elif result_minutes == 60:
                    write_period(hour, summarize_stream_period(bins, 3600))
                else:
                    local_hour = hour.tz_convert(BEIJING)
                    current_day = local_hour.normalize()
                    day_bins.append(bins)
                    if local_hour.hour == 23:
                        write_period(current_day, summarize_stream_period(pd.concat(day_bins), 86400))
                        day_bins.clear()
                handle.flush()
            self.through = processed_end
            if time.monotonic() - self.last_preview >= 1:
                if handle is None and live_offset < len(spectra):
                    chunk = pd.concat(spectra[live_offset:])
                    stride = max(1, len(chunk.columns) // 400)
                    if not chunk.empty:
                        self.partial_sig.emit("psd", chunk.iloc[:, ::stride].copy())
                    live_offset = len(spectra)
                elif rows:
                    self.partial_sig.emit("inversion", pd.DataFrame(rows + list(tail)).drop_duplicates("start_beijing"))
                self.last_preview = time.monotonic()
            self._progress(i + 1, len(hours))
        if handle is not None:
            if day_bins:
                write_period(current_day, summarize_stream_period(pd.concat(day_bins), 86400))
                handle.flush()
            result = pd.DataFrame(rows + list(tail))
            if not result.empty:
                result = result.drop_duplicates("start_beijing").sort_values("start_beijing")
            result.attrs["total_rows"] = row_count
            result.attrs["partial"] = self.isInterruptionRequested()
            result.attrs["selection_start"] = source.start
            result.attrs["selection_end"] = source.end
            result.attrs["model_parameters"] = model_export_parameters(settings["params"]) + (
                ("input_sample_rate_Hz", getattr(source, "input_fs", getattr(source, "native", source).fs)),
                ("analysis_sample_rate_Hz", source.fs),
                ("decimation_factor", getattr(source, "factor", 1)),
                ("hann_samples", nperseg), ("overlap_fraction", overlap),
                ("PSD_minutes", minutes), ("prefilter_Hz", str(tuple(prefilter))),
                ("response_gain", paz["gain"]), ("response_sensitivity", paz["sensitivity"]),
                ("PSD_source", source.path.name if isinstance(source, PSDArchiveSource) else "waveforms"),
                ("result_minutes", result_minutes),
            )
            return result
        append_overview(final=True)
        preview = pd.concat(spectra) if spectra else pd.DataFrame()
        preview.attrs.update(time_start=source.start, time_end=processed_end, psd_minutes=minutes)
        return preview


KG_PER_TONNE = 1000.0
RESULT_SCALES = (
    ("1h", "h", "小时尺度", "Hourly"),
    ("1d", "D", "日尺度", "Daily"),
    ("1month", "MS", "月尺度", "Monthly"),
)
DISPLAY_RESULT_SCALES = RESULT_SCALES[:2]
EXPORT_COLUMNS = (
    "Station",
    "UTC+8 Time (1h)", "Qb_kg_s (1h)", "Cumulative_mass_t (1h)",
    "UTC+8 Time (1day)", "Qb_kg_s (1day)", "Cumulative_mass_t (1day)",
    "UTC+8 Time (1month)", "Qb_kg_s (1month)", "Cumulative_mass_t (1month)",
    "Parameter", "Value",
)


def aggregate_inversion_scales(hourly):
    """Aggregate observed mass and valid duration using Beijing calendar boundaries."""
    hourly = hourly.set_index(pd.to_datetime(hourly["start_beijing"], utc=True).dt.tz_convert(BEIJING))
    hourly.index.name = "period_start"
    fields = ["bedload_mass_kg", "valid_seconds", "Qb_kg_s", "cumulative_mass_kg"]
    scales = {"1h": hourly[fields]}
    for code, rule, _, _ in RESULT_SCALES[1:]:
        result = hourly[["bedload_mass_kg", "valid_seconds"]].resample(rule).sum()
        result["Qb_kg_s"] = np.divide(
            result["bedload_mass_kg"].to_numpy(), result["valid_seconds"].to_numpy(),
            out=np.zeros(len(result)), where=result["valid_seconds"].to_numpy() > 0,
        )
        result["cumulative_mass_kg"] = result["bedload_mass_kg"].cumsum()
        scales[code] = result
    return scales
EXPORT_FORMATS = {"CSV (*.csv)": ".csv", "TXT (*.txt)": ".txt", "Excel (*.xlsx)": ".xlsx"}


def model_export_parameters(params):
    model = params["model_type"]
    items = [("Model", "IMHE model 2026" if model == "IMHE2026" else model)]
    for key in ("D50", "slope", "f_inv_min", "f_inv_max"):
        items.append((key, params[key]))
    items.extend((("D50_unit", "m"), ("D50_source", params.get("D50_source", "provided")),
                  ("band_statistic", params.get("band_statistic", "median"))))
    if params.get("grain_percentiles") is not None:
        d16, _, d84 = params["grain_percentiles"]
        items.extend((("D16_m", d16), ("D84_m", d84)))
    sediment = _dataclass_from_mapping(SedimentParams, params.get("sediment_params", {}))
    items.extend((f"sediment.{key}", value) for key, value in asdict(sediment).items())
    if model == "Luong2026":
        items.extend(params.get("empirical_params", {}).items())
        return tuple(items)

    items.extend((("r0", params["r0"]), ("tau_c50", params.get("tau_c50", 0.045)),
                  ("clip_tau_c", params.get("clip_tau_c", model in MULTIMODE_TYPES))))
    ks = params.get("ks")
    items.extend((("ks", 3 * params["D50"] if ks is None else ks),
                  ("ks_source", "3*D50" if ks is None else "manual")))
    seismic = _dataclass_from_mapping(SeismicParams, params.get("seismic_params", {}))
    seismic_items = asdict(seismic)
    seismic_items.pop("vc0_override")
    seismic_items.pop("zeta_override")
    seismic_items.update(
        vc0=seismic.vc0, zeta=seismic.zeta,
        vc0_source="auto" if seismic.vc0_override is None else "manual",
        zeta_source="auto" if seismic.zeta_override is None else "manual",
    )
    items.extend((f"seismic.{key}", value if value is not None else "off")
                 for key, value in seismic_items.items())
    if model == "Tsai2012":
        items.append(("transport_stage_form", params.get("transport_stage_form", "excess")))
    else:
        for key, default in (("hop_t_min", 0.01), ("hop_t_max", 0.50), ("hop_log_median", 0.08),
                             ("hop_log_sigma", 1.0), ("hop_points", 70)):
            items.append((key, params.get(key, default)))
    grains = params.get("grain_distribution")
    if grains is None:
        items.extend((key, params[key]) for key in ("grain_log_std", "grain_points", "D_min", "D_max")
                     if key in params and params[key] is not None)
        return tuple(items)
    for key, values in grains.items():
        values = np.asarray(values, dtype=float)
        for start in range(0, len(values), 1000):
            name = f"grain_distribution.{key}"
            if len(values) > 1000:
                name += f"[{start}:{min(start + 1000, len(values))}]"
            items.append((name, json.dumps(values[start:start + 1000].tolist(), separators=(",", ":"))))
    return tuple(items)


def result_export_stem(station, selection_start, minutes):
    start = selection_start.tz_convert(BEIJING)
    interval = {1: "1min", 60: "1h", 1440: "1d", None: "1h_1d_1month"}[minutes]
    return f"OBS_{station}_UTC+8_{start:%Y%m%d_%H%M%S}_{interval}"


def export_result_cache(cache, path, parameters, scales=None):
    """Write the complete result series and parameter list side by side."""
    cache.seek(0)
    path = Path(path)
    suffix = path.suffix.lower()
    excel = suffix == ".xlsx"
    if suffix not in EXPORT_FORMATS.values():
        raise ValueError(f"Unsupported export format: {suffix}")

    def scale_rows(code):
        if scales is not None:
            frame = scales[code]
            for stamp, rate, total in frame[["Qb_kg_s", "cumulative_mass_kg"]].itertuples():
                yield stamp.isoformat(), rate, total / KG_PER_TONNE

    def rows():
        reader = csv.DictReader(cache)
        first = next(reader, None)
        station = first["station"] if first is not None else None

        def hourly_rows():
            row = first
            while row is not None:
                yield row["start_beijing"], float(row["Qb_kg_s"]), float(row["cumulative_mass_kg"]) / KG_PER_TONNE
                row = next(reader, None)

        missing = (None, None, None)
        for hour, day, month, parameter in zip_longest(
            hourly_rows(), scale_rows("1d"), scale_rows("1month"), parameters,
        ):
            yield (station, *(hour or missing), *(day or missing), *(month or missing), *(parameter or (None, None)))

    kwargs = {} if excel else {"encoding": "utf-8-sig", "newline": ""}
    with path.open("xb" if excel else "x", **kwargs) as output:
        try:
            if excel:
                from openpyxl import Workbook

                workbook = Workbook(write_only=True)
                rows_per_sheet = 1_048_575
                for index, row in enumerate(rows()):
                    if index % rows_per_sheet == 0:
                        sheet = workbook.create_sheet(f"Results {index // rows_per_sheet + 1}")
                        sheet.append(EXPORT_COLUMNS)
                    sheet.append(row)
                workbook.save(output)
            else:
                writer = csv.writer(output, delimiter="\t" if suffix == ".txt" else ",")
                writer.writerow(EXPORT_COLUMNS)
                writer.writerows(rows())
            output.flush()
        except BaseException:
            output.close()
            path.unlink(missing_ok=True)
            raise


def reserve_export_folder(path):
    parent, stem = path.parent, path.stem
    index = 1
    while True:
        target = parent / (stem if index == 1 else f"{stem}_{index:02d}")
        try:
            target.mkdir()
            return target
        except FileExistsError:
            index += 1


def remove_empty_export_folder(folder):
    try:
        folder.rmdir()
    except OSError:
        pass


class ResultExportWorker(QThread):
    progress_sig = pyqtSignal(int, str)
    finished_sig = pyqtSignal(object)
    error_sig = pyqtSignal(str)

    def __init__(self, cache, path, images, parameters, scales=None):
        super().__init__()
        self.cache = cache
        self.path = Path(path)
        self.images = images
        self.parameters = parameters
        self.scales = scales

    def run(self):
        written = []
        folder = None
        try:
            folder = reserve_export_folder(self.path)
            data_path = folder / self.path.name
            export_result_cache(self.cache, data_path, self.parameters, self.scales)
            written.append(data_path)
            for name, png, pdf in self.images:
                for suffix, data in ((".png", png), (".pdf", pdf)):
                    path = folder / (name + suffix)
                    with path.open("xb") as output:
                        written.append(path)
                        output.write(data)
        except Exception as exc:
            for path in reversed(written):
                path.unlink(missing_ok=True)
            if folder is not None:
                remove_empty_export_folder(folder)
            self.error_sig.emit(str(exc))
        else:
            self.finished_sig.emit(tuple(written))


class IgnoreWheel:
    def wheelEvent(self, event):
        event.ignore()


class DataComboBox(IgnoreWheel, QComboBox):
    pass


class DataSpinBox(IgnoreWheel, QDoubleSpinBox):
    pass


class DataDateEdit(IgnoreWheel, QDateTimeEdit):
    pass


class OBSAnalyzerGUIV336(QMainWindow):
    """Five-page workspace for long-record seismic bedload analysis."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"OBS Bedload Analyzer {APP_VERSION}")
        self.setWindowIcon(create_scientific_icon("waveform"))
        self.resize(1400, 900)
        self.is_dark_theme = False
        self.english = False
        self.translations = []
        self.config = load_default_config()
        self.records = []
        self.skipped_waveform_paths = set()
        self.waveform_issue_dialog = None
        self.waveform_preview = []
        self.waveform_cache = {}
        self.psd_cache = HourlyPSDCache()
        self.imported_psd = None
        self.last_psd_export = None
        self.hydro_input = None
        self.hydrology = None
        self.grain_distribution = None
        self.df_psd = None
        self.psd_mesh = None
        self.df_result = None
        self.result_scales = None
        self.result_cache = None
        self.output_path = None
        self.worker = None
        self.task_active = False
        self.tab_labels = []
        self.directory = ""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.plot_psd_canvas)
        self.init_ui()
        self.update_hydrology()
        self.update_grains()
        self.update_model()
        self.apply_theme()

    def tr_pair(self, zh, en):
        return en if self.english else zh

    def bind_text(self, setter, zh, en):
        self.translations.append((setter, zh, en))
        setter(self.tr_pair(zh, en))

    def label(self, zh, en):
        widget = QLabel()
        widget.setWordWrap(True)
        self.bind_text(widget.setText, zh, en)
        return widget

    def button(self, zh, en, callback, icon=None):
        widget = QPushButton()
        self.bind_text(widget.setText, zh, en)
        if icon:
            widget.setIcon(create_scientific_icon(icon))
        widget.clicked.connect(callback)
        return widget

    def checkbox(self, zh, en, state, callback=None):
        widget = QCheckBox()
        self.bind_text(widget.setText, zh, en)
        widget.setChecked(state)
        if callback:
            widget.toggled.connect(callback)
        return widget

    def combo(self, options, key=None):
        widget = DataComboBox()
        for zh, en, value in options:
            row = widget.count()
            widget.addItem("", value)
            self.bind_text(lambda text, i=row: widget.setItemText(i, text), zh, en)
        if key is not None:
            widget.setCurrentIndex(widget.findData(key))
        return widget

    @staticmethod
    def spin(value, minimum, maximum, decimals=3):
        widget = DataSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        return widget

    def group(self, zh, en, layout):
        widget = QGroupBox()
        self.bind_text(widget.setTitle, zh, en)
        widget.setLayout(layout)
        return widget

    def add_tab(self, tab, zh, en, icon):
        index = self.tabs.addTab(tab, create_scientific_icon(icon), "")
        self.tab_labels.append((zh, en))
        self.bind_text(lambda text: self.tabs.setTabText(index, text), zh, en)

    def init_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        self.setCentralWidget(central)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs)

        toolbar = QWidget()
        tools = QHBoxLayout(toolbar)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.addWidget(self.button("主题", "Theme", self.toggle_theme, "theme"))
        language = self.button("English", "中文", self.toggle_language)
        pixmap = QPixmap(28, 22)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setPen(QColor("#38a8d4"))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "文/A")
        painter.end()
        language.setIcon(QIcon(pixmap))
        tools.addWidget(language)
        self.tabs.setCornerWidget(toolbar, Qt.Corner.TopRightCorner)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.stop_button = self.button("停止", "Stop", self.stop_processing)
        self.stop_button.hide()
        self.statusBar().addPermanentWidget(self.stop_button)
        self.setup_waveforms()
        self.setup_hydrology()
        self.setup_psd()
        self.setup_model()
        self.setup_results()
        self.setup_menus(language.icon())
        self.tabs.currentChanged.connect(self.draw_live)

    def setup_menus(self, language_icon):
        def menu(zh, en):
            widget = self.menuBar().addMenu("")
            self.bind_text(widget.setTitle, zh, en)
            return widget

        def action(parent, zh, en, callback, shortcut=None, icon=None):
            item = QAction(self)
            self.bind_text(item.setText, zh, en)
            item.triggered.connect(callback)
            if shortcut:
                item.setShortcut(shortcut)
            if icon is not None:
                item.setIcon(icon)
            parent.addAction(item)
            return item

        files = menu("文件", "File")
        self.input_actions = [
            action(files, "输入数据", "Input data", self.open_waveforms, "Ctrl+O"),
            action(files, "导入水位与流量…", "Open stage and discharge…", self.open_hydrology),
            action(files, "导入粒径级配…", "Open grain-size distribution…", self.open_grains),
            action(files, "导入 PSD 时频数据…", "Import PSD data…", self.import_psd_data),
            action(files, "导出 PSD 时频数据…", "Export PSD data…", self.export_psd_data),
        ]
        files.addSeparator()
        self.save_action = action(files, "导出结果", "Export results", self.save_results, "Ctrl+S")
        action(files, "退出", "Exit", self.close, "Ctrl+Q")

        analysis = menu("计算", "Analysis")
        self.analysis_actions = [
            action(analysis, "预览所选波形", "Preview selected waveforms", self.start_waveform_preview),
            action(analysis, "计算 PSD 预览", "Preview PSD", self.start_psd_preview),
            action(analysis, "开始反演", "Run inversion", self.start_inversion),
            action(analysis, "标定参考 PSD…", "Calibrate reference PSD…", self.calibrate_psd),
        ]
        self.stop_action = action(analysis, "停止计算", "Stop calculation", self.stop_processing)
        self.model_choice.currentIndexChanged.connect(self.update_task_actions)

        view = menu("视图", "View")
        for index, (zh, en) in enumerate(self.tab_labels):
            action(view, zh, en, lambda checked=False, i=index: self.tabs.setCurrentIndex(i), f"Alt+{index + 1}")
        view.addSeparator()
        action(view, "主题", "Theme", self.toggle_theme, icon=create_scientific_icon("theme"))
        action(view, "English", "中文", self.toggle_language, icon=language_icon)

        help_menu = menu("帮助", "Help")
        action(help_menu, "关于", "About", self.show_about)
        self.update_task_actions()

    def show_about(self):
        QMessageBox.about(
            self, f"OBS Bedload Analyzer {APP_VERSION}",
            self.tr_pair("河流推移质地震反演。默认采用 IMHE model 2026。",
                         "Seismic inversion of river bedload. Default: IMHE model 2026."),
        )

    def update_task_actions(self):
        busy = self.task_active
        for item in self.input_actions + self.analysis_actions:
            item.setEnabled(not busy)
        self.analysis_actions[-1].setEnabled(not busy and self.model_choice.currentData() == "Luong2026")
        self.stop_action.setEnabled(busy and isinstance(self.worker, StreamingWorker))
        can_export = not busy and self.result_cache is not None
        self.save_action.setEnabled(can_export)
        self.save_button.setEnabled(can_export)
        self.result_export_format.setEnabled(not busy)
        self.invert_button.setEnabled(not busy)
        self.psd_import_button.setEnabled(not busy)
        self.psd_export_button.setEnabled(not busy)
        self.psd_compute_button.setEnabled(not busy and self.imported_psd is None)
        self.analysis_actions[1].setEnabled(not busy and self.imported_psd is None)
        for field in (self.psd_minutes, self.fft_length, self.psd_overlap):
            field.setEnabled(not busy and self.imported_psd is None)
        self.processing_threads.setEnabled(not busy)

    def split(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        panel, charts = QWidget(), QWidget()
        settings, graphics = QVBoxLayout(panel), QVBoxLayout(charts)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.addWidget(charts)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([445, 945])
        return splitter, settings, graphics

    def figure(self, layout, toolbar=True):
        panel = DataPlotPanel()
        if toolbar:
            controls = QHBoxLayout()
            controls.addWidget(self.button("重置视图", "Reset view", lambda: panel.autoRange()))
            controls.addStretch()
            layout.addLayout(controls)
        layout.addWidget(panel, 1)
        return panel

    def matplotlib_canvas(self, layout):
        canvas = FigureCanvas(Figure(figsize=(10.4, 5.8)))
        layout.addWidget(NavigationToolbar(canvas, self))
        layout.addWidget(canvas, 1)
        return canvas

    def table(self, layout):
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(table, 1)
        return table

    @staticmethod
    def fill_table(table, headings, rows):
        table.setColumnCount(len(headings))
        table.setHorizontalHeaderLabels(headings)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(value)))

    def setup_waveforms(self):
        page, left, right = self.split()
        left.addWidget(self.button("输入数据", "Input data", self.open_waveforms, "waveform"))
        self.waveform_info = QLabel()
        self.waveform_info.setWordWrap(True)
        left.addWidget(self.waveform_info)
        self.coverage_canvas = DataPlotPanel()
        self.coverage_canvas.setFixedHeight(32)
        self.coverage_canvas.setToolTip(self.tr_pair("蓝色为文件记录，灰色为缺测区间", "Blue: file coverage; grey: gaps"))
        left.addWidget(self.coverage_canvas)
        self.record_gaps = []
        self.header_table = self.table(left)
        form = QFormLayout()
        self.wave_timezone = self.combo(TIME_ZONES, "UTC")
        self.wave_timezone.currentIndexChanged.connect(self.update_wave_timezones)
        self.wave_start = self.date_editor("UTC")
        self.wave_end = self.date_editor("UTC")
        self.trace_choice = DataComboBox()
        self.trace_choice.currentIndexChanged.connect(self.trace_selected)
        form.addRow(self.label("通道", "Channel"), self.trace_choice)
        form.addRow(self.label("波形时窗时区", "Selection time zone"), self.wave_timezone)
        form.addRow(self.label("开始", "Start"), self.wave_start)
        form.addRow(self.label("结束", "End"), self.wave_end)
        self.decimation = DataComboBox()
        self.decimation.addItem("1×", 1)
        self.rate_info = QLabel()
        self.decimation.currentIndexChanged.connect(self.invalidate_spectra)
        form.addRow(self.label("分析采样率", "Analysis sample rate"), self.decimation)
        form.addRow(self.rate_info)
        self.processing_threads = DataComboBox()
        self.processing_threads.addItems([str(n) for n in PROCESSING_THREADS])
        self.processing_threads.setEditable(True)
        self.processing_threads.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.processing_threads.setValidator(QIntValidator(1, 2**31 - 1, self.processing_threads))
        self.bind_text(
            self.processing_threads.setToolTip,
            "线程预算适用于文件头扫描、波形预览、PSD 和反演。按 CPU、内存和小时块数分配，可为短时段 PSD 的 FFT 分配余量。",
            "Thread budget for file headers, waveform previews, PSD and inversion. Allocation depends on CPU, RAM and hourly blocks, with spare threads available for shorter PSD batches.",
        )
        form.addRow(self.label("并行运算", "Parallel workers"), self.processing_threads)
        left.addWidget(self.group("目标区间", "Time selection", form))
        for editor in (self.wave_start, self.wave_end):
            editor.dateTimeChanged.connect(self.invalidate_spectra)

        form = QFormLayout()
        paz = self.config["waveform"]["manual_paz"]
        self.gain = self.spin(paz["gain"], 1e-5, 1e9, 6)
        self.adc = self.spin(paz["adc_sensitivity"], 1, 1e12, 3)
        self.prefilter = QLineEdit()
        self.prefilter.setText("0.2, 0.5, 35.0, 45.0")
        self.auto_prefilter = self.checkbox("自动设置", "Auto", True, self.update_prefilter)
        self.prefilter.textEdited.connect(lambda _: self.auto_prefilter.setChecked(False))
        form.addRow(self.label("增益", "Gain"), self.gain)
        form.addRow(self.label("ADC 灵敏度", "ADC sensitivity"), self.adc)
        form.addRow(self.label("预滤波参数 (Hz)", "Prefilter parameters (Hz)"), self.prefilter)
        form.addRow(self.auto_prefilter)
        left.addWidget(self.group("仪器响应校正", "Instrument response correction", form))
        left.addWidget(self.button("预览所选波形", "Preview selection", self.start_waveform_preview))
        self.canvas_wf = self.figure(right)
        self.add_tab(page, "1 地震波形", "1 Waveforms", "waveform")
        self.update_waveform_info()

    @staticmethod
    def date_editor(zone):
        editor = DataDateEdit()
        editor.setTimeZone(QTimeZone(zone.encode("ascii")))
        editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss.zzz")
        editor.setCalendarPopup(True)
        return editor

    def setup_hydrology(self):
        page, left, right = self.split()
        form = QFormLayout()
        self.station_choice = self.combo([(k, k, k) for k in STATIONS], "PZ12")
        self.hydro_timezone = self.combo(TIME_ZONES, "Asia/Shanghai")
        self.bed_label = QLabel()
        form.addRow(self.label("站点", "Station"), self.station_choice)
        form.addRow(self.label("基准河床高程 (m)", "Reference bed elevation (m)"), self.bed_label)
        form.addRow(self.label("水文输入时区", "Hydrology time zone"), self.hydro_timezone)
        form.addRow(self.button("导入水位与流量", "Open stage and discharge", self.open_hydrology))
        form.addRow(self.label(
            "输入时间、水位高程 (m) 和流量 (m³/s)。水深按水位与基准河床高程之差计算，河宽按本站水位–河宽关系线性插值。",
            "Input columns are time, stage (m) and discharge (m³/s). Width is interpolated from the station rating curve."
        ))
        left.addWidget(self.group("水文数据", "Hydrology", form))
        self.hydro_table = self.table(left)
        self.station_choice.currentIndexChanged.connect(self.update_hydrology)
        self.hydro_timezone.currentIndexChanged.connect(self.update_hydrology)

        form = QFormLayout()
        self.grain_unit = self.combo([("米 (m)", "Metres (m)", "m"), ("毫米 (mm)", "Millimetres (mm)", "mm")], "m")
        self.grain_file_label = QLabel()
        self.grain_file_label.setWordWrap(True)
        form.addRow(self.label("级配文件粒径单位", "Grain-file unit"), self.grain_unit)
        form.addRow(self.button("导入粒径级配", "Open grain-size distribution", self.open_grains))
        form.addRow(self.grain_file_label)
        self.grain_spins = []
        self.grain_d50_label = QLabel()
        self.grain_d50 = None
        self.grain_percentiles = None
        for name, value in (("D16", 0.008), ("D50", 0.02), ("D84", 0.05)):
            control = self.spin(value, 1e-6, 20, 6)
            self.grain_spins.append(control)
            form.addRow(f"{name} (m)", control)
            control.editingFinished.connect(self.update_grains)
        form.addRow(self.button("使用 D16 / D50 / D84", "Use D16 / D50 / D84", self.use_grain_percentiles))
        form.addRow(self.label("反演中值粒径", "Inversion median grain size"), self.grain_d50_label)
        left.addWidget(self.group("泥沙粒径级配", "Grain-size distribution", form))
        self.grain_unit.currentIndexChanged.connect(self.update_grains)
        self.grain_table = self.table(left)
        self.canvas_hydro = self.matplotlib_canvas(right)
        self.add_tab(page, "2 水文资料", "2 Hydrology", "river")

    def setup_psd(self):
        page, left, right = self.split()
        form = QFormLayout()
        self.psd_minutes = self.combo([(f"{n} min", f"{n} min", n) for n in (1, 5, 30)], 1)
        self.fft_length = self.combo([(str(n), str(n), n) for n in (16384, 8192, 4096)], 16384)
        self.psd_overlap = self.combo([(f"{n}%", f"{n}%", n / 100) for n in (5, 25, 50)], 0.5)
        form.addRow(self.label("功率谱密度汇总时长", "PSD interval"), self.psd_minutes)
        form.addRow(self.label("Hann 窗长（采样点）", "Hann window samples"), self.fft_length)
        form.addRow(self.label("相邻谱窗重叠率", "Spectral window overlap"), self.psd_overlap)
        form.addRow(self.label("按整分钟 UTC 对齐。", "Aligned to UTC minute boundaries."))
        self.psd_compute_button = self.button("计算 PSD 预览", "Preview PSD", self.start_psd_preview)
        self.psd_import_button = self.button("导入 PSD 时频数据…", "Import PSD data…", self.import_psd_data)
        self.psd_export_button = self.button("导出 PSD 时频数据…", "Export PSD data…", self.export_psd_data)
        self.psd_source_label = QLabel()
        self.psd_source_label.setWordWrap(True)
        for widget in (self.psd_compute_button, self.psd_import_button, self.psd_export_button, self.psd_source_label):
            form.addRow(widget)
        self.refresh_psd_source()
        left.addWidget(self.group("功率谱密度（PSD）", "Power spectral density (PSD)", form))
        self.psd_minutes.currentIndexChanged.connect(self.invalidate_spectra)
        self.fft_length.currentIndexChanged.connect(self.invalidate_spectra)
        self.psd_overlap.currentIndexChanged.connect(self.invalidate_spectra)

        form = QFormLayout()
        self.psd_fmin = self.spin(0.5, 0, 10000, 3)
        self.psd_fmax = self.spin(35, 0.001, 10000, 3)
        self.frequency_auto = self.checkbox(
            "频率范围跟随预滤波通带 f₂–f₃", "Use response-prefilter passband f₂–f₃",
            True, self.schedule_psd_plot,
        )
        self.colormap = self.combo([(name, name, name) for name in ("viridis", "cividis", "magma", "turbo")])
        self.db_auto = self.checkbox("自动色阶", "Automatic colour limits", True, self.schedule_psd_plot)
        self.db_min = self.spin(-185, -400, 200, 1)
        self.db_max = self.spin(-100, -400, 200, 1)
        self.show_depth = self.checkbox("显示水深", "Show river depth", True, self.schedule_psd_plot)
        self.depth_auto = self.checkbox("自动水深范围", "Automatic depth limits", True, self.schedule_psd_plot)
        self.depth_min = self.spin(0, 0, 10000, 3)
        self.depth_max = self.spin(20, 0.001, 10000, 3)
        self.time_auto = self.checkbox("显示全时段", "Show full time range", True, self.schedule_psd_plot)
        self.psd_start = self.date_editor("Asia/Shanghai")
        self.psd_end = self.date_editor("Asia/Shanghai")
        for zh, en, widget in (
            ("频率下限 (Hz)", "Minimum frequency (Hz)", self.psd_fmin),
            ("频率上限 (Hz)", "Maximum frequency (Hz)", self.psd_fmax),
            ("配色", "Colormap", self.colormap),
            ("色阶下限 (dB)", "Lower colour limit (dB)", self.db_min),
            ("色阶上限 (dB)", "Upper colour limit (dB)", self.db_max),
            ("水深下限 (m)", "Minimum depth (m)", self.depth_min),
            ("水深上限 (m)", "Maximum depth (m)", self.depth_max),
            ("显示开始", "Display start", self.psd_start),
            ("显示结束", "Display end", self.psd_end),
        ):
            form.addRow(self.label(zh, en), widget)
        for widget in (self.frequency_auto, self.db_auto, self.show_depth, self.depth_auto, self.time_auto):
            form.addRow(widget)
        for widget in (self.psd_fmin, self.psd_fmax, self.depth_min, self.depth_max):
            widget.valueChanged.connect(self.schedule_psd_plot)
        for widget in (self.db_min, self.db_max):
            widget.valueChanged.connect(self.update_psd_colors)
        self.colormap.currentIndexChanged.connect(self.update_psd_colors)
        self.prefilter.textChanged.connect(self.schedule_psd_plot)
        for editor in (self.psd_start, self.psd_end):
            editor.dateTimeChanged.connect(self.schedule_psd_plot)
        left.addWidget(self.group("显示", "Display", form))
        left.addStretch()
        self.canvas_psd = self.matplotlib_canvas(right)
        self.add_tab(page, "3 时频分析", "3 Time–frequency analysis", "psd")
        self.sync_psd_auto_fields()

    def setup_model(self):
        page, left, right = self.split()
        self.model_choice = self.combo([
            ("IMHE model 2026（用户自定义）", "IMHE model 2026 (user-defined)", "IMHE2026"),
            ("Tsai 等（2012）单一跃移物理模型", "Tsai et al. (2012) · Saltation", "Tsai2012"),
            ("Luong 等（2024）多输移模式物理模型", "Luong et al. (2024) · Multiple modes", "Luong2024"),
            ("Luong 等（2026）地震–水力经验模型", "Luong et al. (2026) · Seismic–hydraulic", "Luong2026"),
        ], self.config["inversion"]["model_type"])
        left.addWidget(self.model_choice)
        form = QFormLayout()
        self.model_widgets = {}

        def add(form, key, zh, en, value, lo, hi, places=5):
            control = self.spin(value, lo, hi, places)
            form.addRow(self.label(zh, en), control)
            self.model_widgets[key] = control

        for spec in (
            ("fmin", "反演频带下限 (Hz)", "Lower inversion frequency (Hz)", 20, 0.001, 10000),
            ("fmax", "反演频带上限 (Hz)", "Upper inversion frequency (Hz)", 150, 0.002, 10000),
            ("slope", "坡降 S (无量纲)", "Slope S (dimensionless)", 0.00698143, 1e-8, 0.999, 8),
            ("rho_s", "颗粒密度 (kg/m³)", "Sediment density (kg/m³)", 2700, 1001, 6000),
            ("rho_f", "流体密度 (kg/m³)", "Fluid density (kg/m³)", 1000, 100, 2000),
        ):
            add(form, *spec)
        self.band_statistic = self.combo([("中位数", "Median", "median"), ("均值", "Mean", "mean")])
        form.addRow(self.label("频带内线性谱统计量", "Linear PSD band statistic"), self.band_statistic)
        form.addRow(self.button("将反演频带调整至有效通带", "Fit inversion band to passband", self.fit_inversion_band))
        left.addWidget(self.group("通用反演参数", "Shared settings", form))

        form = QFormLayout()
        for spec in (
            ("r0", "震源距离 r₀ (m)", "Source distance r₀ (m)", self.config["site"]["source_distance_m"], 0.01, 100000),
            ("tau_c50", "中值粒径临界 Shields 参数 τc,50*", "Critical Shields τc,50*", 0.045, 0.001, 1),
            ("ks", "粗糙度 kₛ (m；0=自动)", "Roughness kₛ (m; 0=auto)", 0, 0, 100),
            ("vc0_override", "参考相速度 Vc₀ (m/s；0=自动)", "Reference phase velocity Vc₀ (m/s; 0=auto)", 0, 0, 20000),
            ("zeta_override", "频散指数 ζ (-1=自动)", "Dispersion exponent ζ (-1=auto)", -1, -1, 10),
            ("Q0", "品质因子 Q₀", "Quality factor Q₀", 20, 0.01, 10000),
            ("eta", "品质因子频率指数 η", "Quality-factor exponent η", 0, -5, 5),
        ):
            add(form, *spec)
        self.propagation_panel = self.group("物理模型参数", "Physical model settings", form)
        left.addWidget(self.propagation_panel)

        self.model_specific = QStackedWidget()
        tsai = QWidget()
        form = QFormLayout(tsai)
        self.tsai_stage = self.combo([("OBS 超额输移 T−1", "OBS excess stage T−1", "excess"),
                                      ("Tsai 原式 T", "Published stage T", "published")])
        self.clip_tsai = self.checkbox("截断临界 Shields 至 0.03–0.06", "Clip critical Shields to 0.03–0.06", False)
        form.addRow(self.label("输移阶段", "Transport stage"), self.tsai_stage)
        form.addRow(self.clip_tsai)
        self.model_specific.addWidget(tsai)
        multi = QWidget()
        form = QFormLayout(multi)
        for spec in (
            ("restitution", "恢复系数 γ", "Restitution γ", 0, 0, 1),
            ("fz", "法向速度比 fz", "Normal velocity ratio fz", 0.539, 0, 10),
            ("Nzz", "垂向耦合 Nzz", "Vertical coupling Nzz", 0.352, 0, 10),
            ("hop_min", "冲击间隔下限 (s)", "Minimum impact interval (s)", 0.01, 1e-5, 100),
            ("hop_max", "冲击间隔上限 (s)", "Maximum impact interval (s)", 0.5, 1e-5, 100),
            ("hop_median", "冲击间隔中位数 (s)", "Median impact interval (s)", 0.08, 1e-5, 100),
            ("hop_sigma", "冲击间隔对数标准差", "Log-space impact interval SD", 1, 0.001, 10),
        ):
            add(form, *spec)
        self.clip_multi = self.checkbox("截断临界 Shields 至 0.03–0.06", "Clip critical Shields to 0.03–0.06", True)
        form.addRow(self.clip_multi)
        self.model_specific.addWidget(multi)
        empirical = QWidget()
        form = QFormLayout(empirical)
        for spec in (
            ("k", "尺度系数 k", "Scale factor k", 5.2, 1e-5, 1e8),
            ("m", "PSD 指数 m", "PSD exponent m", 0.25, 1e-5, 20),
            ("n", "超额剪切应力指数 n", "Excess-shear exponent n", 1.36, 1e-5, 20),
            ("tau_cr", "起动临界 Shields 参数 τcr*", "Threshold Shields τcr*", 0.045, 0, 1),
            ("A_db", "参考 PSD 斜率 A (dB)", "Reference PSD slope A (dB)", 29, -1000, 1000),
            ("B_db", "参考 PSD 截距 B (dB)", "Reference PSD intercept B (dB)", -127, -1000, 1000),
            ("hydraulic_radius_depth_ratio", "水力半径与水深比 Rh/H", "Hydraulic radius to depth ratio Rh/H", 1, 0.001, 10),
        ):
            add(form, *spec)
        form.addRow(self.button("标定参考 PSD", "Calibrate reference PSD", self.calibrate_psd))
        self.model_specific.addWidget(empirical)
        left.addWidget(self.model_specific)
        left.addStretch()
        panel = QWidget()
        mechanism = QVBoxLayout(panel)
        mechanism.setSpacing(10)
        title = self.label("物理机制及关键公式", "Mechanism and equations")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        mechanism.addWidget(title)
        self.mechanism_text = QLabel()
        self.mechanism_text.setWordWrap(True)
        self.mechanism_text.setTextFormat(Qt.TextFormat.RichText)
        self.formula_canvas = FormulaCanvas(self)
        self.formula_canvas.figure.set_layout_engine(None)
        symbol_title = self.label("参数含义与单位", "Symbols and units")
        symbol_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.symbol_text = QLabel()
        self.symbol_text.setWordWrap(True)
        self.symbol_text.setTextFormat(Qt.TextFormat.RichText)
        mechanism.addWidget(self.mechanism_text)
        mechanism.addWidget(self.formula_canvas)
        mechanism.addWidget(symbol_title)
        mechanism.addWidget(self.symbol_text)
        mechanism.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        right.addWidget(scroll)
        self.model_choice.currentIndexChanged.connect(self.update_model)
        self.add_tab(page, "4 反演模型", "4 Inversion model", "settings")

    def setup_results(self):
        page, left, right = self.split()
        self.invert_button = self.button("开始反演", "Run inversion", self.start_inversion, "inversion")
        self.save_button = self.button("导出结果", "Export results", self.save_results)
        left.addWidget(self.invert_button)
        self.result_export_format = self.combo(
            [(caption, caption, suffix) for caption, suffix in EXPORT_FORMATS.items()], ".csv",
        )
        export_form = QFormLayout()
        export_form.addRow(self.label("导出数据格式", "Export data format"), self.result_export_format)
        left.addLayout(export_form)
        left.addWidget(self.save_button)
        left.addWidget(self.label(
            "显示小时和日尺度结果，导出时同时包含自然月数据。各时段按北京时间划分，输沙率按有效时长加权，累计量截至该时段末。",
            "Charts show hourly and daily results. Exports include calendar months. Intervals use Beijing time, and rates are weighted by valid duration."
        ))
        self.result_info = QLabel()
        self.result_info.setWordWrap(True)
        left.addWidget(self.result_info)
        self.result_table = self.table(left)
        self.canvas_result = self.matplotlib_canvas(right)
        self.add_tab(page, "5 反演结果", "5 Inversion results", "inversion")

    def apply_theme(self):
        self.setStyleSheet(DARK_THEME_QSS if self.is_dark_theme else LIGHT_THEME_QSS)
        for panel in (self.canvas_wf, self.coverage_canvas):
            panel.theme(self.is_dark_theme)
        for canvas in (self.canvas_psd, self.canvas_result):
            canvas.figure.set_facecolor(self.plot_background)
            canvas.draw_idle()
        self.formula_canvas.set_theme(self.is_dark_theme)
        self.refresh_wave_style()
        self.plot_hydrology()
        if self.psd_mesh is not None:
            self.refresh_psd_style()
        else:
            self.plot_psd_canvas()
        self.refresh_result_style()

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def toggle_language(self):
        self.english = not self.english
        for setter, zh, en in self.translations:
            setter(self.tr_pair(zh, en))
        self.update_model()
        self.update_waveform_info()
        self.refresh_hydro_table()
        self.refresh_grain_table()
        self.refresh_result_table()
        self.refresh_psd_source()
        self.apply_theme()

    def update_model(self):
        key = self.model_choice.currentData()
        self.model_specific.setCurrentIndex(1 if key in MULTIMODE_TYPES else MODEL_TYPES.index(key))
        self.propagation_panel.setVisible(key != "Luong2026")
        self.formula_canvas.set_model(key)
        text = {
            "IMHE2026": (
                "本程序基于 Luong 等（2024）的多输移模式物理模型修订。模型保留粒径和冲击间隔分布，并沿用瑞利波传播项。先对各粒级计算实际 Shields 参数 τ*，再与粒径相关的临界值 τc*(D) 比较。未达到起动条件的粒级，其顺流输移速度 Ub 置零；该约束也适用于浅水工况下的滚动速度关系。\n\n"
                "在冲击率与冲量平方的乘积中，速率项的 1/Ub 与冲量平方中的 Ub² 可约简。化简后，源项随颗粒速度和单宽体积通量线性变化，Ub 为零时可直接求得零源项。对各粒级积分得到单位通量理论谱，再与同频带观测谱比较。该修订属于本程序的实现方案。",
                "This implementation revises the Luong et al. (2024) multi-mode physical model. It retains the grain-size and impact-interval distributions and Rayleigh-wave propagation. It compares the applied Shields parameter with the size-dependent threshold. Grain speeds are set to zero below entrainment, including in the shallow-water rolling branch.\n\n"
                "The rate term contains 1/Ub and the squared impulse contains Ub². Cancelling these terms leaves a source proportional to grain speed and volumetric flux per unit width. The source evaluates to zero when Ub is zero. Integrating over grain size yields the unit-flux spectrum for comparison with observations in the inversion band.",
            ),
            "Tsai2012": (
                "水流使床沙跃起。颗粒回落时近乎垂直撞击河床，在弹性近似下，单次冲量约为 2mwi，其中 m 是颗粒质量，wi 是法向撞击速度。对同一粒级，水力条件决定颗粒迁移速度、跃移层厚度和撞击率。\n\n"
                "随机冲击激发瑞利波。台站 PSD 随撞击率和单次冲量平方增加，也受传播距离、频散和衰减影响。对粒径分布积分后，用同频带的观测 PSD 与单位通量理论 PSD 比较来估计输沙率。较粗颗粒的谱贡献往往较大，级配不确定性会直接传入反演。",
                "Flow entrains grains into saltation. Grains strike the bed nearly normally on descent. The elastic approximation gives an impulse of roughly 2mwi, where m is grain mass and wi is normal impact speed. For each size class, hydraulics determine grain speed, transport-layer thickness and impact rate.\n\n"
                "Random impacts excite Rayleigh waves. Receiver PSD increases with impact rate and squared impulse and is shaped by travel distance, dispersion and attenuation. Integrate over grain size, then compare observed and unit-flux PSD in the same band to estimate transport. Coarse grains can contribute disproportionately, so uncertainty in grain size carries into the inversion.",
            ),
            "Luong2024": (
                "相邻冲击的间隔 tD 表征不同输移行为。较短间隔可对应滚动或滑动中的接触，也可能来自反弹；较长间隔可对应跃移。颗粒在两次冲击之间平均移动 Ub·tD，因此在通量相同的情况下，撞击率随间距增大而降低。\n\n"
                "顺流速度 Ub、斜向冲击几何和恢复系数共同决定冲量。模型沿用瑞利波传播项，对粒径和冲击间隔双重积分，再用观测 PSD 除以单位通量理论谱。冲击间隔分布和颗粒速度是反演的主要不确定来源。",
                "The interval tD between impacts represents different transport behaviour. Short intervals can reflect rolling or sliding contacts as well as rebounds; longer intervals can reflect saltation. Grains travel roughly Ub·tD between impacts. At fixed flux, longer spacing means fewer impacts.\n\n"
                "Streamwise grain speed Ub, impact geometry and restitution determine the impulse. The model retains the Rayleigh-wave propagation term, integrates over grain sizes and impact intervals, then estimates flux from observed PSD divided by the unit-flux prediction. The interval distribution and grain speed are key uncertainties.",
            ),
            "Luong2026": (
                "先以水深或水力半径、坡降和 D50 估计无量纲 Shields 应力 τ*。超过起动阈值的部分控制基准输沙率。用现场 PSD 与剪切应力的关系，估计同等剪切应力下的参考功率 Pre。\n\n"
                "用线性功率比 P/Pre 修正基准通量。功率比大于 1 时，正的 m 会提高通量估计。若 PSD 用 dB 表示，先求观测与参考的 dB 差，再转换为线性比。模型将颗粒撞击和传播的细节归入经验关系，跨场地预测前应核对标定系数。",
                "Depth or hydraulic radius, slope and D50 give dimensionless Shields stress τ*. Stress above the entrainment threshold sets baseline transport. A site fit relates PSD to shear stress and supplies reference power Pre at the same stress.\n\n"
                "The linear power ratio P/Pre adjusts the baseline. A ratio above one increases estimated flux when m is positive. With PSD in dB, convert the difference from reference into a linear ratio. The empirical fit absorbs impact and propagation effects, so check calibration before transferring it to another site.",
            ),
        }[key]
        paragraphs = self.tr_pair(*text).split("\n\n")
        self.mechanism_text.setText("".join(
            f'<p style="font-size: 13px; line-height: 150%; margin: 0 0 10px;">{escape(p)}</p>'
            for p in paragraphs
        ))
        self.symbol_text.setText("".join(
            f'<p style="font-size: 13px; line-height: 140%; margin: 0 0 5px;">'
            f'<b>{escape(symbol)}</b>  {escape(self.tr_pair(zh, en))}</p>'
            for symbol, zh, en in FORMULA_SYMBOLS[key]
        ))

    def warn(self, message):
        QMessageBox.warning(self, self.tr_pair("检查设置", "Check settings"), str(message))

    def select_input(self, zh, en, extension, multiple=False):
        dialog = QFileDialog.getOpenFileNames if multiple else QFileDialog.getOpenFileName
        paths, _ = dialog(self, self.tr_pair(zh, en), self.directory, extension)
        if paths:
            self.directory = str(Path(paths[0] if multiple else paths).parent)
        return paths

    def open_waveforms(self):
        paths = self.select_input(
            "输入数据", "Input data",
            ("Waveforms (*.bhz *.BHZ *.bhn *.BHN *.bhe *.BHE *.hyd *.HYD "
             "*.mseed *.MSEED *.miniseed *.MINISEED *.msd *.MSD *.sac *.SAC);;All files (*)"),
            multiple=True,
        )
        if paths:
            try:
                workers = self.requested_workers()
            except ValueError as exc:
                self.warn(exc)
                return
            self.start_worker(HeaderScanWorker(paths, workers, allow_partial=True), self.headers_loaded)

    def headers_loaded(self, report):
        if not isinstance(report, WaveformScanReport):
            report = WaveformScanReport(list(report), [])
        if not report.records and not report.issues:
            self.warn(self.tr_pair("所选文件中没有波形。", "No waveforms found in the selection."))
            return
        records = report.records
        trace_id = self.default_trace_id(records) if records else None
        channel = [r for r in records if r.trace_id == trace_id]
        if channel:
            start, end = self.default_waveform_interval(channel)
            # Keep the same tolerance for selecting traces as the processing path.
            selected = records_overlapping_window(channel, start, end)
        else:
            selected = []
        if report.issues or len({r.sampling_rate_hz for r in selected}) > 1:
            self.show_waveform_issues(
                selected, report.issues, lambda excluded: self.commit_waveform_batch(report, excluded),
            )
        else:
            self.commit_waveform_batch(report, set())

    @staticmethod
    def default_trace_id(records):
        return next((r.trace_id for r in records if r.channel.upper() == "BHZ"),
                    next((r.trace_id for r in records if r.channel.upper().endswith("Z")), records[0].trace_id))

    def default_waveform_interval(self, records):
        start = min(r.start_utc for r in records)
        end = max(r.end_utc + pd.Timedelta(seconds=1 / r.sampling_rate_hz) for r in records)
        return start, min(end, start + pd.Timedelta(hours=self.config["waveform"]["default_selection_hours"]))

    def commit_waveform_batch(self, report, excluded):
        retained = [r for r in report.records if r.path not in excluded]
        if not retained:
            return
        if self.imported_psd is not None:
            self.imported_psd = None
            self.last_sample_rate = None
            self.refresh_psd_source()
        self.records = report.records + [header for issue in report.issues for header in issue.headers]
        self.skipped_waveform_paths = set(excluded)
        self.waveform_preview = []
        self.waveform_cache.clear()
        self.canvas_wf.reset()
        with QSignalBlocker(self.trace_choice):
            self.trace_choice.clear()
            channels = {r.trace_id: r.channel for r in retained}
            for trace_id, component in sorted(channels.items()):
                description = self.tr_pair("水听器", "Hydrophone") if component.upper() == "HYD" else component
                self.trace_choice.addItem(f"{description} · {trace_id}", trace_id)
            self.trace_choice.setCurrentIndex(self.trace_choice.findData(self.default_trace_id(retained)))
        self.trace_selected()

    def channel_records(self):
        return [r for r in self.records if r.trace_id == self.trace_choice.currentData()]

    def active_records(self):
        return [r for r in self.channel_records() if r.path not in self.skipped_waveform_paths]

    def update_waveform_info(self):
        records = self.active_records()
        zone = self.wave_timezone.currentData()
        rates = ", ".join(f"{fs:g}" for fs in sorted({record.sampling_rate_hz for record in records}))
        if records:
            start = min(r.start_utc for r in self.records).tz_convert(zone)
            end = max(r.end_utc for r in self.records).tz_convert(zone)
            self.waveform_info.setText(
                f"{len(records):,} {self.tr_pair('条文件记录', 'file records')} · {rates} Hz\n"
                f"{self.tr_pair('输入总时间范围', 'Full input time range')}\n"
                f"{start.isoformat(sep=' ')} – {end.isoformat(sep=' ')}"
            )
        else:
            self.waveform_info.setText(self.tr_pair("尚未输入数据。", "No waveforms loaded."))
        locations = sorted({str(Path(r.path).parent) for r in self.records})
        if locations:
            label = self.tr_pair("波形文件位置", "Waveform location")
            shown = "\n".join(locations[:2])
            if len(locations) > 2:
                shown += self.tr_pair(f"\n另有 {len(locations) - 2} 个目录", f"\n{len(locations) - 2} more folders")
            self.waveform_info.setText(self.waveform_info.text() + f"\n{label}\n{shown}")
        if self.skipped_waveform_paths:
            count = len(self.skipped_waveform_paths)
            self.waveform_info.setText(self.waveform_info.text() + "\n" + self.tr_pair(
                f"已跳过 {count:,} 个有问题的文件", f"Skipped {count:,} files with issues",
            ))
        self.waveform_info.setToolTip("\n".join(locations))
        rows = [
            (Path(r.path).name, r.trace_id, r.start_utc.tz_convert(zone).isoformat(),
             r.end_utc.tz_convert(zone).isoformat(), f"{r.sampling_rate_hz:g}", str(Path(r.path).parent))
            for r in records[:250]
        ]
        self.fill_table(
            self.header_table,
            [self.tr_pair("文件", "File"), "Trace ID", self.tr_pair("开始", "Start"),
             self.tr_pair("结束", "End"), "Hz", self.tr_pair("所在目录", "Folder")],
            rows,
        )

    def update_waveform_coverage(self):
        original = self.channel_records()
        records = self.active_records()
        spans = sorted((r.start_utc, r.end_utc + pd.Timedelta(seconds=1/r.sampling_rate_hz)) for r in records)
        merged = []
        for begin, end in spans:
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((begin, end))
        self.record_gaps = []
        self.coverage_canvas.reset()
        if original:
            start = min(r.start_utc for r in original)
            end = max(r.end_utc + pd.Timedelta(seconds=1/r.sampling_rate_hz) for r in original)
            cursor = start
            for left, right in merged:
                if left > cursor:
                    self.record_gaps.append((cursor, left))
                cursor = right
            if cursor < end:
                self.record_gaps.append((cursor, end))
            plot = self.coverage_canvas.add_chart(0)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setMouseEnabled(x=False, y=False)
            plot.setMenuEnabled(False)
            x0, x1 = epoch_seconds([start, end])
            plot.addItem(pg.BarGraphItem(x0=x0, x1=x1, y0=0, height=1, pen=None, brush="#b6bdc4"))
            if merged:
                bounds = epoch_seconds([value for pair in merged for value in pair]).reshape(-1, 2)
                plot.addItem(pg.BarGraphItem(x0=bounds[:, 0], x1=bounds[:, 1], y0=0, height=1,
                                            pen=None, brush="#3682ab"))
            plot.setRange(xRange=(x0, x1), yRange=(0, 1), padding=0)
        self.update_waveform_info()

    def trace_selected(self):
        self.waveform_preview = []
        self.waveform_cache.clear()
        self.canvas_wf.reset()
        self.update_waveform_coverage()
        records = self.active_records()
        if not records:
            return
        original = self.channel_records()
        start, default_end = self.default_waveform_interval(original)
        end = max(r.end_utc + pd.Timedelta(seconds=1 / r.sampling_rate_hz) for r in original)
        zone = self.wave_start.timeZone()
        qstart = QDateTime.fromMSecsSinceEpoch(start.value // 10**6, zone)
        qend = QDateTime.fromMSecsSinceEpoch((end.value + 999999) // 10**6, zone)
        with QSignalBlocker(self.wave_start), QSignalBlocker(self.wave_end):
            for editor in (self.wave_start, self.wave_end):
                editor.setDateTimeRange(qstart, qend)
            self.wave_start.setDateTime(qstart)
            self.wave_end.setDateTime(QDateTime.fromMSecsSinceEpoch((default_end.value + 999999) // 10**6, zone))
        self.invalidate_spectra()

    def update_wave_timezones(self):
        zone = QTimeZone(self.wave_timezone.currentData().encode("ascii"))
        for editor in (self.wave_start, self.wave_end):
            epoch = editor.dateTime().toMSecsSinceEpoch()
            editor.blockSignals(True)
            editor.setTimeZone(zone)
            editor.setDateTime(QDateTime.fromMSecsSinceEpoch(epoch, zone))
            editor.blockSignals(False)
        self.update_waveform_info()
        self.refresh_wave_style()

    def waveform_interval(self):
        return self.editor_utc(self.wave_start), self.editor_utc(self.wave_end)

    def selected_waveform_records(self):
        start, end = self.waveform_interval()
        return records_overlapping_window(self.active_records(), start, end) if end > start else []

    def prepare_waveforms(self, resume):
        if self.waveform_issue_dialog is not None:
            return False
        selected = self.selected_waveform_records()
        if len({r.sampling_rate_hz for r in selected}) < 2:
            return True
        self.show_waveform_issues(selected, (), lambda excluded: self.resume_waveform_task(excluded, resume))
        return False

    def resume_waveform_task(self, excluded, resume):
        self.skipped_waveform_paths.update(excluded)
        self.waveform_preview = []
        self.waveform_cache.clear()
        self.canvas_wf.reset()
        self.update_waveform_coverage()
        self.invalidate_spectra()
        if resume is not None:
            resume()

    def show_waveform_issues(self, selected, read_issues, on_accept):
        """Show one non-blocking confirmation for unreadable and conflicting files."""
        by_path = {}
        by_rate = {}
        unreadable = {issue.path: issue for issue in read_issues}
        for record in selected:
            by_path.setdefault(record.path, []).append(record)
            if record.path not in unreadable:
                by_rate.setdefault(record.sampling_rate_hz, set()).add(record.path)

        zone = self.wave_timezone.currentData()
        dialog = QDialog(self)
        self.waveform_issue_dialog = dialog
        dialog.setWindowTitle(self.tr_pair("检查波形文件", "Check waveform files"))
        dialog.resize(1000, 420)
        layout = QVBoxLayout(dialog)
        notice = QLabel(self.tr_pair(
            "检查到文件读取错误或采样率冲突。确认后，跳过下表中的整个文件，其余数据继续处理。",
            "Some files failed to read or have conflicting sample rates. Confirm to skip the listed files and process the remaining data.",
        ))
        notice.setWordWrap(True)
        layout.addWidget(notice)
        rates = QComboBox()
        for rate, paths in sorted(by_rate.items()):
            count = self.tr_pair(f"{len(paths):,} 个文件", f"{len(paths):,} files")
            rates.addItem(f"{rate:.16g} Hz · {count}", rate)
        if by_rate:
            default = (500.0 if 500.0 in by_rate and 501.0 in by_rate else
                       max(sorted(by_rate), key=lambda rate: len(by_rate[rate])))
            rates.setCurrentIndex(rates.findData(default))
            form = QFormLayout()
            form.addRow(self.tr_pair("保留采样率", "Keep sample rate"), rates)
            layout.addLayout(form)
        summary = QLabel()
        summary.setWordWrap(True)
        layout.addWidget(summary)
        report = self.table(layout)
        report.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        report.horizontalHeader().setStretchLastSection(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        proceed = buttons.addButton(self.tr_pair("跳过并继续", "Skip and continue"),
                                    QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr_pair("取消", "Cancel"))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        excluded = set()

        def refresh():
            target = rates.currentData()
            excluded.clear()
            excluded.update(unreadable)
            for path, records in by_path.items():
                if any(r.sampling_rate_hz != target for r in records):
                    excluded.add(path)
            rows = []
            for path in sorted(excluded):
                issue = unreadable.get(path)
                headers = by_path.get(path) or (issue.headers if issue else ())
                frequencies = ", ".join(f"{rate:.16g}" for rate in sorted({r.sampling_rate_hz for r in headers}))
                start = min((r.start_utc for r in headers), default=None)
                stamp = start.tz_convert(zone).isoformat(sep=" ") if start is not None else ""
                reason = (" · ".join(issue.reason.splitlines()) if issue else
                          self.tr_pair("采样率冲突", "Conflicting sample rate"))
                rows.append((Path(path).name, frequencies, stamp, str(Path(path).parent), reason))
            report.setUpdatesEnabled(False)
            try:
                self.fill_table(report, [
                    self.tr_pair("将跳过的文件", "Files to skip"), "Hz", self.tr_pair("开始", "Start"),
                    self.tr_pair("所在目录", "Folder"), self.tr_pair("原因", "Reason"),
                ], rows)
                for i, row in enumerate(rows):
                    report.item(i, 3).setToolTip(str(Path(row[3]) / row[0]))
                    report.item(i, 4).setToolTip(row[4])
            finally:
                report.setUpdatesEnabled(True)
            kept = len(by_rate.get(target, set()) - excluded)
            proceed.setEnabled(kept > 0)
            summary.setText(self.tr_pair(
                f"跳过 {len(excluded):,} 个文件 · 保留 {kept:,} 个文件 · 时间显示时区 {zone}",
                f"Skip {len(excluded):,} files · Keep {kept:,} files · Times in {zone}",
            ) if kept else self.tr_pair("所选时段没有可用文件。", "No usable files in the selection."))

        def finished(result):
            self.waveform_issue_dialog = None
            accepted = result == QDialog.DialogCode.Accepted and proceed.isEnabled()
            paths = set(excluded)
            dialog.deleteLater()
            if accepted:
                QTimer.singleShot(0, lambda: on_accept(paths))

        rates.currentIndexChanged.connect(refresh)
        dialog.finished.connect(finished)
        refresh()
        for i, width in enumerate((205, 65, 212, 235)):
            report.setColumnWidth(i, width)
        dialog.open()

    def source(self, *, attach_cache=True):
        records = self.active_records()
        if not records:
            raise ValueError(self.tr_pair("请先输入数据。", "Input waveform data first."))
        native = StreamingWaveform(records, *self.waveform_interval())
        if attach_cache:
            native.mseed_cache = MiniSeedReadCache(self.records)
        factor = self.decimation.currentData() or 1
        return native if factor == 1 else DecimatedWaveform(native, factor)

    @staticmethod
    def editor_utc(editor):
        return pd.Timestamp(editor.dateTime().toMSecsSinceEpoch(), unit="ms", tz="UTC")

    def invalidate_spectra(self):
        if getattr(self, "psd_fmax", None) is None:
            return
        if self.imported_psd is not None:
            self.update_prefilter()
            return
        dirty = self.df_psd is not None
        self.df_psd = None
        if dirty:
            self.psd_mesh = None
            self.canvas_psd.figure.clear()
            self.canvas_psd.draw_idle()
        self.update_prefilter()
        self.sync_psd_auto_fields()

    def update_prefilter(self):
        if not getattr(self, "records", None):
            return
        try:
            source = self.source(attach_cache=False)
            native_fs = getattr(source, "native", source).fs
            if getattr(self, "input_rate", None) != native_fs:
                self.input_rate = native_fs
                with QSignalBlocker(self.decimation):
                    self.decimation.clear()
                    for factor in (1, 2, 4, 8, 16):
                        self.decimation.addItem(f"{native_fs / factor:g} Hz (÷{factor})", factor)
                source = self.source(attach_cache=False)
            fs = source.fs
        except ValueError:
            return
        if self.auto_prefilter.isChecked():
            self.prefilter.setText(", ".join(
                f"{value:.1f}" if value >= 1 and value.is_integer() else f"{value:g}"
                for value in default_response_prefilter(fs)
            ))
        corners = default_response_prefilter(fs)
        try:
            corners = self.prefilter_corners(fs / 2)
        except ValueError:
            pass
        self.rate_info.setText(self.tr_pair("有效通带", "Passband") +
                               f"  {corners[1]:g}–{min(corners[2], .4*fs if fs < native_fs else fs/2):g} Hz")
        if getattr(self, "last_sample_rate", None) != fs or getattr(self, "fft_native", None) != native_fs:
            self.last_sample_rate = fs
            if self.imported_psd is None:
                self.set_psd_frequency_cap(fs / 2)
            factor = self.decimation.currentData()
            old_factor = getattr(self, "fft_factor", 1)
            native_window = next((n for n in (16384, 8192, 4096) if n / (2 * native_fs) <= 21), 4096)
            if getattr(self, "fft_native", None) == native_fs:
                native_window = self.fft_length.currentData() * old_factor
            self.fft_native, self.fft_factor = native_fs, factor
            window = max(2, round(native_window / factor))
            if self.fft_length.findData(window) < 0:
                self.fft_length.addItem(str(window), window)
            self.fft_length.setCurrentIndex(self.fft_length.findData(window))
        self.sync_psd_auto_fields()

    def open_hydrology(self):
        path = self.select_input("选择水文时间序列", "Select hydrology series", "Tables (*.csv *.tsv *.txt *.xlsx *.xls)")
        if not path:
            return
        try:
            frame = read_tabular_file(path)
            parse_hydrology(frame, self.hydro_timezone.currentData())
        except Exception as exc:
            self.warn(exc)
            return
        self.hydro_input = frame
        self.update_hydrology()

    def update_hydrology(self):
        station = self.station_choice.currentData()
        self.bed_label.setText(f"{STATIONS[station]['bed_m']:.1f}")
        if self.hydro_input is not None:
            try:
                self.hydro_observations = parse_hydrology(self.hydro_input, self.hydro_timezone.currentData())
                self.hydrology = self.hydro_observations.groupby(level=0).mean().sort_index()
            except Exception as exc:
                self.hydrology = None
                self.warn(exc)
        self.refresh_hydro_table()
        self.plot_hydrology()
        self.schedule_psd_plot()

    def refresh_hydro_table(self):
        rows = []
        if self.hydrology is not None:
            rated = rated_hydraulics(self.hydro_observations.iloc[:200], self.station_choice.currentData())
            times = rated.index.tz_convert(self.hydro_timezone.currentData())
            values = rated[["water_level_m", "discharge_m3_s", "W_eff_m"]].itertuples(index=False, name=None)
            rows = [
                (stamp.isoformat(), f"{stage:g}", f"{flow:g}" if np.isfinite(flow) else "",
                 f"{width:.6g}" if np.isfinite(width) else "—")
                for stamp, (stage, flow, width) in zip(times, values)
            ]
        self.fill_table(self.hydro_table, [
            self.tr_pair("时间", "Time"), self.tr_pair("水位高程 (m)", "Stage (m)"),
            self.tr_pair("流量 (m³/s)", "Discharge (m³/s)"), self.tr_pair("河宽 (m)", "River width (m)"),
        ], rows)

    def open_grains(self):
        path = self.select_input("选择粒径级配", "Select grain-size distribution", "Tables (*.csv *.tsv *.txt *.xlsx *.xls)")
        if not path:
            return
        try:
            frame = read_tabular_file(path)
            standardize_grain_dataframe(frame, self.grain_unit.currentData())
        except Exception as exc:
            self.warn(exc)
            return
        self.grain_input = frame
        self.grain_file_label.setText(Path(path).name)
        self.update_grains()

    def use_grain_percentiles(self):
        self.grain_input = None
        self.grain_file_label.clear()
        self.update_grains()

    def update_grains(self, *_):
        frame = getattr(self, "grain_input", None)
        for spin in self.grain_spins:
            spin.setEnabled(frame is None)
        self.grain_d50 = None
        try:
            if frame is None:
                self.grain_percentiles = tuple(spin.value() for spin in self.grain_spins)
                self.grain_distribution = build_grain_from_percentiles(*(d * 1000 for d in self.grain_percentiles))
                self.grain_d50 = self.grain_percentiles[1]
            else:
                self.grain_distribution = standardize_grain_dataframe(frame, self.grain_unit.currentData())
                self.grain_d50 = weighted_quantile_value(
                    self.grain_distribution["D_m"], self.grain_distribution["fraction"], 0.50,
                )
        except Exception as exc:
            self.grain_distribution = None
            self.warn(exc)
        self.grain_d50_label.setText("—" if self.grain_d50 is None else f"{self.grain_d50:.15g} m")
        self.refresh_grain_table()
        self.plot_hydrology()

    def refresh_grain_table(self):
        rows = [] if self.grain_distribution is None else [
            (f"{diameter:.6g}", f"{share * 100:.6g}", f"{passing * 100:.6g}")
            for diameter, share, passing in self.grain_distribution[["D_m", "fraction", "cum_fraction"]].to_numpy()
        ]
        self.fill_table(
            self.grain_table, [self.tr_pair("粒径 D (m)", "Grain size D (m)"),
                               self.tr_pair("粒级质量比例 (%)", "Class mass fraction (%)"),
                               self.tr_pair("累计通过率 (%)", "Cumulative passing (%)")], rows,
        )

    def prefilter_corners(self, nyquist):
        try:
            prefilter = tuple(float(x.strip()) for x in self.prefilter.text().replace("，", ",").split(","))
        except ValueError as exc:
            raise ValueError(self.tr_pair("预滤波需要四个数值。", "Enter four numeric prefilter corners.")) from exc
        if len(prefilter) != 4 or not 0 < prefilter[0] < prefilter[1] < prefilter[2] < prefilter[3] < nyquist:
            raise ValueError(self.tr_pair(
                f"预滤波须满足 0 < f1 < f2 < f3 < f4 < {nyquist:g} Hz。",
                f"Prefilter requires 0 < f1 < f2 < f3 < f4 < {nyquist:g} Hz.",
            ))
        return prefilter

    def requested_workers(self):
        if not self.processing_threads.lineEdit().hasAcceptableInput():
            raise ValueError(self.tr_pair("并行运算线程数须为正整数。", "Enter a positive integer for parallel workers."))
        workers, _ = self.processing_threads.locale().toInt(self.processing_threads.currentText())
        return workers

    def options(self, source, band, *, raw_only=False):
        prefilter = default_response_prefilter(source.fs) if raw_only else self.prefilter_corners(source.fs / 2)
        paz = self.config["waveform"]["manual_paz"]
        workers = self.requested_workers()
        return {
            "prefilter": prefilter,
            "paz": {
                "poles": [complex(value) for value in paz["poles"]],
                "zeros": [complex(value) for value in paz["zeros"]],
                "gain": self.gain.value(),
                "sensitivity": self.gain.value() * self.adc.value(),
            },
            "nperseg": self.fft_length.currentData(),
            "minutes": self.psd_minutes.currentData(),
            "overlap": self.psd_overlap.currentData(),
            "band": band,
            "workers": workers,
        }

    def validate_psd_options(self, source, options):
        nperseg = options["nperseg"]
        hop = nperseg - psd_overlap_samples(nperseg, options["overlap"])
        seconds = hop / source.fs
        if seconds > options["minutes"] * 60:
            raise ValueError(self.tr_pair(
                f"短谱步长 {seconds:.2f} s 大于 PSD 周期。请缩短 Hann 窗或延长 PSD 周期。",
                f"The spectral hop of {seconds:.2f} s exceeds the PSD interval. Choose a shorter Hann window or a longer PSD interval.",
            ))

    def model_params(self):
        if getattr(self, "grain_input", None) is None:
            current = tuple(spin.value() for spin in self.grain_spins)
            if current != self.grain_percentiles:
                self.update_grains()
        if self.grain_distribution is None:
            raise ValueError(self.tr_pair("请输入有效级配。", "Provide a valid grain-size distribution."))
        key = self.model_choice.currentData()
        values = {name: widget.value() for name, widget in self.model_widgets.items()}
        grain = self.grain_distribution
        imported = getattr(self, "grain_input", None) is not None
        d50 = self.grain_d50
        sediment = SedimentParams(rho_s=values["rho_s"], rho_f=values["rho_f"])
        return {
            "model_type": key, "D50": d50, "slope": values["slope"], "W": 1.0,
            "D50_source": "imported_distribution" if imported else "user_percentiles",
            "grain_percentiles": None if imported else self.grain_percentiles,
            "f_inv_min": values["fmin"], "f_inv_max": values["fmax"],
            "band_statistic": self.band_statistic.currentData(),
            "r0": values["r0"], "tau_c50": values["tau_c50"], "ks": values["ks"] or None,
            "transport_stage_form": self.tsai_stage.currentData(),
            "clip_tau_c": self.clip_tsai.isChecked() if key == "Tsai2012" else self.clip_multi.isChecked(),
            "grain_distribution": {"D_m": grain["D_m"].to_numpy(copy=True), "fraction": grain["fraction"].to_numpy(copy=True)},
            "sediment_params": {"rho_s": sediment.rho_s, "rho_f": sediment.rho_f},
            "seismic_params": {
                "Q0": values["Q0"], "eta": values["eta"], "restitution": values["restitution"],
                "fz": values["fz"], "Nzz": values["Nzz"], "vc0_override": values["vc0_override"] or None,
                "zeta_override": values["zeta_override"] if values["zeta_override"] >= 0 else None,
            },
            "hop_t_min": values["hop_min"], "hop_t_max": values["hop_max"],
            "hop_log_median": values["hop_median"], "hop_log_sigma": values["hop_sigma"],
            "empirical_params": {
                name: values[name] for name in (
                    "k", "m", "n", "tau_cr", "A_db", "B_db", "hydraulic_radius_depth_ratio"
                )
            },
        }

    def calibrate_psd(self):
        path = self.select_input("选择独立标定数据", "Select independent calibration data", "Tables (*.csv *.tsv *.txt *.xlsx)")
        if path:
            try:
                frame = read_tabular_file(path)
                a, b = SeismicHydraulicModel.calibrate_reference_psd(frame["psd_linear"], frame["shields"])
                self.model_widgets["A_db"].setValue(a)
                self.model_widgets["B_db"].setValue(b)
            except Exception as exc:
                self.warn(exc)

    def require_seismic_source(self, source):
        native = getattr(source, "native", source)
        if getattr(native, "is_hydrophone", False) or source.trace_id.rsplit(".", 1)[-1].upper() == "HYD":
            raise ValueError(self.tr_pair(
                "HYD 水听器可预览原始计数。地震速度 PSD 与推移质反演请选择 BHZ、BHN 或 BHE 等地震通道。",
                "HYD can be previewed in raw counts. Select a seismic channel for ground-velocity PSD or bedload inversion.",
            ))

    def fit_inversion_band(self):
        if self.imported_psd is None and not self.prepare_waveforms(self.fit_inversion_band):
            return
        try:
            source = self.imported_psd or self.source(attach_cache=False)
            self.require_seismic_source(source)
            corners = (source.metadata["prefilter"] if isinstance(source, PSDArchiveSource)
                       else self.prefilter_corners(source.fs / 2))
            lower, upper = inversion_passband(source, corners)
            if upper <= lower:
                raise ValueError(self.tr_pair("请调整预滤波参数。", "Adjust the prefilter parameters."))
            a, b = self.model_widgets["fmin"], self.model_widgets["fmax"]
            lo, hi = max(lower, a.value()), min(upper, b.value())
            if lo >= hi:
                lo, hi = lower, upper
            a.setValue(lo)
            b.setValue(hi)
        except ValueError as exc:
            self.warn(exc)

    def start_worker(self, worker, slot, *, retry=None):
        if self.task_active:
            return
        self.waveform_retry = retry
        if isinstance(worker, StreamingWorker):
            worker.options.update(english=self.english, pixels=max(400, self.canvas_wf.width()))
            self.live_data = {}
            self.live_artists = {}
            self.live_limits = None
            if not (worker.task == "psd" and isinstance(worker.source, PSDArchiveSource)):
                worker.partial_sig.connect(self.partial_ready)
        self.worker = worker
        self.task_active = True
        self.update_task_actions()
        self.progress_bar.setRange(0, 0 if isinstance(worker, ResultExportWorker) else 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.stop_button.setVisible(isinstance(worker, StreamingWorker))
        worker.progress_sig.connect(self.progress)
        worker.finished_sig.connect(slot)
        worker.finished.connect(self.processing_finished)
        worker.error_sig.connect(self.warn)
        worker.start()

    def progress(self, value, text):
        self.progress_bar.setValue(value)
        if isinstance(self.worker, StreamingWorker):
            active, requested = self.worker.active_workers, self.worker.max_workers
            text += self.tr_pair(f" · 并行块 {active}（请求 {requested}）", f" · Parallel blocks {active} (requested {requested})")
            if self.worker.fft_threads > 1:
                threads = self.worker.fft_threads
                text += self.tr_pair(f" · PSD FFT {threads} 线程/块", f" · PSD FFT {threads} threads/block")
        self.statusBar().showMessage(text)

    def processing_finished(self):
        self.task_active = False
        self.progress_bar.hide()
        self.stop_button.hide()
        self.statusBar().clearMessage()
        self.update_task_actions()
        if isinstance(self.worker, StreamingWorker) and self.worker.input_issue is not None:
            self.report_waveform_read_issue(self.worker.input_issue, self.waveform_retry)

    def report_waveform_read_issue(self, issue, retry):
        selected = self.selected_waveform_records()
        self.show_waveform_issues(
            selected, [issue], lambda excluded: self.resume_waveform_task(excluded, retry),
        )

    def stop_processing(self):
        if self.worker is not None:
            self.worker.requestInterruption()

    def closeEvent(self, event):
        if self.task_active:
            self.stop_processing()
            event.ignore()
            return
        if self.result_cache is not None:
            self.result_cache.close()
            self.result_cache = None
        self.psd_cache.close()
        super().closeEvent(event)

    def start_waveform_preview(self):
        if self.task_active or not self.prepare_waveforms(self.start_waveform_preview):
            return
        try:
            source = self.source()
            raw_only = getattr(source, "native", source).is_hydrophone
            options = self.options(source, (1, source.fs / 2), raw_only=raw_only)
        except Exception as exc:
            self.warn(exc)
            return
        self.canvas_wf.reset()
        self.waveform_preview = []
        worker = StreamingWorker(source, options, "waveforms", waveform_cache=self.waveform_cache)
        self.start_worker(worker, self.waveform_ready, retry=self.start_waveform_preview)

    def partial_ready(self, task, payload):
        if task == "psd":
            self.live_data.setdefault(task, []).append(payload)
        else:
            self.live_data[task] = payload
        self.draw_live()

    def draw_live(self, *_):
        if not self.task_active or not isinstance(self.worker, StreamingWorker):
            return
        task = self.worker.task
        pages = {"waveforms": 0, "psd": 2, "inversion": 4}
        data = getattr(self, "live_data", {}).get(task)
        if self.tabs.currentIndex() != pages.get(task) or data is None:
            return
        if task == "waveforms":
            self.waveform_preview = data
            self.plot_waveform_canvas()
        elif task == "inversion":
            self.plot_results(data)
        else:
            if not data:
                return
            frame = pd.concat(data)
            data.clear()
            lo, hi = self.worker.source.start, self.worker.source.end
            frequencies = frame.columns.to_numpy(dtype=float)
            if task not in self.live_artists:
                grid = np.full((len(frequencies), 1000), np.nan, dtype=np.float32)
                lo_sec, hi_sec = epoch_seconds([lo, hi])
                hz_step = frequencies[1] - frequencies[0] if len(frequencies) > 1 else .002
                extent = (lo_sec, hi_sec, frequencies[0] - hz_step / 2, frequencies[-1] + hz_step / 2)
                mesh = self.build_psd_chart(grid, extent[:2], extent=extent)
                self.live_artists[task] = mesh, grid
            mesh, grid = self.live_artists[task]
            db, positive = psd_display_db(frame.to_numpy(), self.db_min.minimum())
            interval = pd.Timedelta(minutes=self.worker.options["minutes"])
            period = interval * max(1, int(np.ceil(len(self.worker.source.hours()) *
                                                    (60 // self.worker.options["minutes"]) / 1600)))
            centers = frame.index - period / 2
            pos = np.clip(((centers-lo)/(hi-lo)*grid.shape[1]).astype(int), 0, grid.shape[1]-1)
            for i, left in enumerate(pos):
                end = centers[i] + period
                if i + 1 < len(pos):
                    end = min(end, centers[i + 1])
                right = int(np.ceil((end - lo) / (hi - lo) * grid.shape[1]))
                grid[:, left:min(grid.shape[1], max(left + 1, right))] = db[i, :, None]
            in_range = (frequencies >= self.psd_fmin.value()) & (frequencies <= self.psd_fmax.value())
            finite = db[positive & in_range[None, :]]
            if self.db_auto.isChecked() and self.live_limits is None and finite.size:
                low, high = np.percentile(finite, (5, 95))
                low = np.clip(low, self.db_min.minimum(), self.db_max.maximum() - 0.1)
                self.live_limits = self.set_display_spin_range(self.db_min, self.db_max, low, max(high, low + 0.1))
                self.refresh_psd_style()
            mesh.set_data(grid)
            self.canvas_psd.draw_idle()

    def waveform_ready(self, series):
        self.waveform_preview = series
        self.plot_waveform_canvas()

    def refresh_psd_source(self):
        source = self.imported_psd
        if source is None:
            label = self.tr_pair("数据来源：原始波形", "Source: waveforms")
        else:
            stamp = source.start.tz_convert(BEIJING)
            end = source.end.tz_convert(BEIJING)
            label = self.tr_pair("数据来源：已导入的 PSD", "Source: imported PSD")
            label += (f"\\n{source.path.name} · {source.fs:g} Hz · {source.metadata['minutes']} min"
                      f"\\n{stamp:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UTC+8")
        if self.last_psd_export is not None:
            label += "\\n" + self.tr_pair("已导出", "Exported") + f" · {self.last_psd_export.name}"
        self.psd_source_label.setText(label)

    def set_psd_frequency_cap(self, maximum):
        with QSignalBlocker(self.psd_fmin), QSignalBlocker(self.psd_fmax):
            self.psd_fmin.setMaximum(maximum)
            self.psd_fmax.setMaximum(maximum)

    def import_psd_data(self):
        if self.task_active:
            return
        path = self.select_input("导入 PSD 时频数据", "Import PSD data", PSD_ARCHIVE_FILTER)
        if not path:
            return
        try:
            source = PSDArchiveSource(path)
            options = source.options(source.band, self.requested_workers())
            self.validate_psd_options(source, options)
        except Exception as exc:
            self.warn(exc)
            return
        worker = StreamingWorker(source, options, "psd")
        self.start_worker(worker, lambda frame: self.activate_imported_psd(source, frame))

    def activate_imported_psd(self, source, frame):
        if frame.empty or self.worker.isInterruptionRequested():
            return
        self.imported_psd = source
        self.set_psd_frequency_cap(source.band[1])
        station = self.station_choice.findData(source.metadata.get("station"))
        if station >= 0:
            self.station_choice.setCurrentIndex(station)
        self.refresh_psd_source()
        self.spectra_ready(frame)

    def waveform_export_directory(self, source):
        if isinstance(source, PSDArchiveSource):
            return source.path.parent
        native = getattr(source, "native", source)
        records = records_overlapping_window(native.records, source.start, source.end)
        first = min(records or native.records, key=lambda r: r.start_utc)
        return Path(first.path).parent

    def export_psd_data(self, *, destination=None):
        if self.task_active or (self.imported_psd is None and not self.prepare_waveforms(
            lambda: self.export_psd_data(destination=destination),
        )):
            return
        try:
            source = self.imported_psd or self.source()
            self.require_seismic_source(source)
            if isinstance(source, PSDArchiveSource):
                options = source.options(source.band, self.requested_workers())
            else:
                options = self.options(source, (0.0, source.fs / 2))
            self.validate_psd_options(source, options)
        except Exception as exc:
            self.warn(exc)
            return
        station = self.station_choice.currentData()
        if destination is None:
            stamp = source.start.tz_convert(BEIJING)
            default = self.waveform_export_directory(source) / f"OBS_{station}_PSD_{stamp:%Y%m%d_%H%M%S}.obspsd"
            path, _ = QFileDialog.getSaveFileName(
                self, self.tr_pair("导出 PSD 到新文件夹", "Export PSD into a new folder"), str(default),
                PSD_ARCHIVE_FILTER, options=QFileDialog.Option.DontConfirmOverwrite,
            )
            if not path:
                return
            destination = Path(path)
            if destination.suffix.lower() != ".obspsd":
                destination = destination.with_suffix(".obspsd")
        settings = {"psd_output": destination, "station": station, "new_export_folder": True}
        worker = StreamingWorker(source, options, "psd_archive", settings, psd_cache=self.psd_cache)
        self.start_worker(worker, self.psd_export_ready, retry=lambda: self.export_psd_data(destination=destination))

    def psd_export_ready(self, path):
        if path is not None:
            self.directory = str(path.parent)
            self.last_psd_export = path
            self.refresh_psd_source()

    def start_psd_preview(self):
        if self.task_active:
            return
        if self.imported_psd is not None:
            self.plot_psd_canvas()
            return
        if not self.prepare_waveforms(self.start_psd_preview):
            return
        try:
            source = self.source()
            self.require_seismic_source(source)
            # Keep the analysis spectrum so display changes do not rerun FFTs.
            options = self.options(source, (0.0, source.fs / 2))
            self.validate_psd_options(source, options)
        except Exception as exc:
            self.warn(exc)
            return
        self.tabs.setCurrentIndex(2)
        self.df_psd = None
        self.psd_mesh = None
        self.canvas_psd.figure.clear()
        self.canvas_psd.draw_idle()
        self.start_worker(StreamingWorker(source, options, "psd", psd_cache=self.psd_cache),
                          self.spectra_ready, retry=self.start_psd_preview)

    def spectra_ready(self, frame):
        if getattr(self, "live_limits", None) is not None:
            frame.attrs["stream_color_limits"] = self.live_limits
        self.df_psd = frame
        self.tabs.setCurrentIndex(2)
        self.plot_psd_canvas()

    def start_inversion(self):
        if self.task_active or (self.imported_psd is None and not self.prepare_waveforms(self.start_inversion)):
            return
        try:
            source = self.imported_psd or self.source()
            self.require_seismic_source(source)
            params = self.model_params()
            band = (params["f_inv_min"], params["f_inv_max"])
            options = (source.options(band, self.requested_workers()) if isinstance(source, PSDArchiveSource)
                       else self.options(source, band))
            self.validate_psd_options(source, options)
            if self.hydrology is None or self.hydrology.empty:
                raise ValueError(self.tr_pair("请先导入水文数据。", "Load hydrology data first."))
            if self.hydrology.index[-1] < source.start or self.hydrology.index[0] >= source.end:
                raise ValueError(self.tr_pair(
                    "水文数据与地震波形在 UTC 时间上无重叠，请检查水文时区和波形区间。",
                    "Hydrology and waveforms do not overlap in UTC. Check the time zone and selection.",
                ))
            lower, upper = inversion_passband(source, options["prefilter"])
            if not lower <= band[0] < band[1] <= upper:
                raise ValueError(self.tr_pair(
                    f"请在反演模型页调整频带至有效通带 {options['prefilter'][1]:g}–{upper:g} Hz 内。",
                    f"Adjust the inversion band on the model page to the effective passband (upper limit {upper:g} Hz).",
                ))
            if params["model_type"] in MULTIMODE_TYPES and not params["hop_t_min"] < params["hop_t_max"]:
                raise ValueError(self.tr_pair("冲击间隔下限须小于上限。", "Impact interval limits are reversed."))
            f = (source.frequencies if isinstance(source, PSDArchiveSource)
                 else np.fft.rfftfreq(options["nperseg"], 1 / source.fs))
            if np.count_nonzero((f >= band[0]) & (f <= band[1])) < 2:
                raise ValueError(self.tr_pair("反演频带至少需两个 FFT 频点。", "The inversion band needs two FFT bins."))
        except Exception as exc:
            self.warn(exc)
            return
        settings = {"water": self.hydrology, "station": self.station_choice.currentData(),
                    "params": params, "result_minutes": 60,
                    "export_base": self.waveform_export_directory(source)}
        self.tabs.setCurrentIndex(4)
        self.canvas_result.figure.clear()
        self.canvas_result.draw_idle()
        worker = StreamingWorker(source, options, "inversion", settings, psd_cache=self.psd_cache)
        self.start_worker(worker, self.inversion_ready, retry=self.start_inversion)

    def inversion_ready(self, frame):
        cache = self.worker.result_cache
        self.worker.result_cache = None
        if frame.empty:
            cache.close()
            self.plot_results()
            if self.df_result is None:
                self.result_info.setText(self.tr_pair("没有生成可预览的结果。", "No results to preview."))
            return
        if self.result_cache is not None:
            self.result_cache.close()
        self.result_cache = cache
        self.output_path = None
        self.df_result = frame
        self.df_result.attrs["export_base"] = self.worker.settings["export_base"]
        self.result_scales = aggregate_inversion_scales(frame)
        self.refresh_result_table()
        self.plot_results()

    def save_results(self):
        if self.task_active or self.result_cache is None:
            return
        station = self.df_result["station"].iat[0]
        start = self.df_result.attrs["selection_start"]
        parameters = self.df_result.attrs["model_parameters"] + (
            ("output_scales", "1h,1d,1month"),
            ("period_timezone", "Asia/Shanghai"),
            ("rate_aggregation", "period_mass_kg / valid_seconds"),
        )
        base = self.df_result.attrs.get("export_base") or self.directory
        directory = QFileDialog.getExistingDirectory(
            self, self.tr_pair("选择结果保存位置", "Choose a location for the results"),
            str(base), options=QFileDialog.Option.ShowDirsOnly,
        )
        if not directory:
            return
        filename = result_export_stem(station, start, None) + self.result_export_format.currentData()
        path = Path(directory) / filename
        try:
            images = self.export_previews()
        except Exception as exc:
            self.warn(exc)
            return
        worker = ResultExportWorker(self.result_cache, path, images, parameters, self.result_scales)
        self.start_worker(worker, self.results_saved)

    def export_previews(self):
        if self.refresh_timer.isActive():
            self.refresh_timer.stop()
            self.plot_psd_canvas()
        charts = (
            ("01_waveforms", self.canvas_wf, 680),
            ("02_hydrology_grains", self.canvas_hydro, 550),
            ("03_psd", self.canvas_psd, 470),
            ("04_inversion", self.canvas_result, 550),
        )
        images = []
        width = 1040
        for name, panel, height in charts:
            if isinstance(panel, DataPlotPanel):
                if not panel.plots:
                    continue
                from PIL import Image

                png = self.render_qt_preview(panel, width, height)
                fig = Figure(figsize=(width / 100, height / 100), facecolor=self.plot_background)
                ax = fig.add_axes([0, 0, 1, 1])
                with Image.open(BytesIO(png)) as image:
                    ax.imshow(np.asarray(image.convert("RGB")), aspect="auto", interpolation="none")
                ax.set_axis_off()
                with BytesIO() as buffer:
                    fig.savefig(buffer, format="pdf", dpi=160)
                    pdf = buffer.getvalue()
            else:
                if not panel.figure.axes or (panel is self.canvas_psd and self.psd_mesh is None):
                    continue
                fig = panel.figure
                size = fig.get_size_inches()
                try:
                    fig.set_size_inches(width / 100, height / 100, forward=False)
                    with BytesIO() as png_buffer, BytesIO() as pdf_buffer:
                        fig.savefig(png_buffer, format="png", dpi=160)
                        fig.savefig(pdf_buffer, format="pdf", dpi=160)
                        png, pdf = png_buffer.getvalue(), pdf_buffer.getvalue()
                finally:
                    fig.set_size_inches(*size, forward=False)
            images.append((name, png, pdf))
        return images

    def render_qt_preview(self, panel, width, height):
        target = QRectF(0, 0, width, height)
        geometry = panel.ci.geometry()
        master = panel.time_plot
        old_x = None
        if master is not None and self.df_result is not None:
            old_x = master.viewRange()[0], master.vb.autoRangeEnabled()[0]
            master.setXRange(*epoch_seconds([self.df_result.attrs["selection_start"],
                                             self.df_result.attrs["selection_end"]]), padding=0)
        image = QImage(round(width * 1.6), round(height * 1.6), QImage.Format.Format_RGB32)
        image.fill(QColor(self.plot_background))
        raster = QPainter(image)
        try:
            panel.ci.setGeometry(target)
            panel.ci.layout.activate()
            panel.scene().render(raster, QRectF(image.rect()), panel.ci.sceneBoundingRect())
        finally:
            raster.end()
            panel.ci.setGeometry(geometry)
            if old_x is not None:
                master.setXRange(*old_x[0], padding=0)
                master.enableAutoRange(axis="x", enable=old_x[1])
        encoded = QBuffer()
        encoded.open(QIODevice.OpenModeFlag.WriteOnly)
        if not image.save(encoded, "PNG"):
            raise OSError("Cannot encode plot as PNG.")
        return bytes(encoded.data())

    def results_saved(self, paths):
        data_path = paths[0]
        self.output_path = str(data_path.parent)
        self.directory = str(data_path.parent)
        self.refresh_result_table()

    def refresh_result_table(self):
        result = self.df_result
        if result is None or result.empty:
            self.fill_table(self.result_table, [], [])
            return
        columns = ["start_beijing", "Qb_kg_s", "bedload_mass_kg", "coverage_fraction"]
        rows = [
            (t, f"{q:.5g}" if np.isfinite(q) else "—",
             f"{mass:.5g}" if np.isfinite(mass) else "—", f"{coverage:.1%}")
            for t, q, mass, coverage in result.loc[:, columns].tail(200).itertuples(index=False, name=None)
        ]
        self.fill_table(
            self.result_table,
            [self.tr_pair("北京时间", "Beijing time"), self.tr_pair("推移质断面输沙率 (kg/s)", "Cross-section bedload transport rate (kg/s)"),
             self.tr_pair("时段输沙量 (kg)", "Interval mass (kg)"), self.tr_pair("有效数据覆盖率", "Coverage")],
            rows,
        )
        label = " · ".join(f"{self.tr_pair(zh, en)} {len(self.result_scales[code]):,}"
                           for code, _, zh, en in DISPLAY_RESULT_SCALES)
        completion = (self.tr_pair("部分结果", "Partial results") if result.attrs.get("partial")
                      else self.tr_pair("计算完成", "Calculation complete"))
        destination = self.output_path or self.tr_pair("尚未导出", "Not exported")
        self.result_info.setText(
            f"{label}\n{completion} · {destination}"
        )

    def plot_waveform_canvas(self):
        if not self.waveform_preview:
            return
        panel = self.canvas_wf
        if len(panel.plots) != len(self.waveform_preview):
            panel.reset()
            for i in range(len(self.waveform_preview)):
                plot = panel.add_chart(i, self.wave_timezone.currentData())
                panel.ci.layout.setRowStretchFactor(i, 1)
                plot.getAxis("left").setWidth(82)
                plot.getAxis("bottom").setHeight(58)
                if i:
                    plot.setXLink(panel.time_plot)
                if i + 1 < len(self.waveform_preview):
                    plot.getAxis("bottom").setStyle(showValues=False)
                plot.addItem(pg.PlotDataItem(connect="finite", clipToView=True))
                panel.mark_gaps(plot, self.record_gaps)
            self.refresh_wave_style()
        for plot, (times, values) in zip(panel.plots, self.waveform_preview):
            plot.listDataItems()[0].setData(epoch_seconds(times), zero_missing_for_plot(values))

    def refresh_wave_style(self):
        color = "#ffffff" if self.is_dark_theme else "#000000"
        axes = self.canvas_wf.plots
        labels = (("水听器原始波形（计数）", "Hydrophone raw (counts)"),) if len(axes) == 1 else (
            ("原始波形（计数）", "Raw (counts)"), ("去趋势、去均值（计数）", "Detrended (counts)"),
            ("地表质点速度 (m/s)", "Ground velocity (m/s)"),
        )
        for plot, pair in zip(axes, labels):
            plot.setLabel("left", self.tr_pair(*pair))
            plot.listDataItems()[0].setPen(color, width=WAVEFORM_LINE_WIDTH)
            axis = plot.getAxis("bottom")
            axis.zone = self.wave_timezone.currentData()
            axis.picture = None
            axis.update()
        if axes:
            axes[-1].setLabel("bottom", self.wave_timezone.currentText())

    def plot_hydrology(self):
        fig = self.canvas_hydro.figure
        fig.clear()
        fig.set_facecolor(self.plot_background)
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=.11, right=.97, bottom=.13, top=.92, wspace=.33, hspace=.63)
        depth, flow, distribution, cumulative = axes.flat
        depth.set_title(self.tr_pair("水深", "Water depth"), fontweight="semibold")
        flow.set_title(self.tr_pair("流量", "Discharge"), fontweight="semibold")
        depth.set_ylabel(self.tr_pair("水深 (m)", "Water depth (m)"))
        flow.set_ylabel(self.tr_pair("流量 (m³/s)", "Discharge (m³/s)"))
        zone = ZoneInfo(self.hydro_timezone.currentData())
        if self.hydrology is not None and not self.hydrology.empty:
            observed = rated_hydraulics(self.hydrology, self.station_choice.currentData())
            dates = mdates.date2num(observed.index)
            for ax, key, color in ((depth, "H_eff_m", "#38a8d4"), (flow, "discharge_m3_s", "#e58f54")):
                ax.plot(dates, zero_missing_for_plot(observed[key]), color=color,
                        linewidth=HYDROLOGY_LINE_WIDTH, marker="o" if len(dates) == 1 else "None")
                self.time_axis(ax, zone)
        for ax in (depth, flow):
            ax.set_xlabel(self.hydro_timezone.currentText())

        distribution.set_title(self.tr_pair("粒径分布", "Grain-size distribution"), fontweight="semibold")
        cumulative.set_title(self.tr_pair("粒径累计分布", "Cumulative grain-size distribution"), fontweight="semibold")
        distribution.set_ylabel(self.tr_pair("粒级质量比例 (%)", "Class mass fraction (%)"))
        cumulative.set_ylabel(self.tr_pair("累计通过率 (%)", "Cumulative passing (%)"))
        if self.grain_distribution is not None and not self.grain_distribution.empty:
            grain = self.grain_distribution
            edges = np.exp(bin_edges(np.log(grain["D_m"].to_numpy()), np.log(1.2)))
            for ax, key, color in ((distribution, "fraction", "#38a8d4"),
                                   (cumulative, "cum_fraction", "#e58f54")):
                ax.bar(edges[:-1], zero_missing_for_plot(grain[key]) * 100, width=np.diff(edges),
                       align="edge", color=color, edgecolor=self.plot_background, linewidth=0.45)
                ax.set_xscale("log")
                ax.set_xlim(edges[0], edges[-1])
        for ax in (distribution, cumulative):
            ax.set_xlabel(self.tr_pair("粒径 (m)", "Grain diameter (m)"))
        cumulative.set_ylim(0, 105)
        for ax in axes.flat:
            self.style_plot(ax)
        self.canvas_hydro.draw_idle()

    def schedule_psd_plot(self):
        self.sync_psd_auto_fields()
        if getattr(self, "df_psd", None) is not None:
            self.refresh_timer.start(100)

    @property
    def plot_background(self):
        return "#0c0d0e" if self.is_dark_theme else "#ffffff"

    def style_plot(self, ax, grid=True):
        text = "#f1f3f4" if self.is_dark_theme else "#0f172a"
        spine = "#92989d" if self.is_dark_theme else "#526171"
        ax.set_facecolor(self.plot_background)
        ax.tick_params(colors=text, width=AXIS_LINE_WIDTH, labelsize=9)
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title,
                      ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            label.set_color(text)
        for edge in ax.spines.values():
            edge.set(color=spine, linewidth=AXIS_LINE_WIDTH)
        if grid:
            ax.grid(True, linewidth=0.55, color=spine, alpha=0.20)
            ax.set_axisbelow(True)
        else:
            ax.grid(False)

    @staticmethod
    def time_axis(ax, zone=BEIJING):
        locator = mdates.AutoDateLocator(tz=zone, minticks=3, maxticks=5)
        locator.intervald[mdates.HOURLY].append(24)
        locator.intervald[mdates.MINUTELY].append(60)
        locator.intervald[mdates.SECONDLY].append(60)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(CalendarDateFormatter("%Y-%m-%d", tz=zone))

    def update_psd_colors(self):
        if self.psd_mesh is None:
            return
        low, high = self.db_min.value(), self.db_max.value()
        self.psd_mesh.set_cmap(self.colormap.currentData())
        if high > low:
            self.psd_mesh.set_clim(low, high)
        self.canvas_psd.draw_idle()

    def refresh_psd_style(self):
        if self.psd_mesh is None:
            return
        ax = self.psd_mesh.axes
        ax.set_xlabel(self.tr_pair("北京时间", "Beijing time"))
        ax.set_ylabel(self.tr_pair("频率 (Hz)", "Frequency (Hz)"))
        if self.psd_depth_axis is not None:
            self.psd_depth_axis.set_ylabel(self.tr_pair("水深 (m)", "River depth (m)"))
        for axis in self.canvas_psd.figure.axes:
            self.style_plot(axis, grid=False)
        self.update_psd_colors()

    def shade_plot_gaps(self, ax, intervals):
        if intervals:
            dates = (epoch_seconds([stamp for pair in intervals for stamp in pair]) / 86400).reshape(-1, 2)
            bars = [(left, right - left) for left, right in dates if right > left]
            if bars:
                ax.broken_barh(bars, (0, 1), transform=ax.get_xaxis_transform(),
                               color="#92989d", alpha=0.26, linewidth=0, zorder=2)

    def build_psd_chart(self, data, x_range, *, edges=None, extent=None):
        fig = self.canvas_psd.figure
        fig.clear()
        fig.set_facecolor(self.plot_background)
        fig.subplots_adjust(left=.11, right=.88, bottom=.19, top=.86)
        ax = fig.add_subplot()
        low, high = self.db_min.value(), self.db_max.value()
        colors = dict(cmap=self.colormap.currentData(), vmin=low, vmax=max(high, low + 0.1))
        if extent is not None:
            left, right, bottom, top = extent
            x0, x1 = np.asarray([left, right]) / 86400
            image = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest",
                              extent=(x0, x1, bottom, top), **colors)
        else:
            x, y = edges
            image = ax.pcolormesh(x / 86400, y, data, shading="flat", rasterized=True, **colors)
        colorbar = fig.colorbar(image, ax=ax, location="top", pad=.075, aspect=40)
        colorbar.set_label("PSD [dB re (m/s)²/Hz]")
        ax.set_xlim(*(np.asarray(x_range) / 86400))
        ax.set_ylim(self.psd_fmin.value(), self.psd_fmax.value())
        self.time_axis(ax)
        if self.imported_psd is None:
            self.shade_plot_gaps(ax, self.record_gaps)
        self.psd_depth_axis = None
        if self.show_depth.isChecked() and self.hydrology is not None:
            water_depth = self.displayed_depth()
            if not water_depth.empty:
                depth = ax.twinx()
                depth.plot(water_depth.index, zero_missing_for_plot(water_depth),
                           color="#38bdf8", linewidth=DATA_LINE_WIDTH)
                if self.depth_max.value() > self.depth_min.value():
                    depth.set_ylim(self.depth_min.value(), self.depth_max.value())
                self.psd_depth_axis = depth
        self.psd_mesh = image
        self.refresh_psd_style()
        return image

    @staticmethod
    def set_display_spin_range(lower, upper, low, high):
        with QSignalBlocker(lower), QSignalBlocker(upper):
            lower.setValue(float(low))
            upper.setValue(float(high))
        return lower.value(), upper.value()

    def sync_psd_auto_fields(self):
        """Mirror automatic limits in their visible controls."""
        for toggle, fields in (
            (self.frequency_auto, (self.psd_fmin, self.psd_fmax)),
            (self.db_auto, (self.db_min, self.db_max)),
            (self.depth_auto, (self.depth_min, self.depth_max)),
            (self.time_auto, (self.psd_start, self.psd_end)),
        ):
            for field in fields:
                field.setEnabled(not toggle.isChecked())

        if self.frequency_auto.isChecked():
            try:
                corners = (self.imported_psd.metadata["prefilter"] if self.imported_psd is not None
                           else self.prefilter_corners(self.psd_fmax.maximum()))
                _, f2, f3, _ = corners
            except ValueError:
                # Leave the last valid limits visible while the user types.
                pass
            else:
                self.set_display_spin_range(self.psd_fmin, self.psd_fmax, f2, f3)

        if self.time_auto.isChecked():
            limits = None
            preview = self.df_psd
            if preview is not None and not preview.empty:
                if "time_start" in preview.attrs and "time_end" in preview.attrs:
                    limits = preview.attrs["time_start"], preview.attrs["time_end"]
                elif preview.index.size > 1:
                    limits = preview.index[0], preview.index[-1]
                else:
                    minutes = preview.attrs.get("psd_minutes", self.psd_minutes.currentData())
                    half = pd.Timedelta(minutes=minutes / 2)
                    limits = preview.index[0] - half, preview.index[0] + half
            elif self.active_records():
                limits = self.editor_utc(self.wave_start), self.editor_utc(self.wave_end)
            elif self.hydrology is not None and not self.hydrology.empty:
                limits = self.hydrology.index[0], self.hydrology.index[-1]
            if limits is not None and limits[1] > limits[0]:
                for editor, stamp in zip((self.psd_start, self.psd_end), limits):
                    with QSignalBlocker(editor):
                        editor.setDateTime(QDateTime.fromMSecsSinceEpoch(stamp.value // 10**6, editor.timeZone()))

        if self.depth_auto.isChecked():
            self.update_auto_depth()
        if self.db_auto.isChecked() and (self.df_psd is None or self.df_psd.empty):
            self.set_display_spin_range(self.db_min, self.db_max, -185, -100)

    def displayed_depth(self):
        if self.hydrology is None or self.hydrology.empty:
            return pd.Series(dtype=float)
        start, end = self.editor_utc(self.psd_start), self.editor_utc(self.psd_end)
        bed = STATIONS[self.station_choice.currentData()]["bed_m"]
        level = self.hydrology["water_level_m"]
        if end <= start:
            return (level - bed).clip(lower=0.0)
        if end < level.index[0] or start > level.index[-1]:
            return level.iloc[:0]
        left, right = level.index.searchsorted([start, end])
        stop = right + int(right < len(level) and level.index[right] == end)
        visible = (level.iloc[left:stop] - bed).clip(lower=0.0)

        # A line crossing a viewport boundary contributes the depth at the
        # boundary, not the full value of the out-of-view observation.
        edges = {}
        for stamp, pos in ((start, left), (end, right)):
            if not 0 < pos < len(level) or level.index[pos] == stamp:
                continue
            a, b = level.iloc[pos - 1:pos + 1].to_numpy(dtype=float)
            if np.isfinite([a, b]).all():
                fraction = (stamp - level.index[pos - 1]) / (level.index[pos] - level.index[pos - 1])
                depth_a, depth_b = max(0.0, a - bed), max(0.0, b - bed)
                edges[stamp] = depth_a + fraction * (depth_b - depth_a)
        return pd.concat((visible, pd.Series(edges, dtype=float))).sort_index() if edges else visible

    def update_auto_depth(self):
        values = self.displayed_depth().to_numpy(dtype=float)
        if values.size:
            displayed = zero_missing_for_plot(values)
            low, high = 0.9 * float(displayed.min()), 1.1 * float(displayed.max())
            high = max(high, low + 0.001)
        else:
            low, high = 0.0, 20.0
        return self.set_display_spin_range(self.depth_min, self.depth_max, low, high)

    def plot_psd_canvas(self):
        self.sync_psd_auto_fields()
        if self.df_psd is None or self.df_psd.empty:
            return
        fmin, fmax = self.psd_fmin.value(), self.psd_fmax.value()
        if fmax <= fmin:
            return
        psd = self.df_psd
        start, end = self.editor_utc(self.psd_start), self.editor_utc(self.psd_end)
        if end <= start:
            return
        psd = psd.loc[start:end]
        psd = psd.loc[:, (psd.columns >= fmin) & (psd.columns <= fmax)]
        if psd.empty:
            if self.db_auto.isChecked():
                self.set_display_spin_range(self.db_min, self.db_max, -185, -100)
            self.psd_mesh = None
            fig = self.canvas_psd.figure
            fig.clear()
            ax = fig.add_subplot()
            self.style_plot(ax, grid=False)
            ax.set_axis_off()
            ax.text(.5, .5, self.tr_pair("当前显示范围无数据", "No data in the display range"),
                    transform=ax.transAxes, ha="center", va="center",
                    color="#f1f3f4" if self.is_dark_theme else "#0f172a")
            self.canvas_psd.draw_idle()
            return
        p = psd.to_numpy().T
        y = psd.columns.to_numpy(dtype=float)
        if len(y) > 600:
            step = int(np.ceil(len(y) / 600))
            edges = np.arange(0, len(y), step)
            last = np.minimum(edges + step, len(y)) - 1
            y = (y[edges] + y[last]) / 2.0
            p = np.maximum.reduceat(np.where(np.isfinite(p), p, -np.inf), edges, axis=0)
        power_db, positive = psd_display_db(p, self.db_min.minimum())
        if self.db_auto.isChecked():
            limits = self.df_psd.attrs.get("stream_color_limits")
            if limits is None:
                limits = np.percentile(power_db[positive], [5, 95]) if positive.any() else (-185, -100)
            vmin, vmax = limits
            vmin = float(np.clip(vmin, self.db_min.minimum(), self.db_max.maximum() - 0.1))
            vmax = max(vmin + 0.1, float(vmax))
            vmin, vmax = self.set_display_spin_range(self.db_min, self.db_max, vmin, vmax)
        else:
            vmin, vmax = self.db_min.value(), self.db_max.value()
            if vmax <= vmin:
                return
        minutes = self.df_psd.attrs.get("psd_minutes", self.psd_minutes.currentData())
        x = bin_edges(epoch_seconds(psd.index), minutes * 30)
        y = bin_edges(y, .001)
        self.build_psd_chart(power_db, epoch_seconds([start, end]), edges=(x, y))

    def refresh_result_style(self):
        axes = self.canvas_result.figure.axes
        if len(axes) != 4:
            return
        for col, (_, _, zh, en) in enumerate(DISPLAY_RESULT_SCALES):
            rate, mass = axes[col], axes[2 + col]
            rate.set_title(self.tr_pair(zh, en), fontweight="semibold", fontsize=11)
            rate.set_ylabel(self.tr_pair("推移质断面输沙率 (kg/s)", "Bedload rate (kg/s)"))
            mass.set_ylabel(self.tr_pair("累计推移质输沙量 (t)", "Cumulative bedload (t)"))
            mass.set_xlabel(self.tr_pair("北京时间", "Beijing time"))
            for ax, light, dark in ((rate, "#187ca6", "#55bcee"), (mass, "#b65a25", "#f0aa72")):
                self.style_plot(ax)
                ax.lines[0].set_color(dark if self.is_dark_theme else light)
        self.canvas_result.draw_idle()

    def plot_results(self, result=None):
        final = result is None
        scales = self.result_scales if final else aggregate_inversion_scales(result)
        if scales is None or scales["1h"].empty:
            return
        fig = self.canvas_result.figure
        if not fig.axes:
            axes = fig.subplots(2, 2, sharex="col")
            fig.subplots_adjust(left=.11, right=.97, bottom=.18, top=.92, wspace=.33, hspace=.23)
            for axis in axes[1]:
                self.time_axis(axis)
            for ax in axes.flat:
                ax.plot([], [], linewidth=RESULT_LINE_WIDTH, solid_capstyle="round")
        for col, (code, _, _, _) in enumerate(DISPLAY_RESULT_SCALES):
            frame = scales[code]
            rate, mass = fig.axes[col], fig.axes[2 + col]
            series = (zero_missing_for_plot(frame["Qb_kg_s"]),
                      frame["cumulative_mass_kg"].to_numpy() / KG_PER_TONNE)
            for ax, values in zip((rate, mass), series):
                line = ax.lines[0]
                line.set_data(mdates.date2num(frame.index), values)
                line.set_marker("o" if len(frame) == 1 else "None")
                for band in tuple(ax.collections):
                    band.remove()
                ax.relim()
                ax.autoscale_view()
            spacing = pd.Timedelta(hours=1) if code == "1h" else pd.Timedelta(days=1)
            first = frame.index[0]
            margin = (first + spacing - first) * 0.06
            mass.set_xlim(first - margin, frame.index[-1] + spacing)
            if final:
                absent = frame.index[frame["valid_seconds"] <= 0]
                gaps = list(zip(absent, absent + spacing))
                for ax in rate, mass:
                    self.shade_plot_gaps(ax, gaps)
        self.refresh_result_style()


def main():
    """创建 Qt 应用并直接启动图形界面。"""

    app = QApplication(sys.argv)
    try:
        gui = OBSAnalyzerGUIV336()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "程序启动失败",
            f"程序依赖或内置配置初始化失败：\n{exc}",
        )
        return
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
