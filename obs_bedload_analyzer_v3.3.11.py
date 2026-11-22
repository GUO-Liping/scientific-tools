# -*- coding: utf-8 -*-
"""
OBS Bedload Analyzer v3.3.11｜PyQt6。

本文件合并核心算法、波形延迟读取、默认配置与 PyQt6 图形界面。
界面分为地震波形、水文资料、时频分析、反演模型和结果分析五页。

算法依据：
1. Tsai et al. (2012), DOI: 10.1029/2011GL050255；
2. Luong et al. (2024), DOI: 10.1029/2024JF007761；
3. Luong et al. (2026), DOI: 10.1029/2025WR040371；
4. seismic_bedload v0.2.0，commit
   e666876e5750d0f27ef6a3f94b638ef7eb9e16d7。

核心算法采用 UTC 时间体系、统一量纲与参数校验；
支持实测级配、有效水深和河宽计算、波形断档补零及反演诊断。
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
v3.3.6 水文预览添加河宽，导出结果时可选择保存位置。
v3.3.7 支持波形和 PSD 追加、可编辑结果表及具名文件夹导出。
v3.3.8 统一 PSD 时间轴的纳秒精度，支持独立导出已缓存的连续小时块。
v3.3.9 波形与 PSD 重叠追加由用户确认，优先保留已加载的有效数据。导出文件名使用实际保存范围的完整起止日期。
v3.3.10 支持同通道、同采样率 SAC 与 MiniSEED 的跨格式波形追加。
v3.3.11 跨格式追加时可手动对应不同的文件头通道标识。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.special import gamma as gamma_function
from scipy.stats import lognorm


APP_VERSION = "3.3.11"
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
    seconds = times.as_unit("ns").asi8.astype(float) / 1.0e9
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
  "config_version": "3.3.11",
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
    source_trace_id: str | None = None
    file_npts: int | None = None

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

    timestamp = pd.Timestamp(value).as_unit("ns")
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
            headers = [waveform_header(path, trace) for trace in found]
            if headers[0].format_name == "MSEED":
                total = sum(header.npts for header in headers)
                headers = [replace(header, file_npts=total) for header in headers]
            return headers, None
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

QTableView { background-color: #0c0d0e; color: #f1f3f4; gridline-color: #2d3032; border-radius: 6px; alternate-background-color: #1c1f21; }
QTableView::item { color: #f1f3f4; }
QTableView::item:selected { background-color: #2f3843; color: #ffffff; }
QTableView QTableCornerButton::section { background-color: #17191b; border: 1px solid #2d3032; }
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

QTableView { background-color: #ffffff; color: #0f172a; gridline-color: #cbd5e1; border-radius: 6px; alternate-background-color: #f1f5f9; }
QTableView::item { color: #0f172a; background-color: #ffffff; }
QTableView::item:selected { background-color: #0284c7; color: #ffffff; }
QTableView QTableCornerButton::section { background-color: #f8fafc; border: 1px solid #cbd5e1; }
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
    return parsed.tz_convert("UTC").as_unit("ns")


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
    seconds = index.as_unit("ns").asi8.astype(float) / 1.0e9
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


# =============================================================================
# 站点资料、流式处理及五步界面
# =============================================================================

import csv
import re
from threading import BoundedSemaphore
from scipy.fft import set_workers as fft_workers
from scipy.signal import firwin, resample_poly
import pyqtgraph as pg
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QBuffer, QIODevice, QRectF
from PyQt6.QtGui import QImage, QTransform, QKeySequence
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QGraphicsPathItem, QTableView, QMenu,
                             QInputDialog, QToolButton, QSplitterHandle)
from bisect import bisect_left, bisect_right
from hashlib import blake2b
from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, nullcontext
from functools import cached_property, lru_cache
from io import BytesIO, StringIO
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
    """Place the upper response taper between 95% and 99% of Nyquist."""
    fs = _require_positive("采样率 / sample rate", fs)
    return min(0.2, 0.02 * fs), min(0.5, 0.05 * fs), 0.475 * fs, 0.495 * fs


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
        index=utc.as_unit("ns"),
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
    index = valid.index.as_unit("ns")
    centers = centers.as_unit("ns")
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
        return pd.date_range(self.start.floor("h"), self.end.ceil("h"), freq="h", inclusive="left", unit="ns")

    def sample(self, timestamp):
        return self.native.sample(timestamp) / self.factor

    def timestamp(self, indices):
        return self.native.timestamp(np.asarray(indices) * self.factor)

    def fingerprint(self, padding_seconds):
        return ("halfband129-v1", self.factor, self.start, self.end,
                self.native.fingerprint(padding_seconds + 65 * self.factor / self.native.fs))

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
                if record.file_npts is None:
                    self.sizes[record.path] = self.sizes.get(record.path, 0) + 8 * record.npts
                else:
                    self.sizes[record.path] = 8 * record.file_npts
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
        first_record = min(r.start_utc for r in self.records if r.sampling_rate_hz == self.fs)
        self.origin = _as_utc_timestamp(first_record)
        period = 1e9 / self.fs
        # Use integer sample timing when an interval is exactly representable in nanoseconds.
        self.sample_period_ns = int(period) if period.is_integer() else None

    def sample(self, timestamp):
        return (timestamp - self.origin).value * self.fs / 1e9

    def timestamp(self, indices):
        if self.sample_period_ns is not None:
            elapsed = pd.to_timedelta(np.asarray(indices, dtype=np.int64) * self.sample_period_ns, unit="ns")
        else:
            elapsed = pd.to_timedelta(np.asarray(indices, dtype=float) / self.fs, unit="s")
        return self.origin + elapsed

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
        for path, record in records.items():
            try:
                files.append((path, record.source_trace_id, os.stat(path)))
            except OSError as exc:
                raise WaveformInputError(path, exc) from exc
        return (self.trace_id, self.fs, self.origin, self.start, self.end,
                tuple((path, native_id, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                      for path, native_id, info in files))

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
                native_id = record.source_trace_id or record.trace_id
                for trace in stream:
                    if waveform_trace_id(path, trace, record.format_name) == native_id and trace.stats.sampling_rate == self.fs:
                        copy_samples(grid_index(trace.stats.starttime.ns), trace.data)
            except Exception as exc:
                raise WaveformInputError(path, exc) from exc
        return (data, observed) if return_mask else data

    def hours(self):
        return pd.date_range(self.start.floor("h"), self.end.ceil("h"), freq="h", inclusive="left", unit="ns")


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


def psd_input_signature(source, plan):
    if plan is None:
        return (None,)
    active_start, active_end, _, _, centers, _, start, stop = plan
    native = getattr(source, "native", source)
    if isinstance(source, DecimatedWaveform):
        start, stop = (start - 64) * source.factor, (stop + 64) * source.factor
    paths = tuple(native._records_between(native.timestamp(start - 1), native.timestamp(stop)))
    return active_start, active_end, int(centers[0]), len(centers), start, stop, paths


def psd_sample_digest(samples):
    digest = blake2b(digest_size=16)
    for array in samples:
        digest.update(np.ascontiguousarray(array))
    return digest.digest()


def stream_hour_psd(source, hour, nperseg, minutes, paz, prefilter, band, *,
                    overlap=0.5, corrector=None, prepared=None):
    """Compute zero-padded short spectra, retaining independent coverage."""
    fs = source.fs
    interval = pd.Timedelta(minutes=minutes)
    starts = pd.date_range(hour, periods=60 // minutes, freq=interval, unit="ns")
    centers = starts + interval / 2
    full_frequency = np.fft.rfftfreq(nperseg, 1.0 / fs)
    keep = (full_frequency >= band[0]) & (full_frequency <= band[1])
    frequency = full_frequency[keep]
    if frequency.size < 2:
        raise ValueError("PSD 频带至少需要两个频点 / PSD band needs two bins.")
    psd = pd.DataFrame(0.0, index=centers, columns=frequency)
    seconds = np.zeros(len(starts))
    plan = psd_window(source, hour, nperseg, prefilter, overlap) if prepared is None else prepared[0]
    if plan is None:
        return psd, seconds
    active_start, active_end, noverlap, half, center_samples, guard, start, end = plan
    raw, observed = (source.read_samples(start, end, return_mask=True) if prepared is None else prepared[1])
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
        return pd.date_range(self.start.floor("h"), self.end.ceil("h"), freq="h", inclusive="left", unit="ns")

    def open_archive(self):
        return ZipFile(self.path)

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


def check_psd_compatibility(reference, candidate, name):
    labels = {"format": "归档格式", "units": "谱单位", "station": "站点", "trace_id": "波形通道",
              "minutes": "时间分辨率", "nperseg": "谱窗长度", "sample_rate_hz": "分析采样率",
              "input_sample_rate_hz": "原始采样率", "decimation_factor": "降采样倍率",
              "overlap": "重叠率", "band": "频率范围", "prefilter": "预滤波参数", "paz": "仪器响应"}
    for key, label in labels.items():
        if reference[key] != candidate[key]:
            raise ValueError(f"{name} · {label} ({key}) 不一致 / incompatible: "
                             f"{candidate[key]} ≠ {reference[key]}")


class MergedPSDSource(PSDArchiveSource):
    """Index archive hours without copying complete spectrograms into memory."""
    def __init__(self, sources):
        self.parts = {}
        for source in sources:
            for part in (source.parts.values() if isinstance(source, MergedPSDSource) else (source,)):
                self.parts.setdefault(part.path.resolve(), part)
        first = next(iter(self.parts.values()))
        self.start = min(p.start for p in self.parts.values())
        self.end = max(p.end for p in self.parts.values())
        self.metadata = dict(first.metadata, start_utc=self.start.isoformat(), end_utc=self.end.isoformat())
        self.fs, self.input_fs, self.factor = first.fs, first.input_fs, first.factor
        self.trace_id, self.frequencies, self.band, self.rows = first.trace_id, first.frequencies, first.band, first.rows
        self.path = first.path
        self.by_hour = {}
        for part in self.parts.values():
            check_psd_compatibility(first.metadata, part.metadata, part.path.name)
            for hour in part.hours():
                self.by_hour.setdefault(hour, []).append(part)
        self.fingerprint = tuple(p.fingerprint for p in self.parts.values())

    def file_fingerprint(self):
        return tuple(p.file_fingerprint() for p in self.parts.values())

    @contextmanager
    def open_archive(self):
        with ExitStack() as stack:
            yield {part.path: stack.enter_context(part.open_archive()) for part in self.parts.values()}

    def read(self, readers, hour):
        parts = self.by_hour.get(hour, ())
        if not parts:
            return np.zeros((self.rows, len(self.frequencies))), np.zeros(self.rows)
        values, coverage = parts[0].read(readers[parts[0].path], hour)
        if len(parts) > 1:
            values, coverage = values.copy(), coverage.copy()
            for part in parts[1:]:
                if (coverage > 0).all():
                    break
                incoming, seconds = part.read(readers[part.path], hour)
                take = (coverage == 0) & (seconds > 0)
                values[take], coverage[take] = incoming[take], seconds[take]
        return values, coverage


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
        self.archive_metadata = None
        self.directory = None
        self.key = None
        self.bytes_used = 0
        self.can_write = True
        self.inputs = {}
        self.pending = set()

    @staticmethod
    def same_inputs_or_append(previous, current):
        old_fp, *old_settings = previous
        new_fp, *new_settings = current
        if old_settings != new_settings:
            return False
        if old_fp[0] == "halfband129-v1" or new_fp[0] == "halfband129-v1":
            if old_fp[:2] != new_fp[:2]:
                return False
            old_fp, new_fp = old_fp[-1], new_fp[-1]
        if old_fp == new_fp:
            return True
        if old_fp[:3] != new_fp[:3]:
            return False
        old_files, new_files = old_fp[-1], new_fp[-1]
        old_paths = {info[0] for info in old_files}
        # Existing files must be unmodified and keep their read order.
        return (len(new_files) > len(old_files)
                and old_files == tuple(info for info in new_files if info[0] in old_paths))

    def select(self, key):
        if self.key == key:
            return
        if (self.key is not None and self.key[1] == key[1]
                and self.same_inputs_or_append(self.key[0], key[0])):
            self.pending.update(self.inputs)
        else:
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
        self.archive_metadata = None
        self.bytes_used = 0
        self.can_write = True
        self.inputs.clear()
        self.pending.clear()

    def load(self, hour):
        if self.directory is None:
            return None
        path = Path(self.directory.name) / f"{hour.value}.npy"
        try:
            with path.open("rb") as handle:
                return np.load(handle, allow_pickle=False), np.load(handle, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None

    def save(self, hour, values, coverage, inputs):
        if not self.has_space(values.nbytes + coverage.nbytes, hour):
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
            self.confirm(hour, inputs)
        except OSError:
            self.can_write = False
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def cached_archive(self):
        """Only the contiguous prefix with complete, full-frequency disk blocks is portable."""
        if self.directory is None or self.archive_metadata is None:
            raise ValueError("尚无可导出的全频率 PSD 缓存 / No full-frequency PSD blocks are cached.")
        metadata = self.archive_metadata.copy()
        start, end = (_as_utc_timestamp(metadata[key]) for key in ("start_utc", "end_utc"))
        folder = Path(self.directory.name)
        hour = start.floor("h")
        step = pd.Timedelta(hours=1)
        blocks = []
        while hour < end:
            path = folder / f"{hour.value}.npy"
            if hour in self.pending or not path.is_file():
                break
            blocks.append((hour, path))
            hour += step
        if not blocks:
            raise ValueError("首个小时块尚未缓存 / The first hour has not been cached.")
        metadata["end_utc"] = min(hour, end).isoformat()
        return metadata, blocks

    def confirm(self, hour, inputs):
        self.inputs[hour] = inputs
        self.pending.discard(hour)

    def has_space(self, data_bytes, hour=None):
        if hour in self.inputs:
            return self.can_write
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


class CachedPSDExportWorker(QThread):
    progress_sig = pyqtSignal(int, str)
    finished_sig = pyqtSignal(object)
    error_sig = pyqtSignal(str)

    def __init__(self, destination, metadata, blocks):
        super().__init__()
        self.destination, self.metadata, self.blocks = Path(destination), metadata, blocks

    def run(self):
        folder = None
        try:
            folder = reserve_export_folder(self.destination)
            path = folder / self.destination.name
            with write_psd_archive(path, self.metadata, self.isInterruptionRequested) as archive:
                for i, (hour, block) in enumerate(self.blocks, 1):
                    if self.isInterruptionRequested():
                        break
                    # The cache stores exactly the two NPY arrays required by an archive block.
                    archive.write(block, psd_block_name(hour))
                    self.progress_sig.emit(round(100 * i / len(self.blocks)), f"PSD · {i}/{len(self.blocks)} h")
            self.finished_sig.emit(path if path.is_file() else None)
        except Exception as exc:
            self.error_sig.emit(str(exc))
        finally:
            if folder is not None:
                remove_empty_export_folder(folder)


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
        except ValueError as exc:
            self.error_sig.emit(str(exc))
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
            if (cache.key is not None and cache.key[1] == full_band
                    and cache.same_inputs_or_append(cache.key[0], base_key)):
                cache_band = full_band
            cache.select((base_key, cache_band))
            station = (self.settings or {}).get("station") or getattr(self, "station", None)
            if cache_band == full_band and station is not None:
                cache.archive_metadata = psd_archive_metadata(source, self.options, station)
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
            inputs = None
            prepared = None
            newly_computed = False
            if full is not None and cache is not None and hour in cache.pending:
                plan = psd_window(source, hour, nperseg, prefilter, overlap)
                signature = psd_input_signature(source, plan)
                old_signature, old_digest = cache.inputs[hour]
                if signature == old_signature:
                    inputs = signature, old_digest
                elif plan is not None and signature[:-1] == old_signature[:-1]:
                    prepared = plan, source.read_samples(*plan[-2:], return_mask=True)
                    digest = psd_sample_digest(prepared[1])
                    if digest == old_digest:
                        inputs = signature, digest
                if inputs is None:
                    full = None
            if full is None:
                cache_result = cache is not None and cache.has_space(cache_bytes, hour)
                retain = self.psd_writer is not None or cache_result
                if cache_result:
                    if prepared is None:
                        plan = psd_window(source, hour, nperseg, prefilter, overlap)
                        samples = None if plan is None else source.read_samples(*plan[-2:], return_mask=True)
                        prepared = plan, samples
                    inputs = (psd_input_signature(source, prepared[0]),
                              None if prepared[1] is None else psd_sample_digest(prepared[1]))
                with fft_workers(self.fft_threads):
                    frame, coverage = stream_hour_psd(
                        source, hour, nperseg, minutes, paz, prefilter,
                        cache_band if retain else band, overlap=overlap, corrector=corrector, prepared=prepared,
                    )
                if retain:
                    full = frame.to_numpy(dtype=np.float64), coverage
                    newly_computed = True
            if full is not None:
                values, coverage = full
                index = (pd.date_range(hour, periods=60 // minutes, freq=pd.Timedelta(minutes=minutes), unit="ns")
                         + pd.Timedelta(minutes=minutes / 2))
                frame = pd.DataFrame(values[:, selected], index=index, columns=frequencies[selected])
            result = (frame, coverage) if transform is None else transform(frame, coverage)
            return result, full, newly_computed, inputs

        def next_read(hour):
            if cache is not None and cache.directory is not None:
                if (Path(cache.directory.name) / f"{hour.value}.npy").exists():
                    return None
            plan = psd_window(source, hour, nperseg, prefilter, overlap)
            return None if plan is None else plan[-2:]

        for hour, (result, full, newly_computed, inputs) in self._ordered_hours(
            hours, compute, None if imported else next_read,
        ):
            if cache is not None and inputs is not None:
                if newly_computed:
                    cache.save(hour, *full, inputs)
                else:
                    cache.confirm(hour, inputs)
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
                self.psd_reader = stack.enter_context(self.source.open_archive())
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
            result.attrs["processed_end"] = processed_end
            result.attrs["model_parameters"] = model_export_parameters(settings["params"]) + (
                ("input_sample_rate_Hz", getattr(source, "input_fs", getattr(source, "native", source).fs)),
                ("analysis_sample_rate_Hz", source.fs),
                ("decimation_factor", getattr(source, "factor", 1)),
                ("hann_samples", nperseg), ("overlap_fraction", overlap),
                ("PSD_minutes", minutes), ("prefilter_Hz", str(tuple(prefilter))),
                ("response_gain", paz["gain"]), ("response_sensitivity", paz["sensitivity"]),
                ("PSD_source", (" | ".join(p.name for p in source.parts) if isinstance(source, MergedPSDSource)
                                else source.path.name if isinstance(source, PSDArchiveSource) else "waveforms")),
                ("result_minutes", result_minutes),
            )
            return result
        append_overview(final=True)
        preview = pd.concat(spectra) if spectra else pd.DataFrame()
        preview.attrs.update(time_start=source.start, time_end=processed_end, psd_minutes=minutes)
        return preview


class PSDAppendWorker(StreamingWorker):
    """Build a merged source off the GUI thread and commit only after all hours validate."""
    def __init__(self, base, options, paths, station, cache):
        super().__init__(base, options, "append_psd", psd_cache=cache)
        self.paths, self.station = paths, station

    def _spectra(self):
        base = self.source
        archives = [part if isinstance(part, PSDArchiveSource) else PSDArchiveSource(part) for part in self.paths]
        if isinstance(base, PSDArchiveSource):
            reference = base.metadata
        else:
            reference = json.loads(json.dumps(psd_archive_metadata(base, self.options, self.station)))
        for part in archives:
            check_psd_compatibility(reference, part.metadata, part.path.name)
        temporary = None
        try:
            if not isinstance(base, PSDArchiveSource):
                temporary = TemporaryDirectory(prefix="obs-psd-")
                path = Path(temporary.name) / "waveforms.obspsd"
                self.task, self.settings = "psd_archive", {"psd_output": path, "station": self.station}
                if super()._spectra() is None:
                    return None
                base = PSDArchiveSource(path)
                base.temporary = temporary
                base.export_base = self.source_directory
            merged = MergedPSDSource([base, *archives])
            merged.export_base = getattr(base, "export_base", base.path.parent)
            self.source, self.settings, self.task = merged, None, "append_psd"
            self.done_hours = 0
            self.rate_history.clear()
            self.rate_history.append((time.monotonic(), 0))
            frame = super()._spectra()
            if self.isInterruptionRequested():
                return None
            temporary = None  # ownership stays with the backing waveform archive
            return merged, frame
        finally:
            if temporary is not None:
                temporary.cleanup()


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


def export_time_range(start, end):
    def format_stamp(value):
        stamp = _as_utc_timestamp(value).tz_convert(BEIJING)
        seconds = f"{stamp:%Y%m%d_%H%M%S}"
        nanos = stamp.value % 10**9
        return f"{seconds}-{nanos:09d}".rstrip("0") if nanos else seconds

    return f"{format_stamp(start)}_{format_stamp(end)}"


def result_export_stem(station, start, end):
    return f"OBS_{station}_UTC+8_{export_time_range(start, end)}_1h_1d_1month"


def psd_export_filename(path, start, end):
    """Replace the application's date suffix after a custom name or retry."""
    path = Path(path)
    stem = path.stem if path.suffix.lower() == ".obspsd" else path.name
    stamp = r"\d{8}_\d{6}(?:-\d{1,9})?"
    stem = re.sub(rf"_(?:UTC\+8_)?{stamp}_{stamp}$", "", stem)
    return path.with_name(f"{stem}_UTC+8_{export_time_range(start, end)}.obspsd")


def result_sheet(result, scales):
    """Use the complete hourly series that the worker retains for all three scales."""
    parameters = (*result.attrs["model_parameters"], ("output_scales", "1h,1d,1month"),
                  ("period_timezone", "Asia/Shanghai"), ("rate_aggregation", "period_mass_kg / valid_seconds"))
    def periods(frame):
        for stamp, rate, total in frame[["Qb_kg_s", "cumulative_mass_kg"]].itertuples():
            yield stamp.isoformat(), rate, total / KG_PER_TONNE
    station = result["station"].iat[0]
    empty = ("", "", "")
    return [(station, *(hour or empty), *(day or empty), *(month or empty), *(param or ("", "")))
            for hour, day, month, param in zip_longest(*(periods(scales[k]) for k, *_ in RESULT_SCALES), parameters)]


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

    def __init__(self, folder, sheet, images, overwrite):
        super().__init__()
        self.folder, self.sheet = folder, sheet
        self.images, self.overwrite = images, overwrite

    def run(self):
        folder = self.folder
        created = False
        try:
            try:
                folder.mkdir()
                created = True
            except FileExistsError:
                if not self.overwrite or not folder.is_dir():
                    raise
            with TemporaryDirectory(dir=folder) as staging:
                stage = Path(staging)
                data_name = folder.name + ".csv"
                headings, rows = self.sheet
                with (stage / data_name).open("x", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(headings)
                    writer.writerows(rows)
                names = [data_name]
                for name, png, pdf in self.images:
                    for suffix, data in ((".png", png), (".pdf", pdf)):
                        names.append(name + suffix)
                        (stage / names[-1]).write_bytes(data)
                # Replace only managed files; restore the previous set if a file is locked.
                backup = stage / "previous"
                backup.mkdir()
                managed = dict.fromkeys([*names, *(name + ext for name in
                    ("01_waveforms", "02_hydrology_grains", "03_psd", "04_inversion")
                    for ext in (".png", ".pdf"))])
                saved, installed = [], []
                try:
                    for name in managed:
                        target = folder / name
                        if target.exists():
                            if not self.overwrite or not target.is_file():
                                raise FileExistsError(target)
                            target.rename(backup / name)
                            saved.append(name)
                    for name in names:
                        (stage / name).rename(folder / name)
                        installed.append(name)
                except BaseException:
                    for name in installed:
                        (folder / name).unlink()
                    for name in saved:
                        (backup / name).rename(folder / name)
                    raise
            self.finished_sig.emit(tuple(folder / name for name in names))
        except Exception as exc:
            if created:
                remove_empty_export_folder(folder)
            self.error_sig.emit(str(exc))


class PreviewRows:
    """Format file and hydrology rows only as they become visible."""
    def __init__(self, count, fetch):
        self.count, self.fetch = count, lru_cache(maxsize=128)(fetch)

    def __len__(self):
        return self.count

    def __getitem__(self, row):
        return self.fetch(row)


class PreviewModel(QAbstractTableModel):
    """Render cells on demand and retain only the user's edits separately."""
    def __init__(self, parent):
        super().__init__(parent)
        self.headings, self.extra, self.rows, self.edits = [], [], [], {}
        self.count = 0
        self.token = None

    def load(self, headings, rows, token=None):
        self.beginResetModel()
        if token is None or self.token != token:
            self.edits, self.extra = {}, []
            self.count = len(rows)
        self.headings, self.rows, self.token = list(headings), rows, token
        self.count = max(self.count, len(rows))
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.count

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headings) + len(self.extra)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
            return (self.headings + self.extra)[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole, Qt.ItemDataRole.ToolTipRole):
            r, c = index.row(), index.column()
            if (r, c) in self.edits:
                return self.edits[r, c]
            if r < len(self.rows) and c < len(self.headings):
                return str(self.rows[r][c])
            return ""

    def flags(self, index):
        return super().flags(index) | Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        self.edits[index.row(), index.column()] = str(value)
        self.dataChanged.emit(index, index)
        return True

    def add_column(self, name):
        c = self.columnCount()
        self.beginInsertColumns(QModelIndex(), c, c)
        self.extra.append(name)
        self.endInsertColumns()

    def grow(self, rows):
        if rows > self.count:
            self.beginInsertRows(QModelIndex(), self.count, rows - 1)
            self.count = rows
            self.endInsertRows()

    def snapshot(self):
        headings = self.headings + self.extra
        rows, count, edits, width = self.rows, self.count, self.edits.copy(), len(self.headings)
        def values():
            for r in range(count):
                base = rows[r] if r < len(rows) else ()
                yield tuple(edits.get((r, c), base[c] if c < min(width, len(base)) else "")
                            for c in range(len(headings)))
        return headings, values()


class PreviewTable(QTableView):
    def __init__(self, parent, translate):
        super().__init__(parent)
        self.tr_pair = translate
        self.setModel(PreviewModel(self))
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.horizontalHeader().setDefaultSectionSize(165)
        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(
            lambda pos: self.show_menu(self.horizontalHeader().mapToGlobal(pos)))
        self.setToolTip(self.tr_pair("右键添加列，双击编辑。预览编辑仅作用于表格。",
                                      "Right-click to add columns. Double-click to edit the worksheet."))

    def show_menu(self, pos):
        menu = QMenu(self)
        for zh, en, fn in (("添加列", "Add column", self.add_column), ("剪切", "Cut", self.cut),
                           ("复制", "Copy", self.copy), ("粘贴", "Paste", self.paste)):
            menu.addAction(self.tr_pair(zh, en), fn)
        menu.exec(pos)

    def contextMenuEvent(self, event):
        self.show_menu(event.globalPos())

    def next_name(self):
        names = set(self.model().headings + self.model().extra)
        n = len(self.model().extra) + 1
        while self.tr_pair(f"新列 {n}", f"Column {n}") in names:
            n += 1
        return self.tr_pair(f"新列 {n}", f"Column {n}")

    def add_column(self):
        name, ok = QInputDialog.getText(self, self.tr_pair("添加列", "Add column"),
                                        self.tr_pair("列名", "Column name"), text=self.next_name())
        if not ok or not name.strip():
            return
        m = self.model()
        name = name.strip()
        if name in m.headings + m.extra:
            QMessageBox.warning(self, self.tr_pair("列名重复", "Duplicate name"), name)
            return
        m.add_column(name)
        m.grow(max(1, m.rowCount()))
        self.scrollTo(m.index(0, m.columnCount() - 1))

    def copy(self):
        ranges = self.selectionModel().selection()
        if not ranges:
            return
        rect = ranges[0]
        m = self.model()
        stream = StringIO(newline="")
        csv.writer(stream, dialect="excel-tab").writerows(
            [m.data(m.index(r, c)) for c in range(rect.left(), rect.right() + 1)]
            for r in range(rect.top(), rect.bottom() + 1))
        QApplication.clipboard().setText(stream.getvalue())

    def cut(self):
        self.copy()
        for index in self.selectedIndexes():
            self.model().setData(index, "")

    def paste(self):
        text = QApplication.clipboard().text()
        if not text:
            return
        values = list(csv.reader(StringIO(text, newline=""), dialect="excel-tab"))
        m = self.model()
        ranges = self.selectionModel().selection()
        r, c = (ranges[0].top(), ranges[0].left()) if ranges else (0, 0)
        m.grow(r + len(values))
        while m.columnCount() < c + max(map(len, values), default=0):
            m.add_column(self.next_name())
        for i, row in enumerate(values, r):
            for j, value in enumerate(row, c):
                m.setData(m.index(i, j), value)

    def keyPressEvent(self, event):
        for key, handler in ((QKeySequence.StandardKey.Copy, self.copy), (QKeySequence.StandardKey.Cut, self.cut),
                             (QKeySequence.StandardKey.Paste, self.paste)):
            if event.matches(key):
                handler()
                return
        super().keyPressEvent(event)


class SidebarSplitter(QSplitter):
    def createHandle(self):
        handle = QSplitterHandle(self.orientation(), self)
        button = QToolButton(handle)
        layout = QVBoxLayout(handle)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(button)
        layout.addStretch()
        button.setArrowType(Qt.ArrowType.LeftArrow)
        button.setFixedSize(16, 44)
        last_width = 445

        def update(*_):
            nonlocal last_width
            width = self.sizes()[0]
            if width > 0:
                last_width = width
            button.setArrowType(Qt.ArrowType.LeftArrow if width else Qt.ArrowType.RightArrow)

        def toggle():
            left, right = self.sizes()
            target = 0 if left else min(last_width, max(300, (left + right) * 2 // 3))
            self.setSizes([target, max(1, left + right - target)])
            update()

        button.clicked.connect(toggle)
        self.splitterMoved.connect(update)
        return handle


class IgnoreWheel:
    def wheelEvent(self, event):
        event.ignore()


class DataComboBox(IgnoreWheel, QComboBox):
    pass


class DataSpinBox(IgnoreWheel, QDoubleSpinBox):
    pass


class DataDateEdit(IgnoreWheel, QDateTimeEdit):
    pass


class OBSAnalyzerGUIV3311(QMainWindow):
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
        self.wave_origins = {}
        self.full_wave_extent = False
        self.skipped_waveform_paths = set()
        self.waveform_issue_dialog = None
        self.waveform_preview = []
        self.waveform_cache = {}
        self.psd_cache = HourlyPSDCache()
        self.imported_psd = None
        self.last_psd_export = None
        self.wave_psd_origin = None
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
            action(files, "追加波形数据…", "Append waveforms…", lambda: self.open_waveforms(append=True)),
            action(files, "追加 PSD 时频数据…", "Append PSD data…", self.append_psd_data),
        ]
        self.append_wave_action, self.append_psd_action = self.input_actions[-2:]
        self.cached_export_action = action(files, "仅导出已缓存 PSD…", "Export cached PSD only…", self.export_cached_psd)
        self.input_actions.append(self.cached_export_action)
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
        self.stop_action.setEnabled(busy and isinstance(self.worker, (StreamingWorker, CachedPSDExportWorker)))
        can_export = not busy and self.result_cache is not None
        self.save_action.setEnabled(can_export)
        self.save_button.setEnabled(can_export)
        self.wave_input_button.setEnabled(not busy)
        self.append_wave_button.setEnabled(not busy and bool(self.records))
        self.append_wave_action.setEnabled(self.append_wave_button.isEnabled())
        self.psd_append_button.setEnabled(not busy and (self.imported_psd is not None or self.df_psd is not None))
        self.append_psd_action.setEnabled(self.psd_append_button.isEnabled())
        self.invert_button.setEnabled(not busy)
        self.psd_import_button.setEnabled(not busy)
        self.psd_export_button.setEnabled(not busy)
        cached = not busy and self.psd_cache.directory is not None and self.psd_cache.archive_metadata is not None
        self.cached_export_button.setEnabled(cached)
        self.cached_export_action.setEnabled(cached)
        self.psd_compute_button.setEnabled(not busy and self.imported_psd is None)
        self.analysis_actions[1].setEnabled(not busy and self.imported_psd is None)
        for field in (self.psd_minutes, self.fft_length, self.psd_overlap):
            field.setEnabled(not busy and self.imported_psd is None)
        self.processing_threads.setEnabled(not busy)

    def split(self):
        splitter = SidebarSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(18)
        splitter.setChildrenCollapsible(False)
        panel, charts = QWidget(), QWidget()
        settings, graphics = QVBoxLayout(panel), QVBoxLayout(charts)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        splitter.addWidget(charts)
        splitter.setCollapsible(0, True)
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
        table = PreviewTable(self, self.tr_pair)
        layout.addWidget(table, 1)
        return table

    @staticmethod
    def fill_table(table, headings, rows, token=None):
        table.model().load(headings, rows, token)

    def setup_waveforms(self):
        page, left, right = self.split()
        actions = QHBoxLayout()
        self.wave_input_button = self.button("输入数据", "Input data", self.open_waveforms, "waveform")
        actions.addWidget(self.wave_input_button)
        self.append_wave_button = self.button("追加数据", "Append data", lambda: self.open_waveforms(append=True))
        actions.addWidget(self.append_wave_button)
        left.addLayout(actions)
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
        self.prefilter.setText(", ".join(map(str, default_response_prefilter(100))))
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
        self.psd_append_button = self.button("追加 PSD 时频数据…", "Append PSD data…", self.append_psd_data)
        self.cached_export_button = self.button("仅导出已缓存 PSD…", "Export cached PSD only…", self.export_cached_psd)
        self.psd_source_label = QLabel()
        self.psd_source_label.setWordWrap(True)
        for widget in (self.psd_compute_button, self.psd_import_button, self.psd_append_button, self.psd_export_button,
                       self.cached_export_button, self.psd_source_label):
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
        self.invert_button = self.button("开始反演", "Run inversion", self.start_inversion)
        self.save_button = self.button("导出结果", "Export results", self.save_results)
        left.addWidget(self.invert_button)
        left.addWidget(self.save_button)
        left.addWidget(self.label(
            "显示小时和日尺度结果，导出时同时包含自然月数据。各时段按北京时间划分，输沙率按有效时长加权，累计量截至该时段末。",
            "Charts show hourly and daily results. Exports include calendar months. Intervals use Beijing time, and rates are weighted by valid duration."
        ))
        self.result_info = QLabel()
        self.result_info.setWordWrap(True)
        left.addWidget(self.result_info)
        self.result_table = self.table(left)
        left.addWidget(self.label("双击编辑，右键添加列或剪切、复制、粘贴。导出 CSV 采用当前表格，图件使用反演计算值。",
                                  "Double-click to edit; right-click for column and clipboard actions. CSV uses the table; plots use calculated values."))
        self.canvas_result = self.matplotlib_canvas(right)
        self.add_tab(page, "5 结果分析", "5 Result analysis", "inversion")

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

    def open_waveforms(self, *, append=False):
        if self.task_active:
            return
        paths = self.select_input(
            "追加数据" if append else "输入数据", "Append data" if append else "Input data",
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
            if append:
                existing = {Path(r.path).resolve() for r in self.records if r.path not in self.skipped_waveform_paths}
                paths = [path for path in paths if Path(path).resolve() not in existing]
                if not paths:
                    self.warn(self.tr_pair("所选文件均已载入。", "All selected files are already loaded."))
                    return
            slot = self.append_headers_loaded if append else self.headers_loaded
            self.start_worker(HeaderScanWorker(paths, workers, allow_partial=True), slot)

    def append_headers_loaded(self, report):
        existing = [r for r in self.records if r.path not in self.skipped_waveform_paths]
        formats = {r.format_name.upper() for r in existing}
        rates = {r.sampling_rate_hz for r in existing}
        interoperable = {"SAC", "MSEED"}
        if len(rates) != 1:
            self.warn(self.tr_pair("已有波形须采用同一采样率才能追加。",
                                   "The loaded waveforms must have one sample rate before appending."))
            return
        if len(formats) != 1 and not formats <= interoperable:
            self.warn(self.tr_pair("已有波形的格式组合不支持追加。支持 SAC 与 MiniSEED 混合追加。",
                                   "The loaded format combination does not support appending. SAC and MiniSEED can be mixed."))
            return
        rate = next(iter(rates))
        channels = {r.trace_id: r.channel for r in existing}
        files = {}
        for r in report.records:
            files.setdefault(r.path, []).append(r)
        issues = list(report.issues)
        accepted = []
        unmatched = set()
        for path, records in files.items():
            incoming = {r.format_name.upper() for r in records}
            combined = formats | incoming
            found_rates = {r.sampling_rate_hz for r in records}
            if found_rates != {rate}:
                found = ", ".join(f"{fs:.16g}" for fs in sorted(found_rates))
                reason = self.tr_pair(f"采样率不匹配。已有 {rate:.16g} Hz，文件为 {found} Hz。",
                                      f"Sample rate mismatch. Loaded: {rate:.16g} Hz. File: {found} Hz.")
            elif len(combined) > 1 and not combined <= interoperable:
                allowed = ", ".join(sorted(formats))
                found = ", ".join(sorted(incoming))
                reason = self.tr_pair(f"格式不兼容。已有 {allowed}，文件为 {found}。",
                                      f"Incompatible file format. Loaded: {allowed}. File: {found}.")
            else:
                accepted.extend(records)
                if len(combined) > 1:
                    unmatched.update((r.format_name, r.trace_id) for r in records if r.trace_id not in channels)
                continue
            issues.append(WaveformReadIssue(path, reason, tuple(records)))

        def commit(excluded):
            added = [r for r in accepted if r.path not in excluded]
            to_map = {(r.format_name, r.trace_id): r.channel for r in added
                      if (r.format_name, r.trace_id) in unmatched}
            if to_map:
                mapping = self.choose_waveform_channels(added, to_map, existing, channels)
                if mapping is None:
                    return
                mapped = []
                for r in added:
                    target = mapping.get((r.format_name, r.trace_id), r.trace_id)
                    if target is None:
                        continue
                    if target != r.trace_id:
                        r = replace(r, trace_id=target, channel=channels[target], source_trace_id=r.trace_id)
                    mapped.append(r)
                added = mapped
            if added:
                retried = {r.path for r in added}
                old = [r for r in self.records if r.path not in retried]
                if not self.confirm_overlap(
                    ((r.trace_id, r.start_utc, r.end_utc, r.path) for r in existing),
                    ((r.trace_id, r.start_utc, r.end_utc, r.path) for r in added),
                    psd=False,
                ):
                    return
                # Read lower-priority files first, so existing finite samples win.
                combined = WaveformScanReport(list(reversed(added)) + old, [])
                self.commit_waveform_batch(combined, self.skipped_waveform_paths - retried, append=True)
        if issues:
            self.show_waveform_issues(accepted, issues, commit)
        else:
            commit(set())

    def choose_waveform_channels(self, records, unmatched, existing, channels):
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr_pair("对应波形通道", "Match waveform channels"))
        layout = QVBoxLayout(dialog)
        notice = QLabel(self.tr_pair(
            "请选择新文件通道对应的已有通道。确认合并前，请核实两者来自同一测点和分量，且数据单位及仪器响应一致。",
            "Choose the loaded channel corresponding to each incoming channel. Confirm the sensor, component, units and instrument response before merging.",
        ))
        notice.setWordWrap(True)
        layout.addWidget(notice)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        form = QFormLayout(panel)
        scroll.setWidget(panel)
        layout.addWidget(scroll)
        by_key = {(r.format_name, r.source_trace_id): r.trace_id for r in existing if r.source_trace_id}
        fields = {}
        for key, component in sorted(unmatched.items()):
            fmt, incoming = key
            box = QComboBox()
            box.addItem(self.tr_pair("跳过该通道", "Skip this channel"), None)
            box.addItem(self.tr_pair("作为独立通道导入", "Load as a separate channel"), incoming)
            code = component.upper()
            direction = code[-1:] if code[-1:] in ("Z", "N", "E") else None
            candidates = [trace_id for trace_id, channel in sorted(channels.items())
                          if channel.upper() == code or (direction and channel.upper().endswith(direction))]
            for target in candidates:
                box.addItem(self.tr_pair(f"合并到 {target}", f"Merge into {target}"), target)
            suggested = by_key.get(key)
            if suggested not in candidates:
                suggested = candidates[0] if len(candidates) == 1 else None
            if suggested is not None:
                box.setCurrentIndex(box.findData(suggested))
            fields[key] = box
            form.addRow(f"{incoming} ({fmt})", box)
        status = QLabel()
        status.setWordWrap(True)
        layout.addWidget(status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        proceed = buttons.addButton(self.tr_pair("确认通道对应", "Confirm channel mapping"),
                                    QDialogButtonBox.ButtonRole.AcceptRole)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setText(self.tr_pair("取消追加", "Cancel append"))
        cancel.setDefault(True)
        layout.addWidget(buttons)
        result = {}

        def validate():
            result.clear()
            result.update((key, box.currentData()) for key, box in fields.items())
            used = {}
            has_data = False
            for record in records:
                target = result.get((record.format_name, record.trace_id), record.trace_id)
                if target is None:
                    continue
                has_data = True
                key = record.path, target
                original = used.setdefault(key, record.trace_id)
                if original != record.trace_id:
                    status.setText(self.tr_pair(
                        f"{Path(record.path).name} 中的 {original} 与 {record.trace_id} 对应了同一个通道，请调整。",
                        f"{original} and {record.trace_id} in {Path(record.path).name} map to the same channel. Choose distinct destinations.",
                    ))
                    proceed.setEnabled(False)
                    return
            status.setText("" if has_data else self.tr_pair("请选择至少一个要追加的通道。",
                                                           "Select at least one channel to append."))
            proceed.setEnabled(has_data)

        for box in fields.values():
            box.currentIndexChanged.connect(validate)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        validate()
        dialog.resize(680, min(620, 220 + 40 * len(fields)))
        return result if dialog.exec() == QDialog.DialogCode.Accepted else None

    def confirm_overlap(self, existing, added, *, psd):
        """Confirm the earlier-data priority before either source is committed."""
        by_channel = {}
        for is_new, items in ((False, existing), (True, added)):
            for channel, start, end, path in items:
                by_channel.setdefault(channel, []).append((start, is_new, end, path))
        hits = {}
        for items in by_channel.values():
            old_end = new_end = None
            new_path = None
            for start, is_new, end, path in sorted(items):
                if is_new:
                    for other_end in (old_end, new_end):
                        if other_end is not None and (start < other_end if psd else start <= other_end):
                            hits.setdefault(path, start)
                    if new_end is None or end > new_end:
                        new_end, new_path = end, path
                else:
                    if new_end is not None and (start < new_end if psd else start <= new_end):
                        hits.setdefault(new_path, start)
                    if old_end is None or end > old_end:
                        old_end = end
        if not hits:
            return True
        box = QMessageBox(QMessageBox.Icon.Question,
                          self.tr_pair("检测到数据时间重叠", "Overlapping time ranges"),
                          self.tr_pair(f"有 {len(hits):,} 个追加文件存在时间重叠。",
                                       f"{len(hits):,} incoming files have overlapping time ranges."),
                          parent=self)
        box.setInformativeText(self.tr_pair(
            "保留先载入的有效 PSD 窗，仅补充缺测窗口和新增时段。" if psd else
            "保留先载入的有效波形样点，仅补充缺测样点和新增时段。",
            "Keep earlier valid PSD bins and fill missing bins or extend coverage." if psd else
            "Keep earlier valid waveform samples and fill gaps or extend coverage.",
        ))
        box.setDetailedText("\n".join(f"{stamp.tz_convert(BEIJING).isoformat(sep=' ')}  {path}"
                                       for path, stamp in hits.items()))
        confirm = box.addButton(self.tr_pair("仅补充并合并", "Fill gaps and merge"), QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(self.tr_pair("取消追加", "Cancel append"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is confirm

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

    def commit_waveform_batch(self, report, excluded, *, append=False):
        retained = [r for r in report.records if r.path not in excluded]
        if not retained:
            return
        if self.imported_psd is not None:
            self.imported_psd = None
            self.last_sample_rate = None
            self.set_spectrum_controls(self.waveform_psd_controls)
            self.refresh_psd_source()
        previous = self.trace_choice.currentData() if append else None
        self.full_wave_extent = append
        if not append:
            self.wave_origins.clear()
        first_by_channel = {}
        for r in retained:
            key = r.trace_id, r.sampling_rate_hz
            first_by_channel[key] = min(first_by_channel.get(key, r.start_utc), r.start_utc)
        for key, origin in first_by_channel.items():
            self.wave_origins.setdefault(key, origin)
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
            channel = previous if previous in channels else self.default_trace_id(retained)
            self.trace_choice.setCurrentIndex(self.trace_choice.findData(channel))
        self.trace_selected()
        self.update_task_actions()

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
        def row(i):
            r = records[i]
            trace_label = f"{r.source_trace_id} → {r.trace_id}" if r.source_trace_id else r.trace_id
            return (Path(r.path).name, trace_label, r.start_utc.tz_convert(zone).isoformat(),
                    r.end_utc.tz_convert(zone).isoformat(), f"{r.sampling_rate_hz:g}", str(Path(r.path).parent))
        rows = PreviewRows(len(records), row)
        self.fill_table(
            self.header_table,
            [self.tr_pair("文件", "File"), "Trace ID", self.tr_pair("开始", "Start"),
             self.tr_pair("结束", "End"), "Hz", self.tr_pair("所在目录", "Folder")],
            rows, (id(self.records), self.trace_choice.currentData(), zone, frozenset(self.skipped_waveform_paths)),
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
        original = ([r for r in self.records if r.path not in self.skipped_waveform_paths]
                    if self.full_wave_extent else self.channel_records())
        start, default_end = self.default_waveform_interval(original)
        end = max(r.end_utc + pd.Timedelta(seconds=1 / r.sampling_rate_hz) for r in original)
        if self.full_wave_extent:
            default_end = end
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
            "下表列出了无法读取、格式不兼容、采样率不匹配或无法匹配通道的文件。确认后将跳过这些文件。",
            "The table lists unreadable files and files with format, sample rate or channel mismatches. Confirm to skip them.",
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
        report = QTableWidget()
        report.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(report)
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
                headings = [
                    self.tr_pair("将跳过的文件", "Files to skip"), "Hz", self.tr_pair("开始", "Start"),
                    self.tr_pair("所在目录", "Folder"), self.tr_pair("原因", "Reason"),
                ]
                report.setColumnCount(len(headings))
                report.setHorizontalHeaderLabels(headings)
                report.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    for j, value in enumerate(row):
                        report.setItem(i, j, QTableWidgetItem(value))
                    report.item(i, 3).setToolTip(str(Path(row[3]) / row[0]))
                    report.item(i, 4).setToolTip(row[4])
            finally:
                report.setUpdatesEnabled(True)
            kept = len(by_rate.get(target, set()) - excluded)
            proceed.setEnabled(kept > 0)
            summary.setText(self.tr_pair(
                f"跳过 {len(excluded):,} 个文件 · 保留 {kept:,} 个文件 · 时间显示时区 {zone}",
                f"Skip {len(excluded):,} files · Keep {kept:,} files · Times in {zone}",
            ) if kept else self.tr_pair("本次没有可继续处理的文件。", "There are no usable files in this selection."))

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
        origin = self.wave_origins.get((native.trace_id, native.fs))
        if origin is not None and origin != native.origin:
            native.origin = origin
        if attach_cache:
            native.mseed_cache = MiniSeedReadCache(self.records)
        factor = self.decimation.currentData() or 1
        return native if factor == 1 else DecimatedWaveform(native, factor)

    @staticmethod
    def editor_utc(editor):
        return pd.Timestamp(editor.dateTime().toMSecsSinceEpoch(), unit="ms", tz="UTC").as_unit("ns")

    def invalidate_spectra(self):
        if getattr(self, "psd_fmax", None) is None:
            return
        if self.imported_psd is not None:
            self.update_prefilter()
            return
        dirty = self.df_psd is not None
        self.wave_psd_origin = None
        self.psd_append_button.setEnabled(False)
        if hasattr(self, "input_actions"):
            self.append_psd_action.setEnabled(False)
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
            rated = rated_hydraulics(self.hydro_observations, self.station_choice.currentData())
            times = rated.index.tz_convert(self.hydro_timezone.currentData())
            values = rated[["water_level_m", "discharge_m3_s", "W_eff_m"]].to_numpy()
            def row(i):
                stage, flow, width = values[i]
                return (times[i].isoformat(), f"{stage:g}", f"{flow:g}" if np.isfinite(flow) else "",
                        f"{width:.6g}" if np.isfinite(width) else "—")
            rows = PreviewRows(len(rated), row)
        self.fill_table(self.hydro_table, [
            self.tr_pair("时间", "Time"), self.tr_pair("水位高程 (m)", "Stage (m)"),
            self.tr_pair("流量 (m³/s)", "Discharge (m³/s)"), self.tr_pair("河宽 (m)", "River width (m)"),
        ], rows, (id(self.hydro_input), self.station_choice.currentData(), self.hydro_timezone.currentData()))

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
                               self.tr_pair("累计通过率 (%)", "Cumulative passing (%)")], rows, id(self.grain_distribution),
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
            if worker.task == "psd":
                worker.station = self.station_choice.currentData()
            worker.options.update(english=self.english, pixels=max(400, self.canvas_wf.width()))
            self.live_data = {}
            self.live_artists = {}
            self.live_limits = None
            if not isinstance(worker, PSDAppendWorker) and not (worker.task == "psd" and isinstance(worker.source, PSDArchiveSource)):
                worker.partial_sig.connect(self.partial_ready)
        self.worker = worker
        self.task_active = True
        self.update_task_actions()
        self.progress_bar.setRange(0, 0 if isinstance(worker, ResultExportWorker) else 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.stop_button.setVisible(isinstance(worker, (StreamingWorker, CachedPSDExportWorker)))
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
            title = (self.tr_pair(f"已合并 {len(source.parts)} 份数据", f"{len(source.parts)} merged datasets")
                     if isinstance(source, MergedPSDSource) else source.path.name)
            label += (f"\n{title} · {source.fs:g} Hz · {source.metadata['minutes']} min"
                      f"\n{stamp:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UTC+8")
        if self.last_psd_export is not None:
            label += "\n" + self.tr_pair("已导出", "Exported") + f" · {self.last_psd_export.name}"
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

    def append_psd_data(self):
        if self.task_active or (self.imported_psd is None and self.df_psd is None):
            return
        paths = self.select_input("追加 PSD 时频数据", "Append PSD data", PSD_ARCHIVE_FILTER, multiple=True)
        if not paths:
            return
        try:
            base = self.imported_psd
            if base is None:
                base, options, station = self.wave_psd_origin
                options = dict(options, workers=self.requested_workers())
            else:
                options = base.options(base.band, self.requested_workers())
                station = base.metadata["station"]
                loaded = set(base.parts) if isinstance(base, MergedPSDSource) else {base.path.resolve()}
                paths = [p for p in paths if Path(p).resolve() not in loaded]
                if not paths:
                    self.warn(self.tr_pair("所选 PSD 文件均已载入。", "These PSD files are already loaded."))
                    return
            archives = [PSDArchiveSource(path) for path in dict.fromkeys(paths)]
            reference = (base.metadata if isinstance(base, PSDArchiveSource) else
                         json.loads(json.dumps(psd_archive_metadata(base, options, station))))
            for part in archives:
                check_psd_compatibility(reference, part.metadata, part.path.name)
            current = base.parts.values() if isinstance(base, MergedPSDSource) else (base,)
            if not self.confirm_overlap(
                ((base.trace_id, p.start, p.end, str(getattr(p, "path", ""))) for p in current),
                ((p.trace_id, p.start, p.end, str(p.path)) for p in archives),
                psd=True,
            ):
                return
            worker = PSDAppendWorker(base, options, archives, station, self.psd_cache)
            worker.source_directory = self.waveform_export_directory(base)
        except Exception as exc:
            self.warn(exc)
            return
        self.start_worker(worker, lambda result: self.activate_imported_psd(*result) if result else None)

    def activate_imported_psd(self, source, frame):
        if frame.empty or self.worker.isInterruptionRequested():
            return
        if self.imported_psd is None:
            self.waveform_psd_controls = tuple(w.currentData() for w in (self.psd_minutes, self.fft_length, self.psd_overlap))
        self.imported_psd = source
        self.set_spectrum_controls(tuple(source.metadata[key] for key in ("minutes", "nperseg", "overlap")))
        self.wave_psd_origin = None
        self.last_psd_export = None
        self.set_psd_frequency_cap(source.band[1])
        station = self.station_choice.findData(source.metadata.get("station"))
        if station >= 0:
            self.station_choice.setCurrentIndex(station)
        self.refresh_psd_source()
        self.spectra_ready(frame)

    def set_spectrum_controls(self, values):
        for widget, value in zip((self.psd_minutes, self.fft_length, self.psd_overlap), values):
            with QSignalBlocker(widget):
                index = widget.findData(value)
                if index < 0:
                    widget.addItem(str(value), value)
                    index = widget.count() - 1
                widget.setCurrentIndex(index)

    def waveform_export_directory(self, source):
        if isinstance(source, PSDArchiveSource):
            return getattr(source, "export_base", source.path.parent)
        native = getattr(source, "native", source)
        records = records_overlapping_window(native.records, source.start, source.end)
        first = min(records or native.records, key=lambda r: r.start_utc)
        return Path(first.path).parent

    def export_cached_psd(self):
        if self.task_active:
            return
        try:
            metadata, blocks = self.psd_cache.cached_archive()
        except ValueError as exc:
            self.warn(exc)
            return
        start, end = (metadata[key] for key in ("start_utc", "end_utc"))
        default = psd_export_filename(Path(self.directory) / f"OBS_{metadata['station']}_PSD", start, end)
        title = self.tr_pair("仅导出已缓存 PSD", "Export cached PSD only")
        title += f" · {metadata['trace_id']} · {len(blocks)} h · UTC+8"
        file, _ = QFileDialog.getSaveFileName(
            self, title, str(default), PSD_ARCHIVE_FILTER,
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if file:
            path = psd_export_filename(file, start, end)
            self.start_worker(CachedPSDExportWorker(path, metadata, blocks), self.psd_export_ready)

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
        station = (source.metadata["station"] if isinstance(source, PSDArchiveSource)
                   else self.station_choice.currentData())
        if destination is None:
            base = self.waveform_export_directory(source) / f"OBS_{station}_PSD"
            default = psd_export_filename(base, source.start, source.end)
            path, _ = QFileDialog.getSaveFileName(
                self, self.tr_pair("导出 PSD 到新文件夹", "Export PSD into a new folder"), str(default),
                PSD_ARCHIVE_FILTER, options=QFileDialog.Option.DontConfirmOverwrite,
            )
            if not path:
                return
            destination = Path(path)
        destination = psd_export_filename(destination, source.start, source.end)
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
        if self.imported_psd is None:
            self.wave_psd_origin = self.worker.source, dict(self.worker.options), self.worker.station
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
        end = self.df_result.attrs.get("processed_end", self.df_result.attrs["selection_end"])
        base = self.df_result.attrs.get("export_base") or self.directory
        name = result_export_stem(station, start, end)
        suffix = f"_UTC+8_{export_time_range(start, end)}_1h_1d_1month"
        chosen = self.choose_result_folder(Path(base), name, suffix=suffix)
        if chosen is None:
            return
        folder, overwrite = chosen
        try:
            images = self.export_previews()
        except Exception as exc:
            self.warn(exc)
            return
        worker = ResultExportWorker(folder, self.result_table.model().snapshot(), images, overwrite)
        self.start_worker(worker, self.results_saved)

    def choose_result_folder(self, parent, name, *, suffix=""):
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr_pair("导出结果", "Export results"))
        form = QFormLayout(dialog)
        location, folder_name = QLineEdit(str(parent)), QLineEdit(name.removesuffix(suffix))
        destination = QLabel()
        destination.setWordWrap(True)
        def show_path():
            destination.setText(str(Path(location.text()) / (folder_name.text().strip() + suffix)))
        def browse():
            path = QFileDialog.getExistingDirectory(dialog, self.tr_pair("选择保存位置", "Choose parent folder"),
                                                    location.text())
            if path:
                location.setText(path)
        line = QHBoxLayout()
        line.addWidget(location)
        browse_button = QPushButton(self.tr_pair("浏览…", "Browse…"))
        browse_button.clicked.connect(browse)
        line.addWidget(browse_button)
        form.addRow(self.tr_pair("保存位置", "Location"), line)
        form.addRow(self.tr_pair("结果文件夹名称", "Result folder name"), folder_name)
        if suffix:
            form.addRow(self.tr_pair("文件名日期后缀", "File name date suffix"), QLabel(suffix.lstrip("_")))
        form.addRow(self.tr_pair("完整路径", "Full path"), destination)
        location.textChanged.connect(show_path)
        folder_name.textChanged.connect(show_path)
        show_path()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(dialog.reject)
        choice = []
        def accept():
            directory = Path(location.text().strip())
            name = folder_name.text().strip()
            if (not directory.is_dir() or not name or name in (".", "..") or name.endswith(".")
                    or any(ch in '<>:"/\\|?*' or ord(ch) < 32 for ch in name) or getattr(os.path, "isreserved", lambda _: False)(name)):
                self.warn(self.tr_pair("请选择有效的保存位置并填写文件夹名称。", "Choose a valid directory and folder name."))
                return
            target = directory / (name + suffix)
            overwrite = target.exists()
            if overwrite:
                if not target.is_dir():
                    self.warn(self.tr_pair("同名文件已存在，请更换文件夹名称。", "A file has this name. Choose another name."))
                    return
                box = QMessageBox(dialog)
                box.setWindowTitle(self.tr_pair("文件夹已存在", "Folder already exists"))
                box.setText(self.tr_pair(f"结果文件夹已存在。覆盖其中的结果 CSV 和图件？\n{target}",
                                         f"Replace result CSV and figures in the existing folder?\n{target}"))
                replace = box.addButton(self.tr_pair("覆盖结果", "Overwrite results"), QMessageBox.ButtonRole.DestructiveRole)
                rename = box.addButton(self.tr_pair("更换名称", "Rename"), QMessageBox.ButtonRole.RejectRole)
                box.setDefaultButton(rename)
                box.exec()
                if box.clickedButton() != replace:
                    folder_name.setFocus()
                    folder_name.selectAll()
                    return
            choice.append((target, overwrite))
            dialog.accept()
        buttons.accepted.connect(accept)
        form.addRow(buttons)
        dialog.resize(690, dialog.sizeHint().height())
        return choice[0] if dialog.exec() and choice else None

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
                                             self.df_result.attrs.get("processed_end", self.df_result.attrs["selection_end"])]), padding=0)
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
        if self.result_table.model().token != id(result):
            self.fill_table(self.result_table, EXPORT_COLUMNS, result_sheet(result, self.result_scales), id(result))
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
        gui = OBSAnalyzerGUIV3311()
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
