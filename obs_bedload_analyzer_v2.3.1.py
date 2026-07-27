# -*- coding: utf-8 -*-
"""
OBS Analyzer v2.3.0 精简维护版。

本文件合并核心算法、波形延迟读取、默认配置与 PyQt6 图形界面。
算法、界面行为和默认参数与 v2.3.0 标准发布包保持一致。

算法依据：
1. Tsai et al. (2012), DOI: 10.1029/2011GL050255；
2. Luong et al. (2024), DOI: 10.1029/2024JF007761；
3. Luong et al. (2026), DOI: 10.1029/2025WR040371；
4. seismic_bedload v0.2.0，commit
   e666876e5750d0f27ef6a3f94b638ef7eb9e16d7。

相对上游算法的论文校正式、参数边界和数值校验均沿用标准 v2.3.0，
本精简版只调整文件组织，不修改任何计算分支。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.special import gamma as gamma_function
from scipy.stats import lognorm


APP_VERSION = "2.3.0"
UPSTREAM_VERSION = "0.2.0"
UPSTREAM_COMMIT = "e666876e5750d0f27ef6a3f94b638ef7eb9e16d7"
EPS_PSD = 1.0e-40
MODEL_TYPES = ("Tsai2012", "Luong2024", "Luong2026")


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
    impact_azimuth_rad: float = 0.0

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
    if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} 网格至少含两个严格递增点。")
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

    def seismic_spectral_factor(
        self, frequency_hz: np.ndarray, source_distance_m: float
    ) -> np.ndarray:
        """计算 Tsai 2012 式(5)、(9)的瑞利波频散和衰减项。"""

        f = np.asarray(frequency_hz, dtype=float)
        r0 = _require_positive("台站距离 r0", source_distance_m)
        if f.ndim != 1 or f.size == 0 or np.any(f <= 0.0):
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
        u_star = np.sqrt(sd.g * depth * S)
        shields = u_star**2 / (R * sd.g * grains)
        critical = self.critical_shields_by_size(
            grains, d50, tau_c50, S, clip_tau_c
        )
        transport_stage = shields / critical

        # Tsai 2012 式(14)–(15)；U 的指数严格采用论文中的 1/6。
        roughness = 3.0 * d50
        flow_velocity = 8.1 * u_star * (depth / roughness) ** (1.0 / 6.0)
        bedload_velocity = (
            1.56 * np.sqrt(R * sd.g * grains) * transport_stage**0.56
        )
        layer_height = 1.44 * grains * transport_stage**0.50
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
            * np.sqrt(np.maximum(0.0, 1.0 - np.exp(-hb_hat)))
        )
        denominator = 2.0 * np.log(
            np.exp(hb_hat / 2.0) + np.sqrt(np.maximum(0.0, np.exp(hb_hat) - 1.0))
        )
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
        source = np.where(transport_stage >= 1.0, source, 0.0)
        spectral = self.seismic_spectral_factor(frequency, r0)
        psd_by_size = source[:, None] * spectral[None, :]
        total = np.trapezoid(density[:, None] * psd_by_size, grains, axis=0)
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
    ) -> np.ndarray:
        """按论文在 H<0.4 m 与较高水深分别选用式(14)和式(13)。"""

        sd = self.sediment
        R = (sd.rho_s - sd.rho_f) / sd.rho_f
        u_star = np.sqrt(sd.g * H * slope)
        shields = u_star**2 / (R * sd.g * D)
        roughness = 3.0 * D50
        upper_velocity = 8.1 * u_star * (H / roughness) ** (1.0 / 6.0)
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

        sd = self.sediment
        sp = self.seismic
        critical = self.critical_shields_by_size(
            grains, d50, tau_c50, S, clip_tau_c
        )
        velocity = self._bedload_velocity(grains, depth, S, d50, critical)
        volume = np.pi * grains**3 / 6.0
        mass = sd.rho_s * volume

        # ∫p(t)/t dt 可先计算，数学上等价于仓库中的二维梯形积分。
        inverse_hop_time = float(np.trapezoid(time_pdf / times, times))
        rate = np.divide(
            width * flux * inverse_hop_time,
            volume * velocity,
            out=np.zeros_like(volume),
            where=(volume * velocity) > 0.0,
        )

        azimuth = sp.impact_azimuth_rad
        nxz = 0.8 * np.cos(azimuth) if azimuth != 0.0 else 0.0
        nyz = 0.8 * np.sin(azimuth) if azimuth != 0.0 else 0.0
        ix = sp.fx * (1.0 + sp.restitution) * velocity
        iy = sp.fy * (1.0 + sp.restitution) * velocity
        iz = sp.fz * (1.0 + sp.restitution) * velocity
        directional_impulse_sq = (
            np.abs(ix * nxz) + np.abs(iy * nyz) + np.abs(iz * sp.Nzz)
        ) ** 2

        source = (
            rate
            * mass**2
            * np.pi**2
            * directional_impulse_sq
            / (4.0 * sd.rho_s**2)
        )
        spectral = self.seismic_spectral_factor(frequency, r0)
        psd_by_size = source[:, None] * spectral[None, :]
        total = np.trapezoid(grain_pdf[:, None] * psd_by_size, grains, axis=0)
        return np.maximum(total, 0.0)


class SeismicHydraulicModel:
    """Luong et al. (2026) 式(13)地震—水力经验反演模型。"""

    def __init__(
        self, sediment_params: SedimentParams | None = None
    ) -> None:
        self.sediment = sediment_params or SedimentParams()

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

    该函数只服务于可视化；预处理、PSD 和反演始终使用完整数据。
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
    """
    按所选 Tsai 2012、Luong 2024 或 Luong 2026 模型反演推移质通量。

    观测谱和单位通量模型谱均在线性 PSD 空间对反演频带取中位数，
    避免先对 dB 做算术平均所导致的偏差。
    """

    if df_psd.empty:
        raise ValueError("PSD 数据为空。")
    frequencies = np.asarray(df_psd.columns, dtype=float)
    fmin = _require_positive("反演频带下限", params["f_inv_min"])
    fmax = _require_positive("反演频带上限", params["f_inv_max"])
    if fmin >= fmax:
        raise ValueError("反演频带下限必须小于上限。")
    mask = (frequencies >= fmin) & (frequencies <= fmax)
    if np.count_nonzero(mask) < 2:
        raise ValueError("反演频带内至少需要两个 PSD 频率点。")
    n_frequency = max(2, int(params.get("n_frequency_bins", 60)))
    f_model = np.linspace(fmin, fmax, n_frequency)

    observed = np.asarray(df_psd, dtype=float)
    observed_on_grid = np.full((len(df_psd), n_frequency), np.nan)
    band_f = frequencies[mask]
    for index, row in enumerate(observed[:, mask]):
        finite = np.isfinite(row) & (row >= 0.0)
        if np.count_nonzero(finite) >= 2:
            observed_on_grid[index] = np.interp(
                f_model, band_f[finite], row[finite], left=np.nan, right=np.nan
            )
    observed_band = np.nanmedian(observed_on_grid, axis=1)
    depths = align_depth_to_psd(depth_series, df_psd.index)

    sediment = _dataclass_from_mapping(
        SedimentParams, params.get("sediment_params", {})
    )
    model_type = str(params["model_type"])
    if model_type not in MODEL_TYPES:
        raise ValueError(f"未知模型类型：{model_type}")

    reference = np.full(len(df_psd), np.nan)
    qb = np.full(len(df_psd), np.nan)
    diagnostics: dict[str, np.ndarray] = {}

    if model_type == "Luong2026":
        empirical = SeismicHydraulicModel(sediment)
        outputs = empirical.inverse_bedload(
            observed_band_psd_linear=observed_band,
            depth_m=depths,
            slope=params["slope"],
            D50=params["D50"],
            **params.get("empirical_params", {}),
        )
        qb = outputs["qb_vol_m2_s"]
        reference = 10.0 ** (outputs["reference_psd_db"] / 10.0)
        diagnostics = {
            "dimensionless_shields": outputs["shields"],
            "reference_psd_db": outputs["reference_psd_db"],
            "observed_reference_power_ratio": outputs["power_ratio"],
            "dimensionless_bedload_flux": outputs["dimensionless_flux"],
        }
    else:
        seismic = _dataclass_from_mapping(
            SeismicParams, params.get("seismic_params", {})
        )
        D, pdf_D = build_grain_distribution(
            D50=params["D50"],
            log_std=params["grain_log_std"],
            n_points=params.get("grain_points", 80),
            D_min=params.get("D_min"),
            D_max=params.get("D_max"),
        )
        if model_type == "Tsai2012":
            model: SaltationModel | MultimodeModel = SaltationModel(
                sediment, seismic
            )
            hop_times = pdf_t = None
        else:
            model = MultimodeModel(sediment, seismic)
            hop_times, pdf_t = build_hop_time_distribution(
                t_min=params.get("hop_t_min", 0.01),
                t_max=params.get("hop_t_max", 0.50),
                log_median=params.get("hop_log_median", 0.08),
                log_sigma=params.get("hop_log_sigma", 1.0),
                n_points=params.get("hop_points", 70),
            )
        valid = (
            np.isfinite(depths)
            & (depths > 0.0)
            & np.isfinite(observed_band)
        )
        for index in np.flatnonzero(valid):
            common = dict(
                f=f_model,
                D=D,
                H=float(depths[index]),
                W=params["W"],
                slope=params["slope"],
                r0=params["r0"],
                qb=1.0,
                D50=params["D50"],
                pdf_D=pdf_D,
                tau_c50=params.get("tau_c50", 0.045),
                clip_tau_c=bool(
                    params.get("clip_tau_c", model_type == "Luong2024")
                ),
            )
            if model_type == "Tsai2012":
                modeled = model.forward_psd(**common)
            else:
                modeled = model.forward_psd(
                    **common, hop_times=hop_times, pdf_t=pdf_t
                )
            reference[index] = float(np.nanmedian(modeled))
            if np.isfinite(reference[index]) and reference[index] > EPS_PSD:
                qb[index] = observed_band[index] / reference[index]

    rho_s = sediment.rho_s
    width = float(params["W"])
    result = pd.DataFrame(index=df_psd.index)
    result["river_depth_m"] = depths
    result["observed_band_psd_linear"] = observed_band
    result["observed_band_psd_db"] = 10.0 * np.log10(
        np.maximum(observed_band, EPS_PSD)
    )
    if model_type == "Luong2026":
        result["reference_psd_linear"] = reference
    else:
        result["reference_psd_per_unit_qb"] = reference
    for name, values in diagnostics.items():
        result[name] = values
    result["qb_vol_m2_s"] = qb
    result["qb_mass_kg_m_s"] = qb * rho_s
    result["Qb_mass_kg_s"] = qb * rho_s * width
    result["model_name"] = model_type
    return result


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
  "config_version": "2.3.0",
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
    "display_fmin_hz": 2.0,
    "display_fmax_hz": 30.0
  },
  "inversion": {
    "_说明": "2-20 Hz为当前OBS工程频带，不是ADLP论文通用频带；应依据场地噪声源和传播试验校准。",
    "model_type": "Luong2024",
    "f_inv_min_hz": 2.0,
    "f_inv_max_hz": 20.0,
    "n_frequency_bins": 60
  },
  "site": {
    "_说明": "以下保留附件中的现场占位值。slope是无量纲坡降S=tan(theta)，算法中不再二次取tan。",
    "channel_width_m": 132.5,
    "slope": 0.0069814304,
    "source_distance_m": 500.0,
    "bed_elevation_m": 710.6
  },
  "sediment": {
    "_说明": "D50、粒径分布宽度与最大粒径必须替换为现场筛分或图像识别结果。",
    "D50_m": 0.02,
    "D_min_m": null,
    "D_max_m": 0.08,
    "grain_log_std": 0.45,
    "grain_points": 80,
    "rho_s_kg_m3": 2550.0,
    "rho_f_kg_m3": 1000.0,
    "tau_c50": 0.045,
    "clip_tau_c": true
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
    "default_selection_hours": 24.0
  },
  "performance": {
    "_说明": "PSD按时间块分批计算以限制峰值内存；波形仅在绘图时进行min-max降采样，数值计算始终使用完整数据。",
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


def read_waveform_headers(
    file_paths: Iterable[str | Path],
) -> list[WaveformHeaderRecord]:
    """
    仅读取地震文件头，不加载波形数据。

    ObsPy 的 ``headonly=True`` 对 MiniSEED/MSEED、BHZ、MSD、SAC 等
    常见格式只解析元数据，适合数月文件的快速覆盖范围检查。
    """

    try:
        from obspy import read
    except ImportError as exc:
        raise RuntimeError("读取地震文件头需要安装 ObsPy。") from exc

    records: list[WaveformHeaderRecord] = []
    errors: list[str] = []
    for raw_path in file_paths:
        path = str(Path(raw_path))
        try:
            stream = read(path, headonly=True)
        except Exception as exc:
            errors.append(f"{Path(path).name}: {exc}")
            continue
        for trace in stream:
            stats = trace.stats
            records.append(
                WaveformHeaderRecord(
                    path=path,
                    trace_id=str(trace.id),
                    channel=str(stats.channel),
                    start_utc=_as_utc_timestamp(stats.starttime.datetime),
                    end_utc=_as_utc_timestamp(stats.endtime.datetime),
                    sampling_rate_hz=float(stats.sampling_rate),
                    npts=int(stats.npts),
                    format_name=str(getattr(stats, "_format", "") or "AUTO"),
                )
            )
    if errors:
        details = "\n".join(errors[:12])
        suffix = "\n…" if len(errors) > 12 else ""
        raise RuntimeError(
            f"{len(errors)} 个文件无法读取头信息：\n{details}{suffix}"
        )
    if not records:
        raise RuntimeError("所选文件中没有可识别的地震波形头信息。")
    return sorted(
        records,
        key=lambda item: (item.trace_id, item.start_utc, item.path),
    )


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
    只读取与目标时段相交的同一垂向通道数据。

    对 MiniSEED，ObsPy 会按 ``starttime/endtime`` 从文件中截取目标记录，
    不会先把数月的完整样本全部载入内存。
    """

    try:
        from obspy import Stream, UTCDateTime, read
    except ImportError as exc:
        raise RuntimeError("读取地震波形需要安装 ObsPy。") from exc

    selected = records_overlapping_window(records, start_utc, end_utc)
    if not selected:
        raise RuntimeError("所选目标时段与已加载文件没有交集。")

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
    paths = sorted(
        {
            record.path
            for record in selected
            if record.trace_id == trace_id
        }
    )

    start = UTCDateTime(_as_utc_timestamp(start_utc).to_pydatetime())
    end = UTCDateTime(_as_utc_timestamp(end_utc).to_pydatetime())
    stream = Stream()
    for path in paths:
        stream += read(path, starttime=start, endtime=end)
    stream = stream.select(id=trace_id)
    stream.trim(start, end)
    if not stream:
        raise RuntimeError("目标时段未读取到有效样本。")
    return stream, paths, trace_id

# =============================================================================
# PyQt6 图形界面
# =============================================================================

# -*- coding: utf-8 -*-
"""
===============================================================================
OBS Analyzer v2.3.0 - 床载推移质地震方法反演系统
-------------------------------------------------------------------------------
基于文献与开源算法库:
1. Tsai, V.C. et al., 2012. A physical model for seismic noise generation from
   sediment transport in rivers. Geophysical Research Letters, 39(2).
2. Luong, L. et al., 2024. Seismic modeling of bedload transport in a gravel-bed
   alluvial channel. Journal of Geophysical Research: Earth Surface, 129(9).
3. GitHub 仓库: https://github.com/locluong09/seismic-based-bedload-transport-models

核心功能模块:
- 1. 地震波形导入、去趋势、去均值与仪器响应校正 (Counts -> m/s)
- 2. 水位/水深数据导入、UTC 时区统一与时间轴对齐
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
    QHeaderView, QSizePolicy, QAbstractSpinBox, QAbstractItemView, QFrame
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
                ("水力基准", ["Rh≈H, S, D50", "τ*=RhS/(R·D50)"]),
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
        bg = "#111827" if self.is_dark else "#ffffff"
        fg = "#d7dee8" if self.is_dark else "#263445"
        muted = "#94a3b8" if self.is_dark else "#64748b"
        accent = "#2f8fbf" if self.is_dark else "#256b91"
        titles = {
            "Tsai2012": "TSAI 2012  ·  SINGLE-MODE PHYSICAL MODEL",
            "Luong2024": "LUONG 2024  ·  MULTIMODE PHYSICAL MODEL",
            "Luong2026": "LUONG 2026  ·  SEISMIC–HYDRAULIC MODEL",
        }
        formulas = {
            "Tsai2012": [
                (
                    r"$P(f)=\int p(D)\,\dot N(D)\,"
                    r"\mathcal{K}_{T}(f,D)\,\mathrm{d}D$",
                    "(1)",
                ),
                (
                    r"$\mathcal{K}_{T}="
                    r"\frac{\pi^{2}f^{3}m^{2}w_i^{2}}"
                    r"{\rho_s^{2}v_c^{3}v_u^{2}}\,\chi(\beta)$",
                    "(2)",
                ),
                (
                    r"$q_b=\underset{f\in\mathcal{B}}{\operatorname{median}}"
                    r"\left[\frac{P_{\rm obs}(f)}{P_{q_b=1}(f)}\right]$",
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
                    r"$q_b=\underset{f\in\mathcal{B}}{\operatorname{median}}"
                    r"\left[\frac{P_{\rm obs}(f)}{P_{q_b=1}(f)}\right]$",
                    "(3)",
                ),
            ],
            "Luong2026": [
                (
                    r"$q_b^{*}=k\left(\frac{P}{P_{\rm re}}\right)^m"
                    r"\left(\tau^{*}-\tau_{\rm cr}^{*}\right)^n$",
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
        self.figure.clear()
        self.figure.patch.set_facecolor(bg)
        axis = self.figure.add_subplot(111)
        axis.set_facecolor(bg)
        axis.axis("off")
        axis.plot(
            [0.025, 0.975], [0.89, 0.89],
            color=accent, linewidth=1.1, transform=axis.transAxes,
        )
        axis.text(
            0.03, 0.93, titles[self.model_key],
            ha="left", va="bottom", color=muted,
            fontsize=8.5, fontweight="semibold",
            transform=axis.transAxes,
        )
        y_positions = (0.70, 0.43, 0.16)
        for index, (formula, tag) in enumerate(formulas[self.model_key]):
            axis.text(
                0.05,
                y_positions[index],
                formula,
                ha="left",
                va="center",
                color=fg,
                fontsize=10.8,
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
    """对单一通道连续波形去趋势、去均值并按指定 PAZ 转为速度。"""

    st = st_raw.copy()
    try:
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        pass
    
    st.detrend("linear")
    st.detrend("demean")
    
    if paz is not None and HAS_OBSPY:
        fs = st[0].stats.sampling_rate
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
QMainWindow { background-color: #0f172a; color: #f8fafc; }
QWidget { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 10pt; color: #f8fafc; }
QMenuBar { background-color: #1e293b; color: #f8fafc; border-bottom: 1px solid #334155; }
QMenuBar::item:selected { background-color: #334155; border-radius: 4px; }
QMenu { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; }
QMenu::item:selected { background-color: #0284c7; color: #ffffff; }

QScrollArea, QScrollArea > QWidget > QWidget { background-color: #1e293b; border: none; }

QTabWidget::pane { border: 1px solid #334155; background-color: #1e293b; border-radius: 8px; }
QTabBar::tab { background-color: #0f172a; color: #94a3b8; padding: 8px 18px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; }
QTabBar::tab:selected { background-color: #1e293b; color: #38bdf8; border-bottom: 2px solid #38bdf8; }

QGroupBox { font-weight: bold; border: 1px solid #334155; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: #1e293b; color: #38bdf8; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; color: #38bdf8; }

QPushButton { background-color: #334155; color: #f8fafc; border: 1px solid #475569; padding: 7px 15px; border-radius: 6px; font-weight: bold; }
QPushButton:hover { background-color: #0284c7; color: #ffffff; border-color: #0284c7; }

QLineEdit, QDoubleSpinBox, QSpinBox, QDateTimeEdit, QComboBox { background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 6px; min-height: 24px; color: #f8fafc; selection-background-color: #0284c7; }
QDoubleSpinBox, QSpinBox, QDateTimeEdit { padding-right: 30px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 26px; background-color: #334155; border-left: 1px solid #475569; }
QAbstractSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 5px; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 5px; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background-color: #0284c7; }
QComboBox QAbstractItemView { background-color: #1e293b; color: #f8fafc; selection-background-color: #0284c7; border: 1px solid #334155; }

QCalendarWidget { background-color: #111827; color: #f8fafc; border: 1px solid #475569; }
QCalendarWidget QWidget#qt_calendar_navigationbar { background-color: #1e293b; border-bottom: 1px solid #475569; }
QCalendarWidget QToolButton { background-color: #1e293b; color: #f8fafc; border: none; min-height: 26px; padding: 3px 8px; }
QCalendarWidget QToolButton:hover { background-color: #334155; color: #ffffff; }
QCalendarWidget QSpinBox { background-color: #0f172a; color: #f8fafc; selection-background-color: #0284c7; }
QCalendarWidget QAbstractItemView:enabled { background-color: #111827; color: #e2e8f0; selection-background-color: #256b91; selection-color: #ffffff; outline: 0; }
QCalendarWidget QAbstractItemView:disabled { color: #64748b; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: #64748b; min-height: 38px; border-radius: 5px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed { background: #38bdf8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { height: 0; background: none; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal { background: #64748b; min-width: 38px; border-radius: 5px; }
QScrollBar::handle:horizontal:hover, QScrollBar::handle:horizontal:pressed { background: #38bdf8; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { width: 0; background: none; }
QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #2f8fbf; border-radius: 3px; }
QSlider::handle:horizontal { background: #e2e8f0; border: 2px solid #2f8fbf; width: 16px; height: 16px; margin: -6px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { background: #ffffff; border-color: #38bdf8; }

QTableWidget { background-color: #0f172a; color: #f8fafc; gridline-color: #334155; border-radius: 6px; alternate-background-color: #182238; }
QTableWidget::item { color: #f8fafc; background-color: #0f172a; }
QTableWidget::item:selected { background-color: #0284c7; color: #ffffff; }
QTableWidget QTableCornerButton::section { background-color: #1e293b; border: 1px solid #334155; }
QHeaderView::section { background-color: #1e293b; color: #38bdf8; font-weight: bold; border: 1px solid #334155; padding: 6px; }

QStatusBar { background-color: #0f172a; color: #94a3b8; }
QProgressBar { border: 1px solid #334155; border-radius: 6px; text-align: center; background-color: #0f172a; color: #f8fafc; }
QProgressBar::chunk { background-color: #38bdf8; border-radius: 6px; }

QTextEdit, QPlainTextEdit { background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; font-family: 'Microsoft YaHei', 'Segoe UI'; selection-background-color: #0284c7; }
QCheckBox { color: #f8fafc; spacing: 8px; }

QToolButton { background-color: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 4px; margin: 2px; padding: 3px; }
QToolButton:hover { background-color: #0284c7; color: #ffffff; }

QMessageBox { background-color: #1e293b; color: #f8fafc; }
QMessageBox QLabel { color: #f8fafc; font-size: 10pt; }
QMessageBox QPushButton { background-color: #334155; color: #f8fafc; min-width: 70px; border: 1px solid #475569; border-radius: 4px; padding: 5px; }
"""

LIGHT_THEME_QSS = """
QMainWindow { background-color: #f8fafc; color: #0f172a; }
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

    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = list(file_paths)

    def run(self):
        try:
            self.progress_sig.emit(10, "正在快速扫描波形文件头...")
            records = read_waveform_headers(self.file_paths)
            self.progress_sig.emit(
                100, f"文件头扫描完成：{len(records)} 条Trace记录"
            )
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

            long_gaps = [
                gap
                for gap in st_raw.get_gaps()
                if float(gap[6]) > self.maximum_gap_seconds
            ]
            if long_gaps:
                raise RuntimeError(
                    f"检测到 {len(long_gaps)} 个超过 "
                    f"{self.maximum_gap_seconds:g} s 的数据缺口。"
                    "为避免把长缺口伪造成信号，程序已停止；请先分段处理。"
                )

            self.progress_sig.emit(50, "正在合并目标时段并插值短缺口...")
            st_raw.merge(method=1, fill_value="interpolate")
            
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
        bg_color = '#1e293b' if self.is_dark_theme else '#ffffff'
        axis_bg = '#0f172a' if self.is_dark_theme else '#f8fafc'
        text_color = '#f8fafc' if self.is_dark_theme else '#0f172a'
        title_color = '#38bdf8' if self.is_dark_theme else '#0284c7'
        grid_color = '#334155' if self.is_dark_theme else '#cbd5e1'
        spine_color = '#475569' if self.is_dark_theme else '#94a3b8'
        
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
        self.spin_psd_fmin = QDoubleSpinBox(); self.spin_psd_fmin.setValue(self.config["psd"]["display_fmin_hz"])
        pl.addWidget(self.spin_psd_fmin, 1, 1)

        pl.addWidget(QLabel("显示 Fmax (Hz):"), 2, 0)
        self.spin_psd_fmax = QDoubleSpinBox(); self.spin_psd_fmax.setValue(self.config["psd"]["display_fmax_hz"])
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
            "clip_tau_c": sediment_cfg["clip_tau_c"],
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
# 6. 程序入口
# =============================================================================

def main():
    """创建 Qt 应用并直接启动图形界面。"""

    app = QApplication(sys.argv)
    try:
        gui = OBSAnalyzerGUI()
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
