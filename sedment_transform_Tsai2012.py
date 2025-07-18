import numpy as np
import pandas as pd


def model_bedload(
    gsd=None,
    d_s=None,
    s_s=None,
    r_s=None,
    q_s=None,
    h_w=None,
    w_w=None,
    a_w=None,
    f=(1, 100),
    r_0=None,
    f_0=None,
    q_0=None,
    e_0=None,
    v_0=None,
    x_0=None,
    n_0=None,
    n_c=None,
    res=100,
    adjust=True,
    **kwargs
):
    """
    Model the seismic spectrum due to bedload transport in rivers.

    This function calculates a seismic spectrum as predicted
    by the model of Tsai et al. (2012)
    for river bedload transport. It's based on the R implementation
    by Sophie Lagarde and Michael Dietze.

    Parameters:
    -----------
    gsd : array-like, optional
        Grain-size distribution function.
        Should be provided as a 2D array with two columns:
        grain-size class (in m) and weight/volume percentage per class.
    d_s : float, optional
        Mean sediment grain diameter (m). Alternative to gsd.
    s_s : float, optional
        Standard deviation of sediment grain diameter (m). Alternative to gsd.
    r_s : float
        Specific sediment density (kg/m^3)
    q_s : float
        Unit sediment flux (m^2/s)
    h_w : float
        Fluid flow depth (m)
    w_w : float
        Fluid flow width (m)
    a_w : float
        Fluid flow inclination angle (radians)
    f : tuple of float, optional
        Frequency range to be modelled (Hz). Default is (1, 100).
    r_0 : float
        Distance of seismic station to source (m)
    f_0 : float
        Reference frequency (Hz)
    q_0 : float
        Ground quality factor at f_0
    e_0 : float
        Exponent characterizing quality factor increase
        with frequency (dimensionless)
    v_0 : float
        Phase speed of the Rayleigh wave at f_0 (m/s)
    x_0 : float
        Exponent of the power law variation of Rayleigh wave
        velocities with frequency
    n_0 : float or array-like
        Green's function displacement amplitude coefficients
    n_c : float, optional
        Option to include single particle hops coherent in time
    res : int, optional
        Output resolution, i.e., length of the spectrum vector.
        Default is 100.
    adjust : bool, optional
        Option to adjust PSD for wide grain-size distributions.
        Default is True.
    **kwargs : dict, optional
        Additional parameters:
        - g : Gravitational acceleration (m/s^2). Default is 9.81.
        - r_w : Fluid specific density (kg/m^3). Default is 1000.
        - k_s : Roughness length (m). Default is 3 * d_s.
        - log_lim : Limits of grain-size distribution function template.
            Default is (0.0001, 100).
        - log_length : Length of grain-size distribution function template.
            Default is 10000.
        - nu : Kinematic viscosity of water at 18 degree  (m^2/s):. Default is 1.0533e-6.
        - power_d : Grain-size power exponent. Default is 3.
        - gamma : Gamma parameter, after Parker (1990). Default is 0.9.
        - s_c : Drag coefficient parameter. Default is 0.8.
        - s_p : Drag coefficient parameter. Default is 3.5.
        - c_1 : Inter-impact time scaling, after Sklar & Dietrich (2004).
            Default is 2/3.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with two columns:
        - 'frequency': The frequency vector (Hz)
        - 'power': The corresponding power spectral density

    Notes:
    ------
    When no user-defined grain-size distribution function is provided,
    the function calculates the raised cosine distribution function as
    defined in Tsai et al. (2012).

    The adjustment option is only relevant for wide grain-size distributions,
    i.e., s_s > 0.2.
    In such cases, the unadjusted version tends to underestimate seismic power.

    References:
    -----------
    Tsai, V. C., B. Minchew, M. P. Lamb, and J.-P. Ampuero (2012),
    A physical model for seismic noise generation from sediment transport
    in rivers, Geophys. Res. Lett., 39, L02404, doi:10.1029/2011GL050255.

    Examples:
    ---------
    >>> result = model_bedload(d_s=0.7, s_s=0.1, r_s=2650, q_s=0.001, h_w=4,
    ...                        w_w=50, a_w=0.005, f=(0.1, 20), r_0=600,
    ...                        f_0=1, q_0=20, e_0=0, v_0=1295,
    ...                        x_0=0.374, n_0=1, res=100)
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(result[:, 0], 10 * np.log10(result[:, 1]))
    >>> plt.xlabel('Frequency (Hz)')
    >>> plt.ylabel('Power Spectral Density (dB)')
    >>> plt.show()
    """
    # Default values for additional parameters
    g = kwargs.get("g", 9.81)
    r_w = kwargs.get("r_w", 1000)  
    k_s = kwargs.get("k_s", 3 * d_s) 
    nu = kwargs.get("nu", 1e-6)
    power_d = kwargs.get("power_d", 3)
    gamma = kwargs.get("gamma", 0.9)
    s_c = kwargs.get("s_c", 0.8)
    s_p = kwargs.get("s_p", 3.5)
    c_1 = kwargs.get("c_1", 2 / 3)

    if gsd is None:
        x_log = np.logspace(np.log10(0.0001), np.log10(10), num=1000)
        # s为颗粒粒径分布方差
        s = s_s / np.sqrt(1 / 3 - 2 / np.pi**2)
        # p_s是沉积物粒径的概率密度分布函数，用于描述不同粒径颗粒np.log(x_log)在整体粒径分布[np.log(d_s) - s, np.log(d_s) + s]中所占的比例。
        # p_s颗粒粒径分布函数——升余弦分布Raised cosine distribution---------------------------------eq.15 
        p_s = (1 / (2*s) * (1 + np.cos(np.pi * (np.log(x_log)-np.log(d_s)) / s))) / x_log
        p_s[(np.log(x_log) - np.log(d_s)) > s] = 0
        p_s[(np.log(x_log) - np.log(d_s)) < -s] = 0

        x_log = x_log[p_s > 0]
        p_s = p_s[p_s > 0]
        if not adjust:
            p_s = p_s / np.sum(p_s)
    else:
        d_min = 10 ** np.ceil(np.log10(np.min(gsd[:, 0]) / 10))
        d_max = 10 ** np.ceil(np.log10(np.max(gsd[:, 0])))
        x_log = np.logspace(np.log10(d_min), np.log10(d_max), num=10000)
        p_s_gsd = np.interp(x_log, gsd[:, 0], gsd[:, 1], left=0, right=0)
        mask = ~np.isnan(p_s_gsd)
        x_log = x_log[mask]
        p_s_gsd = p_s_gsd[mask]
        f_density = np.sum(p_s_gsd * np.diff(x_log, prepend=x_log[0]))
        p_s = p_s_gsd / f_density
        d_s = x_log[np.argmin(np.abs(np.cumsum(p_s) - 0.5))]

    # 颗粒与流体的相对比重,值越大,沉积物在水中的下沉和运动阻力越明显
    r_b = (r_s - r_w) / r_w

    # 摩擦流速,反映了水流对河床表面产生的剪切作用--------------------------------------------------eq.15 
    u_s = np.sqrt(g * h_w * np.sin(a_w))

    # 流体表面的最大流速-------------------------------------------------------------------------eq.15 
    u_m = 8.1 * u_s * (h_w / k_s) ** (1 / 6)

    # 基于流体坡度的无量纲参数，反映了坡度和摩擦的关系
    chi = 0.407 * np.log(142 * np.tan(a_w))

    # 临界剪切应力比,用于描述坡度和剪切应力对沉积物运动起始条件的影响
    t_s_c50 = np.exp(
        2.59e-2*chi**4 + 8.94e-2*chi**3 + 0.142*chi**2 + 0.41*chi - 3.14
    )
    # 生成频率
    f_i = np.linspace(f[0], f[1], res)

    # 相速度------------------------------------------------------------------------------------eq.5
    v_c = v_0 * (f_i / f_0) ** (-x_0)

    # 群速度------------------------------------------------------------------------------------eq.5
    v_u = v_c / (1 + x_0)

    # 无量纲参数beta----------------------------------------------------------------------------eq.8 / eq.9
    b = (2 * np.pi * r_0 * (1 + x_0) * f_i ** (1 + x_0 - e_0)) / (
        v_0 * q_0 * f_0 ** (x_0 - e_0)
    )
    x_b = 2 * np.log(1 + (1 / b)) * np.exp(-2 * b) + (1 - np.exp(-b)) * np.exp(
        -b
    ) * np.sqrt(2 * np.pi / b)

    # 计算沉积物的运动和冲击行为-------------------------------------------------------------------eq.6(Dietrich, 1982)
    s_x = np.log10((r_b * g * x_log**power_d) / nu**2)

    # 计算沉积物的运动和冲击行为-------------------------------------------------------------------eq.9(Dietrich, 1982)
    r_1 = (
        -3.76715
        + 1.92944 * s_x
        - 0.09815 * s_x**2
        - 0.00575 * s_x**3
        + 0.00056 * s_x**4
    )
    # 跃移概率分布的修正项，调整了阻力系数和粒径的相互影响--------------------------------------------eq.16(Dietrich, 1982)
    r_2 = (
        np.log10(1 - ((1 - s_c) / 0.85))
        - (1 - s_c) ** 2.3 * np.tanh(s_x - 4.6)
        + 0.3 * (0.5 - s_c) * (1 - s_c) ** 2 * (s_x - 4.6)
    )
    # 跃移粒径对阻力的修正项-----------------------------------------------------------------------eq.18(Dietrich, 1982)
    r_3 = (0.65 - ((s_c/2.83) * np.tanh(s_x-4.6))) ** (1 + ((3.5-s_p) / 2.5))

    # 跃移距离------------------------------------------------------------------------------------eq.19(Dietrich, 1982)
    w_1 = r_3 * 10 ** (r_2 + r_1)  
    # 颗粒在流体中的沉降速度-----------------------------------------------------------------------eq.5(Dietrich, 1982)
    w_2 = (r_b * g * nu * w_1) ** (1 / 3)

    # 阻力系数，表示流体对跃移颗粒的阻力影响---------------------------------------------------------eq.4(Dietrich, 1982)
    c_d = (4 / 3) * (r_b * g * x_log) / (w_2**2)

    # 颗粒的剪切应力比-----------------------------------------------------------------------------eq.15
    t_s = (u_s**2) / (r_b * g * x_log)

    #临界剪切应力比--------------------------------------------------------------------------------eq.15
    t_s_c = t_s_c50 * ((x_log / d_s) ** (-gamma))

    # 平均跃移高度，依赖于粒径和流体剪切应力条件------------------------------------------------------eq.15 
    h_b = 1.44 * x_log * (t_s / t_s_c) ** 0.5
    h_b[h_b > h_w] = h_w

    # 跃移颗粒的平均水平速度，与粒径和流体剪切条件相关------------------------------------------------eq.14 
    u_b = 1.56 * np.sqrt(r_b * g * x_log) * (t_s / t_s_c) ** 0.56
    u_b[u_b > u_m] = u_m

    # 单颗粒的体积，假设颗粒为球形
    v_p = (4 / 3) * np.pi * (x_log / 2) ** 3

    # 颗粒质量
    m = r_s * v_p

    # 无阻力终端速度-------------------------------------------------------------------------------eq.16 
    w_st = np.sqrt(4 * r_b * g * x_log / (3 * c_d))

    # 颗粒跃移高度相对于粒径和流体条件的无量纲比例----------------------------------------------------eq.16 
    h_b_2 = 3 * c_d * r_w * h_b / (2 * r_s * x_log * np.cos(a_w))

    # 跃移颗粒间的相对碰撞速度，依赖于跃移高度比例和终端速度-------------------------------------------eq.16 
    w_i = w_st * np.cos(a_w) * np.sqrt(1 - np.exp(-h_b_2))

    # 颗粒跃移的平均输移速度，结合了跃移高度比例和终端速度---------------------------------------------eq.17 
    w_s = (h_b_2 * w_st * np.cos(a_w)) / (
        2 * np.log(np.exp(h_b_2 / 2) + np.sqrt(np.exp(h_b_2) - 1))
    )

    # 功率谱密度-----------------------------------------------------------------------------------eq.7 / eq.13 
    def calculate_psd(f, x_b, v_c, v_u):
        if n_c is not None:
            z = np.exp(-1j * n_c * np.pi * f * h_b / (c_1 * w_s))
            f_t = (np.abs(1 + z) ** 2) / 2
            psd_raw = (
                (c_1 * w_w * q_s * w_s * np.pi**2 * f**3 * m**2 * w_i**2 * x_b)
                * f_t
                / (v_p * u_b * h_b * r_s**2 * v_c**3 * v_u**2)
            )
        else:
            psd_raw = (
                c_1 * w_w * q_s * w_s * np.pi**2 * f**3 * m**2 * w_i**2 * x_b
            ) / (v_p * u_b * h_b * r_s**2 * v_c**3 * v_u**2)

        if adjust:
            psd_f = np.sum(p_s * psd_raw * n_0**2 *
                           np.diff(x_log, prepend=x_log[0]))
        else:
            psd_f = np.sum(p_s * psd_raw * n_0**2)
        return psd_f

    z = np.array([calculate_psd(f, x, v, u)
                  for f, x, v, u in zip(f_i, x_b, v_c, v_u)])

    # Return the result as a pandas DataFrame
    return pd.DataFrame({"frequency": f_i, "power": z})


if __name__ == "__main__":
    # Set parameters
    d_s = 0.7
    s_s = 0.1
    r_s = 2650
    q_s = 0.001     # p5: qb = 10^-3 m2/s
    h_w = 4         # p5: H = 4 m
    w_w = 50        # p5: W = 50 m
    a_w = 0.005
    r_0 = 600       # p5: r0 = 600 m
    f_0 = 1         # p2: f0 = 1 Hz
    q_0 = 20        # p2: Q0 ≈ 20
    e_0 = 0         # p2: η = 0
    v_0 = 1295      # p2: vc0 = 1295 m/s
    x_0 = 0.374     # p2: x = 0.374
    n_0 = 1

    # Run the model
    result = model_bedload(
        d_s=d_s,
        s_s=s_s,
        r_s=r_s,
        q_s=q_s,
        h_w=h_w,
        w_w=w_w,
        a_w=a_w,
        f=(0.1, 20),  # 截断频率 0.1Hz至20Hz
        r_0=r_0,
        f_0=f_0,
        q_0=q_0,
        e_0=e_0,
        v_0=v_0,
        x_0=x_0,
        n_0=n_0,
        res=1000,
    )

    # Plot the result
    print(result.head())
    print("\nShape of the result:", result.shape)

    show_plot = True  # Set this to False to skip the plot
    if show_plot:
        import matplotlib.pyplot as plt

        plt.plot(result["frequency"], 10 * np.log10(result["power"]))

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB)")
        plt.title("Seismic Spectrum due to Bedload Transport")
        plt.show()
