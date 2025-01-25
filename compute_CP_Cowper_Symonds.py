# python coding
# 根据文献中对国标中常见钢材的试验结果计算其应变率效应
# 具体来说，计算HPB235, HRB335, HRB400, HRB500钢材的C值与P值

import numpy as np
from scipy.stats import linregress

def calculate_cowper_symonds_parameters(strain_rates, stresses, static_stress):
    # Calculate the logarithmic values needed for linear regression
    log_strain_rate = np.log(strain_rates)
    log_stress_ratio = np.log((stresses / static_stress) - 1)

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(log_strain_rate, log_stress_ratio)

    # Calculate P and C
    P = 1 / slope
    C = np.exp(-P * intercept)
    return C, P

def verify_model(strain_rates, stresses, static_stress, C, P):
    calculated_stresses = static_stress * (1 + (strain_rates / C) ** (1 / P))
    print("\nVerification of calculated stresses:")
    for rate, exp_stress, calc_stress in zip(strain_rates, stresses, calculated_stresses):
        print(f"Strain rate: {rate:.2f}/s, Experimental stress: {exp_stress:.2f} MPa, Calculated stress: {calc_stress:.2f} MPa")

# Experimental data
materials = {
    "HPB235": {
        "strain_rates": np.array([6.5, 21.0, 74.9]),
        "yield_stresses": np.array([420.8, 436.0, 464.9]),
        "static_stress": 329,
        "ultimate_strain": np.array([0.153, 0.197, 0.201, 0.204])
    },
    "HRB335": {
        "strain_rates": np.array([2.9, 8.4, 38.1]),
        "yield_stresses": np.array([438.3, 461.5, 524.0]),
        "static_stress": 386.7,
        "ultimate_strain": np.array([0.153, 0.126, 0.101, 0.110])
    },
    "HRB400": {
        "strain_rates": np.array([3.0, 9.3, 40.5]),
        "yield_stresses": np.array([483.5, 503.3, 521.2]),
        "static_stress": 404.0,
         "ultimate_strain": np.array([0.156, 0.118, 0.130, 0.115])
   },
    "HRB500": {
        "strain_rates": np.array([5.2, 26.6, 54.2]),
        "yield_stresses": np.array([658.3, 694.6, 719.2]),
        "static_stress": 582.4,
        "ultimate_strain": np.array([0.095, 0.098, 0.112, 0.134])
    }
}

# Process each material
for material, data in materials.items():
    print(f"\nMaterial: {material}")
    strain_rates = data["strain_rates"]
    yield_stresses = data["yield_stresses"]
    static_stress = data["static_stress"]

    # Calculate C and P
    C, P = calculate_cowper_symonds_parameters(strain_rates, yield_stresses, static_stress)

    # Print results
    print(f"Strain-rate constant (C): {C:.3f}")
    print(f"Strain-rate sensitivity exponent (P): {P:.3f}")

    # Verify the model
    verify_model(strain_rates, yield_stresses, static_stress, C, P)