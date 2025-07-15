'''
Eurocode 2 Table of concrete design properties
Web site: https://eurocodeapplied.com/design/en1992/concrete-design-properties
Design aid - Table of concrete design properties including strength properties (fck, fcd, fctm, fctd) elastic deformation properties (Ecm), minimum longitudinal reinforcement against brittle failure, and minimum shear reinforcement
According to: Eurocode 2: Design of Concrete Structures, EN 1992-1-1:2004+AC2:2010, 
'''

import pandas as pd
import matplotlib.pyplot as plt

# 构建 DataFrame
data = [
    ["f_{ck} (MPa)", "Characteristic cylinder compressive strength", 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90],
    ["f_{ck,cube} (MPa)", "Characteristic cube compressive strength", 15, 20, 25, 30, 37, 45, 50, 55, 60, 67, 75, 85, 95, 105],
    ["f_{cm} (MPa)", "Mean cylinder compressive strength", 20, 24, 28, 33, 38, 43, 48, 53, 58, 63, 68, 78, 88, 98],
    ["f_{ctm} (MPa)", "Mean tensile strength", 1.57, 1.90, 2.21, 2.56, 2.90, 3.21, 3.51, 3.80, 4.07, 4.21, 4.35, 4.61, 4.84, 5.04],
    ["f_{cd} (MPa)\n(for alpha_{cc}=1.00)", "Design compressive strength\n(for alpha_{cc}=1.00)", 8.00, 10.67, 13.33, 16.67, 20.00, 23.33, 26.67, 30.00, 33.33, 36.67, 40.00, 46.67, 53.33, 60.00],
    ["f_{cd} (MPa)\n(for alpha_{cc}=0.85)", "Design compressive strength\n(for alpha_{cc}=0.85)", 6.80, 9.07, 11.33, 14.17, 17.00, 19.83, 22.67, 25.50, 28.33, 31.17, 34.00, 39.67, 45.33, 51.00],
    ["f_{ctd} (MPa)\n(for alpha_{ct}=1.00)", "Design tensile strength\n(for alpha_{ct}=1.00)", 0.73, 0.89, 1.03, 1.20, 1.35, 1.50, 1.64, 1.77, 1.90, 1.97, 2.03, 2.15, 2.26, 2.35],
    ["E_{cm} (GPa)", "Modulus of elasticity", 27,29,30,31,33,34,35,36,37,38,39,41,42,44],
    ["rho_{min} (%)", "Minimum longitudinal tension reinforcement ratio", 0.130, 0.130, 0.130, 0.133, 0.151, 0.167, 0.182, 0.197, 0.212, 0.219, 0.226, 0.240, 0.252, 0.262],
    ["rho_{w,min} (%)", "Minimum shear reinforcement ratio", 0.055, 0.064, 0.072, 0.080, 0.088, 0.095, 0.101, 0.113, 0.119, 0.124, 0.134, 0.143, 0.152, 0.152]
]

columns = [
    "Symbol",
    "Description",
    "C12/15",
    "C16/20",
    "C20/25",
    "C25/30",
    "C30/37",
    "C35/45",
    "C40/50",
    "C45/55",
    "C50/60",
    "C55/67",
    "C60/75",
    "C70/85",
    "C80/95",
    "C90/105"
]

df = pd.DataFrame(data, columns=columns)

# integer location of the data: iloc
row1 = df.iloc[0, 2:]  # 第 1 行的数值
row5 = df.iloc[7, 2:]  # 第 8 行的数值

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.plot(row1, row5, '-o', label=r"$f_{ck} (MPa)$ vs $E_{cm} (GPa)$")

plt.xlabel(r"$f_{ck} (MPa)$")
plt.ylabel(r"$E_{cm} (GPa)$")
plt.legend()
plt.show()
