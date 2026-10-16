# Excel数据批量按条件（峰值位置）裁剪 v1.2
from pathlib import Path
import pandas as pd

# 参数配置区域
raw_head = 6 # 前6行为表头
ROOT = Path.cwd()
OUT = ROOT / "excel裁剪结果"
OUT.mkdir(exist_ok=True)
sampling_rate = 1000  # 采样频率
before_peak = 2  # 峰值前2s
after_peak = 3   # 峰值后3s
fitter_colume = 7  # 按照第8列中的最大值位置进行定位，设立参考

# 主程序区
for file in ROOT.iterdir():
    if not file.is_file() or file.suffix.lower() not in {".xlsx", ".xls", ".xlsm"} or file.name.startswith("~$"):
        continue
    target = OUT / f"{file.stem}_cropped.xlsx"
    try:
        with pd.ExcelFile(file) as xls, pd.ExcelWriter(target, engine="openpyxl") as writer:
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, header=None)
                if len(df) > raw_head and df.shape[1] >= fitter_colume:
                    h = pd.to_numeric(df.iloc[raw_head:, fitter_colume], errors="coerce")
                    if h.notna().any():
                        peak = h.idxmax()
                        start, end = max(raw_head, peak - before_peak * sampling_rate), min(len(df), peak + after_peak*sampling_rate+1)
                        df = pd.concat([df.iloc[:raw_head], df.iloc[start:end]])
                    else:
                        print(f"H列无有效数值，保留原数据：{file.name} / {sheet}")
                else:
                    print(f"数据行不足或列数不足：{file.name} / {sheet}")
                df.to_excel(writer, sheet_name=sheet, index=False, header=False)
        print(f"完成：{file.name} -> {target.name}")
    except Exception as e:
        print(f"处理失败：{file.name}，原因：{e}")

print(f"处理完成，结果目录：{OUT}")