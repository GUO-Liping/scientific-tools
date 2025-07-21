import pandas as pd

# 读取数据
df = pd.read_csv("tail_velocity.csv", header=None)

# 初始化参数
data_rows = 1502
result = pd.DataFrame()
i = 0
col_idx = 1

# 首块：8 行表头 + 1502 行数据
i1 = 7
first_block = df.iloc[i1+1:i1+1+data_rows, [1, 2]].reset_index(drop=True)
first_block.columns = ['time', f'Velocity_{col_idx}']
result['time'] = first_block['time']
result[f'Velocity_{col_idx}'] = first_block[f'Velocity_{col_idx}']
i = i1+1 + data_rows
col_idx += 1

# 后续块：每块 4 行表头 + 1502 行数据
i2 = 4
while i + i2 + data_rows <= len(df):
    block = df.iloc[i+i2 : i+i2+data_rows, [1, 2]].reset_index(drop=True)
    result[f'Velocity_{col_idx}'] = block[2]
    i += i2 + data_rows
    col_idx += 1

# 保存结果
result.to_csv("reshaped_tail_velocity.csv", index=False)
