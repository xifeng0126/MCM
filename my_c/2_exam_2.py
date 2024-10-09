import pandas as pd
from scipy.stats import wilcoxon

# 加载数据
df = pd.read_csv('Wimbledon_featured_matches_last.csv')

# 计算均值
mean_p1_perform = df['p1_perform'].mean()
mean_p1_momt = df['p1_momt'].mean()

# 生成二元序列
binary_sequence = [(1 if p1_perform > mean_p1_perform else 0, 1 if p1_momt > mean_p1_momt else 0) for
                   p1_perform, p1_momt in zip(df['p2_perform'], df['p2_momt'])]

# 计算游程
runs, n1, n2 = 0, 0, 0
prev_value = None

for value in binary_sequence:
    if value[0] == value[1]:
        n1 += 1
    else:
        n2 += 1

    if prev_value is None or value != prev_value:
        runs += 1
        prev_value = value

# 进行游程检验
z = (runs - ((2 * n1 * n2) / (n1 + n2) + 1)) / (
            (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))) ** 0.5

print(f"Runs: {runs}, Expected Runs: {(2 * n1 * n2) / (n1 + n2) + 1}, Z-value: {z}")
