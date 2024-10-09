import pandas as pd
from scipy.stats import norm

def runs_test(binary_sequence):
    n1 = binary_sequence.count(1)
    n2 = binary_sequence.count(0)

    # 计算游程的数量
    runs = 1
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i - 1]:
            runs += 1

    # 计算期望的游程数量和方差
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    variance = (expected_runs - 1) * (expected_runs - 2) / (n1 + n2 - 1)

    # 计算Z得分
    z = (runs - expected_runs) / (variance ** 0.5)

    # 确定显著性
    p_value = 2 * (1 - norm.cdf(abs(z)))  # 双尾检验

    return runs, expected_runs, z, p_value
# 加载数据
df = pd.read_csv('Wimbledon_featured_matches_last.csv')

# 生成二进制序列
binary_sequence_p1 = [1 if perform > momt else 0 for perform, momt in zip(df['p1_perform'], df['p1_momt'])]
binary_sequence_p2 = [1 if perform > momt else 0 for perform, momt in zip(df['p2_perform'], df['p2_momt'])]

# 进行游程检验
runs_result_p1 = runs_test(binary_sequence_p1)
runs_result_p2 = runs_test(binary_sequence_p2)

print(f"p1_perform vs. p1_momt Runs Test: Z = {runs_result_p1[0]}, P-value = {runs_result_p1[1]}")
print(f"p2_perform vs. p2_momt Runs Test: Z = {runs_result_p2[0]}, P-value = {runs_result_p2[1]}")
