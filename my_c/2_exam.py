import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 加载数据
df = pd.read_csv('Wimbledon_featured_matches_last.csv')

# 皮尔逊相关性检验
pearson_corr_p1, p_value_p1 = pearsonr(df['p1_perform'], df['p1_momt'])
pearson_corr_p2, p_value_p2 = pearsonr(df['p2_perform'], df['p2_momt'])

print(f"Player 1 Perform vs. P1 Momt Pearson Correlation: {pearson_corr_p1}, P-value: {p_value_p1}")
print(f"Player 2 Perform vs. P2 Momt Pearson Correlation: {pearson_corr_p2}, P-value: {p_value_p2}")

# 斯皮尔曼相关性检验
spearman_corr_p1, p_value_spearman_p1 = spearmanr(df['p1_perform'], df['p1_momt'])
spearman_corr_p2, p_value_spearman_p2 = spearmanr(df['p2_perform'], df['p2_momt'])

print(f"Player 1 Perform vs. P1 Momt Spearman Correlation: {spearman_corr_p1}, P-value: {p_value_spearman_p1}")
print(f"Player 2 Perform vs. P2 Momt Spearman Correlation: {spearman_corr_p2}, P-value: {p_value_spearman_p2}")
