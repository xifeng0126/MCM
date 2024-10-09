from scipy.stats import pearsonr
import pandas as pd

df = pd.read_csv('Wimbledon_featured_matches.csv')
# 两个变量的数据
x = df['server']
y = df['point_victor']
# 计算皮尔逊相关系数和p-value
pearson_coefficient, p_value = pearsonr(x, y)

print("皮尔逊相关系数:", pearson_coefficient)
print("p-value:", p_value)
