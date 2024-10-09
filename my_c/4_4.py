import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# 加载数据集
df2 = pd.read_csv('2022-wimbledon-1101.csv')

# 计算误差
df2['error'] = df2['momentum_shift_pred'] - df2['momentum_shift']

# 设置视觉样式
sns.set_style("whitegrid")

# 创建散点图
plt.figure(figsize=(12, 8))
colors = df2['error'].apply(lambda x: 'red' if x < 0 else 'green')  # 根据误差的正负设置颜色
size = 50  # 设置点的大小
plt.scatter(df2.index, df2['error'], c=colors, s=size, alpha=0.6, edgecolor='w', linewidth=0.5)

# 自定义图表
plt.title('Prediction Error for Each Data Point', font=Path('palatino.ttf'), fontsize=18)
plt.xlabel('Index',font=Path('palatino.ttf'), fontsize=16)
plt.ylabel('Error (Predicted - Actual)', font=Path('palatino.ttf'), fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Highlighting the points with the largest errors
top_errors = df2['error'].abs().nlargest(5).index
for index in top_errors:
    plt.scatter(index, df2.loc[index, 'error'], s=120, color='orange', edgecolor='white', linewidth=0.8)  # Highlight with larger points
    plt.text(index, df2.loc[index, 'error'], f'  Index {index}\n  Error {df2.loc[index, "error"]:.2f}', fontsize=12)

plt.savefig('error.pdf', format='pdf')
# Display the plot
plt.show()

