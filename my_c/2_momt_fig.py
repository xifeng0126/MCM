import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 加载数据
df = pd.read_csv('Wimbledon_featured_matches_last.csv')

# 选取数据
y1 = df['p1_momt'].iloc[:300]
y2 = df['p2_momt'].iloc[:300]

# 使用Seaborn改善美观
sns.set(style="whitegrid")

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制线条并添加标记
plt.plot(df['point_no'].iloc[:300], y1, label='Player 1', marker='o', linestyle='-', color='#FFA07A', markersize=5)
plt.plot(df['point_no'].iloc[:300], y2, label='Player 2', marker='o', linestyle='-', color='#20B2AA', markersize=5)

# 添加标签和标题
plt.xlabel('Point Number', font=Path('palatino.ttf'),fontsize=12)
plt.ylabel('Momentum',font=Path('palatino.ttf'), fontsize=12)
plt.title('Player 1 vs. Player 2 Momentum', font=Path('palatino.ttf'), fontsize=14)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 添加图例
plt.legend()

plt.savefig('1_80.pdf', format='pdf')
# 显示图形
plt.show()