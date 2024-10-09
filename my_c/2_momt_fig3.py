import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 加载数据
df = pd.read_csv('Wimbledon_featured_matches_last2.csv')

# 选取数据
y1 = df['p1_momt'].iloc[6952:]
y2 = df['p2_momt'].iloc[6952:]

# 对数据进行滑动平均处理以平滑曲线
y1_smooth = y1.rolling(window=5).mean()
y2_smooth = y2.rolling(window=5).mean()

# 使用Seaborn改善美观
sns.set(style="whitegrid")

# 设置图形大小
plt.figure(figsize=(12, 6))

# 绘制平滑曲线而非每个数据点
plt.plot(df['point_no'].iloc[6952:], y1_smooth, label='Player 1', linestyle='-', color='#FFA07A')
plt.plot(df['point_no'].iloc[6952:], y2_smooth, label='Player 2', linestyle='-', color='#20B2AA')

# 添加标签和标题
plt.xlabel('Point Number', font=Path('palatino.ttf'), fontsize=12)
plt.ylabel('Momentum', font=Path('palatino.ttf'), fontsize=12)
plt.title('Player 1 vs. Player 2 Momentum - Smoothed',font=Path('palatino.ttf'), fontsize=14)

# 调整y轴范围以更清晰地展示波动
plt.ylim(-2, 2)

# 添加图例
plt.legend()

plt.savefig('2_1701_momt_smooth.pdf', format='pdf')
# 显示图形
plt.show()
