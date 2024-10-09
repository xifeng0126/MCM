import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设定Seaborn样式
sns.set(style="whitegrid", palette="pastel")

# 假设您已经加载了数据到DataFrame df中
df = pd.read_csv('Wimbledon_featured_matches.csv')

df['p1_distance_sum'] = df.groupby(['match_id'])['p1_distance_run'].cumsum()
df['p2_distance_sum'] = df.groupby(['match_id'])['p2_distance_run'].cumsum()

# 按match_id分组并获取每组的最大值
grouped_max = df.groupby('match_id')[['p1_distance_sum', 'p2_distance_sum']].max()

# 重置索引，便于绘图
grouped_max = grouped_max.reset_index()

# 创建图表
plt.figure(figsize=(12, 6))

# 获取分组的数量，用于确定x轴的条形位置
n_groups = len(grouped_max)
index = range(n_groups)

# 绘制p1_distance_sum的最大值
bars1 = plt.bar(index, grouped_max['p1_distance_sum'], width=0.4, label='Player 1 Max Distance', align='center')

# 绘制p2_distance_sum的最大值，通过添加宽度的一半来调整位置
bars2 = plt.bar([i+0.4 for i in index], grouped_max['p2_distance_sum'], width=0.4, label='Player 2 Max Distance', align='center')

# 添加标题和轴标签
plt.title('Maximum Distance by Player in Each Match', font=Path('palatino.ttf'), fontsize=16)
plt.xlabel('Match ID (Last Four Digits)',font=Path('palatino.ttf'), fontsize=14)
plt.ylabel('Maximum Distance (meters)',font=Path('palatino.ttf'), fontsize=14)

# 设置x轴刻度标签，只保留后四位数字
x_labels = [str(match)[-4:] for match in grouped_max['match_id']]
plt.xticks([i+0.2 for i in index], x_labels, rotation=45, ha='right', fontsize=12)


# 添加图例
plt.legend(fontsize=12)

# 优化布局
plt.tight_layout()
plt.savefig('max_distance.pdf', format='pdf')
# 显示图表
plt.show()
