import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 使用Seaborn改善美观
sns.set(style="whitegrid")

# 假设您已经加载了数据到DataFrame df中
# df = pd.read_csv('your_data_file.csv')  # 如果您需要从文件加载数据
df = pd.read_csv('Wimbledon_featured_matches.csv')

# 提取p1_score和p2_score列
p1_score = df['p1_score'].iloc[6952:]
p2_score = df['p2_score'].iloc[6952:]

# 计算两位玩家的分数差距
score_diff = p1_score - p2_score

# 创建图表
plt.figure(figsize=(11, 8))

# 绘制分数差距的条形图
bars = plt.bar(range(len(score_diff)), score_diff, alpha=0.8, color=sns.color_palette("coolwarm", n_colors=len(score_diff)))

# 为正值和负值设置不同的颜色
for bar, value in zip(bars, score_diff):
    if value > 0:
        bar.set_color('seagreen')
    else:
        bar.set_color('crimson')

# 添加水平线以标示无差距的情况
plt.axhline(0, color='black', linewidth=1)

# 添加标题和轴标签
plt.title('Score Difference Between Player 1 and Player 2',font=Path('palatino.ttf'),fontsize=16)
plt.xlabel('Point Number', font=Path('palatino.ttf'),fontsize=14)
plt.ylabel('Score Difference',font=Path('palatino.ttf'), fontsize=14)

# 优化轴刻度
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 显示图表
plt.tight_layout()  # 调整整体布局以防止标签被截断
plt.savefig('1_1701_players_points_dif.pdf', format='pdf')
plt.show()