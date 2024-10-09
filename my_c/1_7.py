import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 使用Seaborn改善美观
sns.set(style="whitegrid")

# 定义异常区间的开始和结束点
start, end = 142, 156

# 假设您已经加载了数据到DataFrame df中
# df = pd.read_csv('Wimbledon_featured_matches.csv')
df = pd.read_csv('back.csv')
# 提取p1_score和p2_score列
p1_score = df['p1_score'].iloc[:300]
p2_score = df['p2_score'].iloc[:300]

# 创建图表
plt.figure(figsize=(12, 6))

# 使用Seaborn的颜色
colors = sns.color_palette('husl', 2)  # 选用HUSL颜色系统中的颜色

# 绘制阶梯图展示分数变化
plt.step(range(len(p1_score)), p1_score, label='Player 1 Score', where='post', linewidth=2, alpha=0.8, color=colors[0])
plt.step(range(len(p2_score)), p2_score, label='Player 2 Score', where='post', linewidth=2, alpha=0.8, color=colors[1], linestyle='--')

# 绘制异常区间的线
plt.plot(range(start, end), p1_score[start:end], color='red', linewidth=2, label='Anomaly - Player 1')
plt.plot(range(start, end), p2_score[start:end], color='darkorange', linewidth=2, label='Anomaly - Player 2')

# 添加标题和轴标签
plt.title('Player 1 vs Player 2 Score Changes', font=Path('palatino.ttf'), fontsize=14, fontweight='bold')
plt.xlabel('Point Number',font=Path('palatino.ttf'), fontsize=12)
plt.ylabel('Score', font=Path('palatino.ttf'), fontsize=12)
plt.yticks([0, 15, 30, 40, 50])  # 调整为网球比赛的标准分数

# 添加图例
plt.legend()

# 优化轴刻度和网格线样式
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 调整整体布局以防止标签被截断
plt.tight_layout()

# 保存图表
plt.savefig('1_players_points.pdf', format='pdf')

# 显示图表
plt.show()
