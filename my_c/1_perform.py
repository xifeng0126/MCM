import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 读取CSV文件
csv_file_path = 'Wimbledon_featured_matches_last.csv'
df = pd.read_csv(csv_file_path)
# data['p1_perform'] = data['point_victor'].apply(lambda x: 1 if x == 1 else 0)*data['server'].apply(lambda x: 0.347 if x == 1 else 0.653) + ((data['p1_winner'] == 1) | (data['p1_ace'] == 1)) + data['p1_net_pt_won'] + (data['p1_score'] == 50) * data['point_victor'].apply(lambda x: 1 if x == 1 else -1) - (data['point_victor'] != 1) * data['server'].apply(lambda x: 0.347 if x == 2 else 0.653) - data['p1_unf_err'] - data['p1_net_pt'] + data['p1_net_pt_won']
# data['p2_perform'] = data['point_victor'].apply(lambda x: 1 if x == 2 else 0)*data['server'].apply(lambda x: 0.347 if x == 2 else 0.653) + ((data['p2_winner'] == 1) | (data['p2_ace'] == 1)) + data['p2_net_pt_won'] + (data['p2_score'] == 50) * data['point_victor'].apply(lambda x: 1 if x == 2 else -1) - (data['point_victor'] != 2) * data['server'].apply(lambda x: 0.347 if x == 1 else 0.653) - data['p2_unf_err'] - data['p2_net_pt'] + data['p2_net_pt_won']
#
# data['p1_perform'] = data.groupby(['match_id','set_no', 'game_no'])['p1_perform'].cumsum()
# data['p2_perform'] = data.groupby(['match_id','set_no', 'game_no'])['p2_perform'].cumsum()
# # # 加载数据
# df = pd.read_csv('Wimbledon_featured_matches.csv')

# 选取数据
y1 = df['p1_perform'].iloc[6952:]
y2 = df['p2_perform'].iloc[6952:]

# 使用Seaborn改善美观
sns.set(style="whitegrid")

# 设置图形大小
plt.figure(figsize=(11, 8))

# 绘制线条并添加标记
plt.plot(df['point_no'].iloc[6952:], y1, label='Player 1', marker='o', linestyle='-', color='#FFA07A', markersize=5)
plt.plot(df['point_no'].iloc[6952:], y2, label='Player 2', marker='o', linestyle='-', color='#20B2AA', markersize=5)

# 添加标签和标题
plt.xlabel('Point Number', font=Path('palatino.ttf'),fontsize=12)
plt.ylabel('Performance',font=Path('palatino.ttf'), fontsize=12)
plt.title('Player 1 vs. Player 2 Performance', font=Path('palatino.ttf'), fontsize=14)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 添加图例
plt.legend()

plt.savefig('1_peform_1701.pdf', format='pdf')
# 显示图形
plt.show()
# 将修改后的数据保存回CSV文件
# data.to_csv('Wimbledon_featured_matches.csv', index=False)