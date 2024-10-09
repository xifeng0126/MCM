import pandas as pd

# 加载数据
data = pd.read_csv('Wimbledon_featured_matches.csv')  # 替换为您的文件路径

# 初始化状态列
data['p1_state'] = 0
data['p2_state'] = 0

# 按照'match_id'和'game_no'进行分组
grouped_data = data.groupby(['match_id', 'game_no'])

for name, group in grouped_data:
    # 获取当前分组在原DataFrame中的索引
    idx = group.index

    # 比较盘数和局数，设置状态
    data.loc[idx, 'p1_state'] = (group['p1_sets'] > group['p2_sets']) * 3 + (group['p1_games'] > group['p2_games']) * 2
    data.loc[idx, 'p2_state'] = (group['p2_sets'] > group['p1_sets']) * 3 + (group['p2_games'] > group['p1_games']) * 2

# 保存结果
data.to_csv('Wimbledon_featured_matches.csv', index=False)
