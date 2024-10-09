import pandas as pd

# # 加载数据集
df = pd.read_csv('2022-wimbledon-1101.csv')
#
# # 过滤特定的match_id
# filtered_data = df[df['match_id'] == '2022-wimbledon-1101']
#
# # 保存数据集
# filtered_data.to_csv('2022-wimbledon-1101.csv', index=False)

# def calculate_state_diff(group):
#     group['state_diff'] = 0  # 初始化state_diff列
#     if group['P1GamesWon'].iloc[0] > group['P2GamesWon'].iloc[0]:
#         group['state_diff'] += 2
#     elif group['P1GamesWon'].iloc[0] < group['P2GamesWon'].iloc[0]:
#         group['state_diff'] -= 2
#
#
#     return group

# 按SetNo和GameNo分组，并应用calculate_state_diff函数
df['state_diff'].iloc[153:] += 3    # 初始化state_diff列

# 查看结果
df.to_csv('2022-wimbledon-1101.csv', index=False)