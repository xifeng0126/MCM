import pandas as pd

# 加载数据
data = pd.read_csv('2022-wimbledon-1101.csv')  # 替换为您的文件路径

# 初始化last_winner列为0
data['last_winner'] = 0

# 定义函数来处理每个分组
def assign_last_winner(group):
    # 除了分组的第一行外，将'game_victor'列的值向下移动一行来填充'last_winner'
    group['last_winner'][1:] = group['PointWinner'].iloc[:-1]
    return group

# 按照'match_id'和'game_no'进行分组，并应用上述函数
data = data.groupby(['SetNo', 'GameNo']).apply(assign_last_winner).reset_index(drop=True)

# 保存结果
data.to_csv('2022-wimbledon-1101.csv', index=False)
