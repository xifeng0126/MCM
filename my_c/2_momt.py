import pandas as pd

# 加载数据
data = pd.read_csv('Wimbledon_featured_matches_last.csv')


# 定义一个函数来更新每局的势头
def update_momentum(group):
    # 在每局开始时，重置势头为初始态
    group['p1_momt'] = group['p1_state'].iloc[0] * 0.3
    group['p2_momt'] = group['p2_state'].iloc[0] * 0.3
    # group['p1_momt'] = 0
    # group['p2_momt'] = 0

    # 遍历每一行（每一球）来更新势头
    for i in range(1, len(group)):
        row = group.iloc[i]
        prev_row = group.iloc[i - 1]

        # 连续进球
        if row['point_victor'] == '1' and prev_row['last_winner'] == '1':
            group.at[group.index[i], 'p1_momt'] += 5 * 0.3
        elif row['point_victor'] == '2' and prev_row['last_winner'] == '2':
            group.at[group.index[i], 'p2_momt'] += 5 * 0.3

        # 点数状态
        if row['p1_points_won'] > row['p2_points_won']:
            group.at[group.index[i], 'p1_momt'] += 1 * 0.26
        elif row['p2_points_won'] > row['p1_points_won']:
            group.at[group.index[i], 'p2_momt'] += 1 * 0.26

        # 好球
        for col in ['p1_ace', 'p1_winner', 'p1_net_pt_won']:
            if row[col] == 1:
                group.at[group.index[i], 'p1_momt'] += 5 * 0.12
        for col in ['p2_ace', 'p2_winner', 'p2_net_pt_won']:
            if row[col] == 1:
                group.at[group.index[i], 'p2_momt'] += 5 * 0.12

        # 失误
        for col in ['p1_double_fault', 'p1_unf_err']:
            if row[col] == 1:
                group.at[group.index[i], 'p1_momt'] -= 5 * 0.17
        for col in ['p2_double_fault', 'p2_unf_err']:
            if row[col] == 1:
                group.at[group.index[i], 'p2_momt'] -= 5 * 0.17

    return group


# 应用函数并更新DataFrame
data = data.groupby(['match_id', 'set_no', 'game_no']).apply(update_momentum).reset_index(drop=True)

# 保存结果
data.to_csv('Wimbledon_featured_matches_last2.csv', index=False)
