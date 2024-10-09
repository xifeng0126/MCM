import pandas as pd

# 加载数据
data = pd.read_csv('2022-wimbledon-1101.csv')


# 定义一个函数来更新每局的势头
def update_momentum(group):
    # 在每局开始时，重置势头为初始态
    group['momentum_shift'] = group['state_diff'].iloc[0] * 0.3
    # group['p1_momt'] = 0
    # group['p2_momt'] = 0

    # 遍历每一行（每一球）来更新势头
    for i in range(1, len(group)):
        row = group.iloc[i]
        prev_row = group.iloc[i - 1]

        # 连续进球
        if row['PointWinner'] == '1' and prev_row['last_winner'] == '1':
            group.at[group.index[i], 'momentum_shift'] += 5 * 0.3
        elif row['PointWinner'] == '2' and prev_row['last_winner'] == '2':
            group.at[group.index[i], 'momentum_shift'] -= 5 * 0.3

        # 点数状态
        if row['P1Score'] > row['P2Score']:
            group.at[group.index[i], 'momentum_shift'] += 1 * 0.26
        elif row['P2Score'] > row['P1Score']:
            group.at[group.index[i], 'momentum_shift'] -= 1 * 0.26

        # 好球
        for col in ['P1Ace', 'P1Winner', 'P1NetPointWon']:
            if row[col] == 1:
                group.at[group.index[i], 'momentum_shift'] += 5 * 0.12
        for col in ['P2Ace', 'P2Winner', 'P2NetPointWon']:
            if row[col] == 1:
                group.at[group.index[i], 'momentum_shift'] -= 5 * 0.12

        # 失误
        for col in ['P1DoubleFault', 'P1UnfErr']:
            if row[col] == 1:
                group.at[group.index[i], 'momentum_shift'] -= 5 * 0.17
        for col in ['P2DoubleFault', 'P1UnfErr']:
            if row[col] == 1:
                group.at[group.index[i], 'momentum_shift'] += 5 * 0.17

    return group


# 应用函数并更新DataFrame
data = data.groupby(['SetNo', 'GameNo']).apply(update_momentum).reset_index(drop=True)

# 保存结果
data.to_csv('2022-wimbledon-1101.csv', index=False)
