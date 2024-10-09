import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设df是包含比赛数据的DataFrame，包含'point_no', 'server', 'point_victor', 'p1_score', 'p2_score'等列
df = pd.read_csv('Wimbledon_featured_matches.csv')

# 势头计算函数
def calculate_momentum(df, window_size=10, server_advantage_factor=0.8):
    """
    计算给定窗口大小内的势头，考虑发球方优势。
    server_advantage_factor: 发球方得分的权重因子，小于1表示减少发球方优势。
    """
    momentum_scores = []

    for i in range(window_size, len(df) + 1):
        window = df.iloc[i-window_size:i]
        p1_points = (window['point_victor'] == 1).sum()
        p2_points = (window['point_victor'] == 2).sum()

        # 考虑发球方优势
        for j, row in window.iterrows():
            if row['server'] == row['point_victor']:  # 发球方赢得该点
                if row['server'] == 1:
                    p1_points -= 1  # 减少发球方优势
                    p1_points += server_advantage_factor
                else:
                    p2_points -= 1
                    p2_points += server_advantage_factor

        momentum = p1_points - p2_points
        momentum_scores.append(momentum)

    return momentum_scores

# 应用势头计算函数
momentum_scores = calculate_momentum(df)

# 可视化势头变化
plt.figure(figsize=(12, 6))
plt.plot(df['point_no'][len(df['point_no']) - len(momentum_scores):], momentum_scores, label='Momentum (P1 - P2)')
plt.axhline(0, color='grey', lw=0.5)
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Match Momentum Visualization')
plt.legend()
plt.show()
