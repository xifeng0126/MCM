import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 假设 df 是您的 DataFrame
# 定义目标变量，这里我们简单地使用 p1_momt - p2_momt
df = pd.read_csv('Wimbledon_featured_matches_last2.csv')  # 替换为您的文件路径
df['momentum_shift'] = df['p1_momt'] - df['p2_momt']
df['unf_err_diff'] = df['p1_unf_err'] - df['p2_unf_err']
df['net_pt_won_diff'] = df['p1_net_pt_won'] - df['p2_net_pt_won']
df['double_fault_diff'] = df['p1_double_fault'] - df['p2_double_fault']
df['ace_diff'] = df['p1_ace'] - df['p2_ace']
df['winner_diff'] = df['p1_winner'] - df['p2_winner']
# 五个特征
df['points_won_diff'] = df['p1_points_won'] - df['p2_points_won']
df['state_diff'] = df['p1_state'] - df['p2_state']
df['good_point'] = df['ace_diff'] + df['winner_diff'] + df['net_pt_won_diff']
df['bad_point'] = df['double_fault_diff'] + df['unf_err_diff']
df['contin_point'] = ((df['point_victor'] == 1) & (df['last_winner'] == 1)).astype(int) - ((df['point_victor'] == 2) & (df['last_winner'] == 2)).astype(int)


# 选择特征列
features = ['points_won_diff', 'state_diff', 'good_point', 'bad_point', 'contin_point'] # 添加您选择的特征
X = df[features]
y = df['momentum_shift']  # 或者您转换后的二元目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 XGBoost 回归模型
model = xgb.XGBRegressor(objective ='reg:squarederror')
model.fit(X_train, y_train)

#使用模型
df2 = pd.read_csv('2022-wimbledon-1101.csv')
df2['unf_err_diff'] = df2['P1UnfErr'] - df2['P2UnfErr']
df2['net_pt_won_diff'] = df2['P1NetPointWon'] - df2['P2NetPointWon']
df2['double_fault_diff'] = df2['P1DoubleFault'] - df2['P2DoubleFault']
df2['ace_diff'] = df2['P1Ace'] - df2['P1Ace']
df2['winner_diff'] = df2['P1Winner'] - df2['P2Winner']
# 五个特征
df2['points_won_diff'] = df2['P1Score'] - df2['P2Score']
df2['good_point'] = df2['ace_diff'] + df2['winner_diff'] + df2['net_pt_won_diff']
df2['bad_point'] = df2['double_fault_diff'] + df['unf_err_diff']
df2['contin_point'] = ((df2['PointWinner'] == 1) & (df2['last_winner'] == 1)).astype(int) - ((df2['PointWinner'] == 2) & (df2['last_winner'] == 2)).astype(int)
X_input = df2[features]
y_input = model.predict(X_input)
df2['momentum_shift_pred'] = y_input
df2.to_csv('2022-wimbledon-1101.csv', index=False)