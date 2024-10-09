from sklearn.model_selection import ParameterGrid, train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 假设您已经完成了数据准备和分割
# X_train, X_test, y_train, y_test 已准备好
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


# 扩大参数范围并增加参数种类
param_grid = {
    'max_depth': [3, 6, 9, 12],  # 更广泛的范围
    'n_estimators': [50, 100, 200, 300],
}

# 使用网格搜索训练模型
rmse_results = []

for params in ParameterGrid(param_grid):
    model = xgb.XGBRegressor(objective='reg:squarederror', **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmse_results.append({**params, 'RMSE': rmse})

# 转换为 DataFrame
rmse_df = pd.DataFrame(rmse_results)

# 选择一组参数进行透视，例如：max_depth 和 n_estimators
pivot_table = rmse_df.pivot('max_depth', 'n_estimators', 'RMSE')

# 绘制热力图，整体偏红色，移除数字注释
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap="Reds", cbar_kws={'label': 'RMSE'})
plt.title("RMSE Heatmap for XGBoost Parameters",font=Path('palatino.ttf'),fontsize=16)
plt.xlabel('Number of Estimators', font=Path('palatino.ttf'), fontsize=14)
plt.ylabel('Maximum Depth', font=Path('palatino.ttf'), fontsize=14)
plt.savefig('3_4.pdf', format='pdf')
plt.show()
