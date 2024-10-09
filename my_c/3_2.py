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
features = ['contin_point','points_won_diff', 'state_diff', 'good_point', 'bad_point'] # 添加您选择的特征
X = df[features]
y = df['momentum_shift']  # 或者您转换后的二元目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 XGBoost 回归模型
model = xgb.XGBRegressor(objective ='reg:squarederror')
model.fit(X_train, y_train)

# # 预测测试集
# y_pred = model.predict(X_test)
# # 计算MAE
# mae = mean_absolute_error(y_test, y_pred)
# print(f"MAE: {mae}")
#
# # 计算MAPE
# mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# print(f"MAPE: {mape}%")
#
# # 计算MSE
# mse = mean_squared_error(y_test, y_pred)
# print(f"MSE: {mse}")
#
# # 计算RMSE（均方根误差）
# rmse = np.sqrt(mse)
# print(f"RMSE: {rmse}")
#
# # 计算R²
# r2 = r2_score(y_test, y_pred)
# print(f"R²: {r2}")




# 特征重要性
# xgb.plot_importance(model)
# plt.show()

# # 获取feature importance
# plt.figure(figsize=(10, 8))
# plt.bar(range(len(features)), model.feature_importances_)
# plt.xticks(range(len(features)), X, fontsize=14)
# plt.title('Feature importance', fontsize=14)
# plt.show()
#
# model是在第1节中训练的模型
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[features])
# print(shap_values.shape)
# # 对单个样本的分析
j = 7000
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[j], df[features].iloc[j])
# # 保存图形
plt.savefig('shap_force_plot.pdf', bbox_inches='tight', format='pdf')
# 对特征的整体分析
# shap.summary_plot(shap_values, df[features], show=False)

# 使用图形对象的 savefig 方法保存图像
# plt.savefig('shap_summary_plot.pdf', bbox_inches='tight', format='pdf')

# 关闭图形，避免内存泄漏
# 重要性排序
# shap.summary_plot(shap_values, df[features], plot_type="bar", show=False)
# plt.savefig('shap_importance_plot.pdf', bbox_inches='tight', format='pdf')
# # 依赖关系
# shap.dependence_plot('points_won_diff', shap_values, df[features], interaction_index=None, show=False)
# plt.savefig('shap_dependence_plot.pdf', bbox_inches='tight', format='pdf')
# # 交互关系
# shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(df[features])
# shap.summary_plot(shap_interaction_values, df[features], max_display=5, show=False)
# plt.savefig('shap_interaction_plot.pdf', bbox_inches='tight', format='pdf')
# # #两个列名  一个potential一个是 international——reputation
# # # 两个变量交互下变量对目标值的影响。
# shap.dependence_plot('points_won_diff', shap_values, df[features], interaction_index='good_point')
