import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#线性回归

# 加载糖尿病数据集(特征，目标值)
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)


# 仅使用一个特征
diabetes_X = diabetes_X[:, np.newaxis, 2]


# 将数据分割为训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# 将目标值分割为训练集和测试集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# 创建线性回归对象
regr = linear_model.LinearRegression()

# 使用训练集训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 使用测试集进行预测
diabetes_y_pred = regr.predict(diabetes_X_test)

# 系数
print('系数: ', regr.coef_)
# 均方误差
print('均方误差: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 决定系数：1 表示完美预测
print('决定系数: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# 绘制输出
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

plt.show()
