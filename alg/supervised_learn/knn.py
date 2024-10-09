from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split#用于模型划分
from sklearn.neighbors import KNeighborsClassifier##KNN算法(k近邻算法)
import numpy as np

# 载入数据集
iris_dataset = load_iris()
X = iris_dataset['data']#特征
Y = iris_dataset['target']#类别
print(X)
print(Y)

# 数据划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#训练阶段
knn = KNeighborsClassifier(n_neighbors=5)#设置邻居数K
knn.fit(X_train, Y_train)#构建基于训练集的模型

#测试评估模型
Y_pred=knn.predict(X_test)
print("Test set score:{:.2f}".format(knn.score(X_test, Y_test)))

# 做出预测，预测花萼长5cm宽2.9cm，花瓣长1cm宽0.2cm的花型
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))

