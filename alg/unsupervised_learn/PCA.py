# PCA,提取主要特征，降维
# 结合SVM进行人脸识别
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



# 在stdout上显示进度日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# #############################################################################
# 下载数据，如果尚未保存到磁盘，则将其加载为numpy数组

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 探查图像数组以查找形状（用于绘图）
n_samples, h, w = lfw_people.images.shape

# 对于机器学习，我们直接使用2维数据（由于此模型忽略了相对像素位置信息）
X = lfw_people.data
n_features = X.shape[1]

# 要预测的标签是人的ID
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("总数据集大小:")
print("样本数量: %d" % n_samples)
print("特征数量: %d" % n_features)
print("类别数量: %d" % n_classes)

# #############################################################################
# 使用分层k折将数据集拆分为训练集和测试集

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# #############################################################################
# 在人脸数据集上计算PCA（特征脸）：无监督特征提取/降维
n_components = 150

print("从 %d 张脸中提取前 %d 个特征脸"
      % (X_train.shape[0], n_components))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("耗时 %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("在正交基上投影输入数据")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("耗时 %0.3fs" % (time() - t0))

# #############################################################################
# 训练SVM分类模型

print("将分类器拟合到训练集")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("耗时 %0.3fs" % (time() - t0))
print("网格搜索找到的最佳估计器:")
print(clf.best_estimator_)

# #############################################################################
# 在测试集上对模型质量进行定量评估

print("在测试集上预测人物的名字")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("耗时 %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# #############################################################################
# 使用matplotlib对预测结果进行定性评估

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """绘制肖像图画廊的辅助函数"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# 在测试集的一部分上绘制预测结果

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'pre: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# 绘制最显著的特征脸图库

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
