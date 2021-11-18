import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn import tree, svm, neighbors, ensemble
from matplotlib import pyplot as plt
import featureGet

feature = pd.read_csv('../feature.csv')
X = feature[['B', 'G', 'R', 'DENSITY']].to_numpy()
Y = feature['RESULT'].to_numpy()

# 数据标准化
X_transform = preprocessing.MinMaxScaler().fit_transform(X)

# 分割训练集、测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=5)

# 建立最小二乘线性回归模型
model1 = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

# lasso回归
model2 = LassoCV()

# ElasticNet回归
model3 = ElasticNetCV()

# 决策树回归
model4 = tree.DecisionTreeRegressor()

# SVM回归
model5 = svm.SVR()

# KNN回归
model6 = neighbors.KNeighborsRegressor()

# 随机森林回归
model7 = ensemble.RandomForestRegressor(n_estimators=20)

# 模型拟合
model = model4
model.fit(X_train, Y_train)

# score = model.score(X_test, Y_test)

fileName = "982.jpg"
B, G, R, DENSITY = featureGet.featureGet(fileName)
X = [[B, G, R, DENSITY]]
X = np.asarray(X)
Y_predict = model.predict(X)
print(Y_predict)

# print("this model's score is " + str(score))
# print("actual result: ")
# print(Y_test)
# print("predict result: ")
# Y_predict = np.around(Y_predict, 0)
# print(Y_predict)
# print("correct ratio: ")
# print(np.sum(Y_test == Y_predict) / len(Y_test))

# color_b = X[:, 0]
# color_g = X[:, 1]
# color_r = X[:, 2]
# density = X[:, 3]
#
# plt.subplot(2, 2, 1)
# plt.scatter(color_b, Y)
# plt.title("B")
#
# plt.subplot(2, 2, 2)
# plt.scatter(color_g, Y)
# plt.title("G")
#
# plt.subplot(2, 2, 3)
# plt.scatter(color_r, Y)
# plt.title("R")
#
# plt.subplot(2, 2, 4)
# plt.scatter(density, Y)
# plt.title("DENSITY")
#
# plt.suptitle("Feature Scatter")
# plt.tight_layout()
# plt.show()
