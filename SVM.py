# -*- coding:utf-8 -*-
"""
作者：yzf93
日期:2022年04月16日
"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# 按老师的要求，我们自己处理数据集
iris = {}
with open(file='iris.data', mode='r', encoding='utf-8') as f:
    temp_data = []
    temp_label = []
    for item in f.readlines():
        if item:
            temp1_data = []
            for item1 in item.split(',')[:-1]:
                temp1_data.append(float(item1))
            temp_data.append(temp1_data)
            if item.split(',')[-1][:-1] == 'Iris-setosa':
                temp_label.append(0)
            elif item.split(',')[-1][:-1] == 'Iris-versicolor':
                temp_label.append(1)
            else:
                temp_label.append(2)
    iris['data'] = np.array(temp_data)
    iris['label'] = np.array(temp_label)

wine = {}
with open(file='wine.data', mode='r', encoding='utf-8') as f:
    temp_data = []
    temp_label = []
    for item in f.readlines():
        if item:
            item.strip()
            temp1_data = []
            for item1 in item.split(',')[1:]:
                temp1_data.append(float(item1))
            temp_data.append(temp1_data)
            if item.split(',')[0] == '1':
                temp_label.append(0)
            elif item.split(',')[0] == '2':
                temp_label.append(1)
            else:
                temp_label.append(2)
    wine['data'] = np.array(temp_data)
    wine['label'] = np.array(temp_label)

data = wine['data']  # 从数据集字典中导入数据
label = wine['label']  # 从数据集字典中导入标签
# 按比例选取test和train，stratify为按标签分类，random是随机数种子，种子不同导致最后训练效果不同
data_train, data_test, label_train, target_test = train_test_split(data, label, stratify=label, random_state=0)
print('**********************以下为红酒数据集训练效果**********************')
# 调用SVC，采用线性核，C是课堂上讲到的软间隔的惩罚系数，gamma为核函数系数
svc = SVC(kernel='linear', gamma='auto', C=1)
svc.fit(data_train, label_train)
print(f"线性核准确率（训练集）: {svc.score(data_train, label_train)}")
print(f"线性核准确率（测试集）: {svc.score(data_test, target_test)}")

# 调用SVC，采用高斯核，C是课堂上讲到的软间隔的惩罚系数
svc2 = SVC(kernel='rbf', gamma='scale', C=1)
svc2.fit(data_train, label_train)
print(f"高斯核准确率（训练集）: {svc2.score(data_train, label_train)}")
print(f"高斯核准确率（测试集）: {svc2.score(data_test, target_test)}")

# （这项是我自己加的）调用SVC，采用多项式核，C是课堂上讲到的软间隔的惩罚系数
svc3 = SVC(kernel='poly', degree=7, gamma='auto', coef0=1.0, C=1)
svc3.fit(data_train, label_train)
print(f"多项式核准确率（训练集）: {svc3.score(data_train, label_train)}")
print(f"多项式核准确率（测试集）: {svc3.score(data_test, target_test)}")

# random_state随机数种子，max_iter迭代次数，alpha正则化参数
mlp = MLPClassifier(random_state=1, max_iter=2000, alpha=0.001, activation='tanh')
mlp.fit(data_train, label_train)
print(f"BP神经网络采用tanh激活函数准确率（训练集）: {mlp.score(data_train, label_train)}")
print(f"BP神经网络采用tanh激活函数准确率（测试集）: {mlp.score(data_test, target_test)}")

# random_state随机数种子，max_depth树最大深度，criterion='entropy'代表C4.5算法，默认是GINI
tree = DecisionTreeClassifier(random_state=3, max_depth=4, criterion='entropy')
tree.fit(data_train, label_train)
print(f"决策树准确率（训练集）: {tree.score(data_train, label_train)}")
print(f"决策树准确率（测试集）: {tree.score(data_test, target_test)}")


data = iris['data']  # 从数据集字典中导入数据
label = iris['label']  # 从数据集字典中导入标签
# 按比例选取test和train，stratify为按标签分类，random是随机数种子，种子不同导致最后训练效果不同
data_train, data_test, label_train, target_test = train_test_split(data, label, stratify=label, random_state=0)
print('**********************以下为鸢尾花数据集训练效果**********************')
# 调用SVC，采用线性核，C是课堂上讲到的软间隔的惩罚系数，gamma为核函数系数
svc1 = SVC(kernel='linear', gamma='auto', C=1)
svc1.fit(data_train, label_train)
print(f"线性核准确率（训练集）: {svc1.score(data_train, label_train)}")
print(f"线性核准确率（测试集）: {svc1.score(data_test, target_test)}")

# 调用SVC，采用高斯核，C是课堂上讲到的软间隔的惩罚系数
svc2 = SVC(kernel='rbf', gamma='auto', C=1)
svc2.fit(data_train, label_train)
print(f"高斯核准确率（训练集）: {svc2.score(data_train, label_train)}")
print(f"高斯核准确率（测试集）: {svc2.score(data_test, target_test)}")

# （这项是我自己加的）调用SVC，采用多项式核，C是课堂上讲到的软间隔的惩罚系数
svc3 = SVC(kernel='poly', degree=7, gamma='auto', coef0=1.0, C=1)
svc3.fit(data_train, label_train)
print(f"多项式核准确率（训练集）: {svc3.score(data_train, label_train)}")
print(f"多项式核准确率（测试集）: {svc3.score(data_test, target_test)}")

# random_state随机数种子，max_iter迭代次数，alpha正则化参数
mlp = MLPClassifier(random_state=1, max_iter=10000, alpha=0.001, activation='tanh')
mlp.fit(data_train, label_train)
print(f"BP神经网络采用tanh激活函数准确率（训练集）: {mlp.score(data_train, label_train)}")
print(f"BP神经网络采用tanh激活函数准确率（测试集）: {mlp.score(data_test, target_test)}")

# random_state随机数种子，max_depth树最大深度，criterion='entropy'代表C4.5算法，默认是GINI
tree = DecisionTreeClassifier(random_state=3, max_depth=4, criterion='entropy')
tree.fit(data_train, label_train)
print(f"决策树准确率（训练集）: {tree.score(data_train, label_train)}")
print(f"决策树准确率（测试集）: {tree.score(data_test, target_test)}")
