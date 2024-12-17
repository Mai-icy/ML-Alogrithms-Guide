# 非线性回归任务对比

## 非线性回归任务概述

相比于线性回归，非线性回归的自变量和因变量之间的关系是曲线型的。

非线性回归模型会有多种形式
* 多项式回归
* 指数回归
* 对数回归
* 双曲正切回归
* Sigmoid函数
* ...

对于非线性回归模型，可以通过数学变换转换成线性回归的形式。也可以通过对应的训练模型来处理。

## 数据预处理

数据预处理的方式与线性回归相似，都是将数据集拆分位训练集和测试集两部分。下面用糖尿病进展预测数据集作为例子。

## 数据集介绍
Diabetes 130-US hospitals for years 1999-2008，该数据集来自 Kaggle，其详细信息包括：

* 数据集内容：包含来自130家美国医院的糖尿病患者数据，收集了1999至2008年间的记录。数据包括患者的基本信息（如年龄、性别、体重等）、诊断信息（如糖尿病类型）、治疗方案、实验室结果（如血糖水平、血压等）等。
* 目标变量：预测糖尿病的进展，通常涉及预测患者未来的血糖水平、并发症或其他健康指标的变化。
* 特征：常见的特征包括年龄、性别、体重、饮食、生活习惯、药物使用、血糖水平、体重指数（BMI）等。
* 使用场景：通过机器学习模型，研究人员可以预测糖尿病患者的病情变化，帮助制定个性化治疗方案。

## 数据集预处理

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# 读取数据集
df = pd.read_csv('diabetes_data.csv')
# 查看数据集的前几行
df.head()
# 标准化数值型特征
scaler = StandardScaler()
df[['numerical_column1', 'numerical_column2']] = scaler.fit_transform(df[['numerical_column1', 'numerical_column2']])
# 使用pandas的get_dummies进行独热编码
df = pd.get_dummies(df, columns=['categorical_column'])
# 假设目标变量是 'target_column'
X = df.drop('target_column', axis=1)
y = df['target_column']
# 拆分数据集，80%作为训练集，20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

将数据集拆分为训练集（80%）和测试集（20%），确保模型训练时没有使用测试数据。

## 模型统一训练及评估

```python
# 训练模型
    model.fit(X_train, y_train)
    # 进行预测
    y_pred = model.predict(X_test)
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 打印评估指标
    print("模型准确率: {:.2f}%".format(accuracy * 100))
    print("\n分类报告:\n", class_report)
    print("\n混淆矩阵:\n", conf_matrix)
```
## 测试函数

```python
def preprocess_and_train_model(data_path, model, target_column, numerical_columns, categorical_columns):
    # 加载数据集
    df = pd.read_csv(data_path)
    # 查看数据集的前几行
    print(df.head())
    # 标准化数值型特征
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    # 使用pandas的get_dummies进行独热编码
    df = pd.get_dummies(df, columns=categorical_columns)
    # 拆分数据集
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练模型
    model.fit(X_train, y_train)
    # 进行预测
    y_pred = model.predict(X_test)
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 打印评估指标
    print("模型准确率: {:.2f}%".format(accuracy * 100))
    print("\n分类报告:\n", class_report)
    print("\n混淆矩阵:\n", conf_matrix)

    return model
```

<!-- ## 支持向量机（核方法）

核方法（kernel methods）是一种用于处理无法人力标注特征值的复杂在数据的技巧。

将数据引入一个更高维度的空间，就可以将原本线性不可分的数据变为线性可分。同时，支持向量机可以在高维空间中找到适合的决策边界再投影回原始的低维空间。

![1.png](images/1.png)

通过核函数，核方法可以无须构建具体的高维空间。

```python
model = SVC()
model.fit(X_train, y_train)
```

代码中没有明确指定使用哪个核方法，因为默认使用RBF（Radial Basis Function，径向基函数）核方法。

### 结果展示

![2.png](images/2.png)

不同的核函数得到的决策边界的形状也不同。

## 随机森林

随机森林采用决策树作为弱分类器，在bagging的样本随机采样基础上，⼜加上了特征的随机选择。

当前结点特征集合（d个特征），随机选择k个特征子集，再选择最优特征进行划分。k控制了随机性的引入程度，推荐值：$k=log_2d$

对预测输出进行结合时，分类任务——简单投票法；回归任务——简单平均法

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)
```

### 参数解释

- n_estimators：参数控制随机森林中决策树的数量。更多的树通常能提高模型的性能，但也会增加计算成本。增加 n_estimators 可以提高模型的稳定性，减少单棵树的过拟合风险，但同时增加训练时间和内存消耗。
- random_state：随机种子
- oob_score：是否使用袋外样本来估计泛化精度。默认False。

## 神经网络

在开头引用的例子便是神经网络模型。其将输入层读入的数据在中间层用Sigmoid等非线性函数计算，在输出层也使用非线性函数计算并输出结果。

```python 
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    epochs=20,  # 训练轮数
                    batch_size=128,  # 批量大小
                    validation_split=0.2)  # 使用20%的数据作为验证集
# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
```

### 参数解释

- **Sequential Model**：`Sequential` 是一种线性堆叠模型，意味着你可以一层接一层地添加层。
  
- **Dense Layer**：`Dense` 层是全连接层，每个节点都与上一层的所有节点相连。第一层需要指定 `input_shape` 参数，表示输入特征的形状。激活函数 'relu' （Rectified Linear Unit）被广泛应用于隐藏层中。

- **Dropout Layer**：`Dropout` 是一种正则化技术，它随机地按照一定比例关闭一些节点，以防止模型过拟合。这里的 `0.5` 表示每次更新参数时，随机丢弃一半的单元。

- **Output Layer**：最后一层只有一个节点，并使用了 'sigmoid' 激活函数，这适用于二分类问题，输出可以解释为属于某一类的概率。

- **Compile**：在编译阶段，我们指定了损失函数（`binary_crossentropy` 对于二分类问题很常见）、优化算法 (`Adam`) 和评估模型性能的指标（如准确率 `accuracy`）。

## KNN

KNN训练时不用计算，无须在意数据是否线性，所以可以用于非线性回归。

其将未知数据和训练数据的距离进行计算，通过多数表决找到最邻近的k个点，再进行分类。

```python
knn = KNeighborsClassifier(n_neighbors=3)
# 训练模型
knn.fit(X_train, y_train)
# 使用模型进行预测
y_pred = knn.predict(X_test)
```

### 参数解释
- **n_neighbors** :即k值，是邻居的数量。

### 结果展示

k值不同的决策边界结果哦不同

![3.png](images/3.png) -->

