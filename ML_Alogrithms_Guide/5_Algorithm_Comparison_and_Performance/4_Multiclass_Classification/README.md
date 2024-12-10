# 多元分类任务对比

## 数据集介绍

Iris 数据集（鸢尾花数据集）是一个经典的机器学习数据集，广泛用于模式识别、分类算法的测试和机器学习模型的评估。该数据集由英国生物学家 Ronald A. Fisher 于 1936 年提出，是统计学和机器学习领域中的重要数据集之一。它常用于教学和模型验证，特别是在多元分类问题的演示中。

## 数据集基本信息

Iris 数据集包含了 150 个鸢尾花样本，每个样本有 4 个特征和 1 个类别标签。数据集的目标是基于这 4 个特征来预测鸢尾花的种类。数据集的结构如下：

- **类别标签（目标变量）**：
  - **Setosa**：一种鸢尾花，通常具有较小的花瓣和萼片。
  - **Versicolor**：另一种鸢尾花，具有中等大小的花瓣和萼片。
  - **Virginica**：第三种鸢尾花，通常具有较大的花瓣和萼片。
  
  总共有 3 类鸢尾花。

- **特征（自变量）**：
  1. **Sepal Length（萼片长度）**：花萼的长度（单位：厘米）
  2. **Sepal Width（萼片宽度）**：花萼的宽度（单位：厘米）
  3. **Petal Length（花瓣长度）**：花瓣的长度（单位：厘米）
  4. **Petal Width（花瓣宽度）**：花瓣的宽度（单位：厘米）

每个特征都是连续值。

### 数据集组成
- **样本数**：150
- **特征数**：4
- **类别数**：3（Setosa、Versicolor、Virginica）

### 数据集预览

``` python
# 导入必要的库
from sklearn.datasets import load_iris
import pandas as pd

# 加载 Iris 数据集
iris = load_iris()

# 将数据转换为 DataFrame 格式
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# 添加目标列（即鸢尾花的种类）
iris_df['Species'] = iris.target_names[iris.target]

# 查看数据集的基本信息
print(iris_df.info())

# 显示前五行数据
print(iris_df.head())
```

## 多元分类任务概述

多元分类指的是目标变量（即待预测的变量）有多个类别，每个样本只能属于其中的一个类别。与二元分类不同，多元分类的问题要求模型能够识别并分类到多个可能的类别中。

我们将数据集分成两份，一份是训练集，一份是测试集。

通过训练集输入给训练模型，训练模型形成其内部的参数，进行模型建立。

## **数据预处理**

```python

# 1. 特征缩放（标准化）
X = iris_df.drop('Species', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 类别编码
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(iris_df['Species'])

# 3. 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# 显示拆分后的数据集信息
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

```

- **`X`**: 选择数据集中的特征列（即鸢尾花的四个特征），并将目标变量 `Species` 删除。
- **`scaler`**: 创建一个 `StandardScaler` 对象，用于标准化特征数据，使其均值为 0，标准差为 1。
- **`X_scaled`**: 通过 `fit_transform` 方法对特征数据进行标准化处理，返回标准化后的数据。
- **`encoder`**: 创建一个 `LabelEncoder` 对象，用于将类别标签转换为数值型（如 Setosa → 0, Versicolor → 1, Virginica → 2）。
- **`y_encoded`**: 将 `Species` 列的标签转换为数值型编码，返回编码后的目标变量。
- **`X_train, X_test`**: 将标准化后的特征数据划分为训练集和测试集，70% 用作训练，30% 用作测试。
- **`y_train, y_test`**: 将编码后的目标标签划分为训练集和测试集。
- **`test_size=0.3`**: 指定测试集占 30%。
- **`random_state=42`**: 确保数据拆分的可重复性，使每次运行代码时拆分结果相同。

## 模型统一训练和测试

```python
    # 训练模型
    model.fit(X_train_pca, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 打印分类报告和准确率
    print(f"Classification Report for {model.__class__.__name__}:\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
```

## 绘制效果图片

```python
    # PCA降维并可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 绘制降维后的数据点和分类边界
    plt.figure(figsize=(8, 6))

    # 绘制数据点，使用目标标签进行着色
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', edgecolors='k', s=50)

    # 设置标题和标签
    plt.title(f'{model.__class__.__name__} on Iris Dataset (PCA Reduced)', fontsize=14)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # 绘制分类边界
    h = .02  # 网格步长
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(pca.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # 显示图形
    plt.colorbar()
    plt.show()
```

## 测试函数

总结得到下面的函数，可输入模型观察到结果
```python
def train_and_visualize_iris_model(model):
    # 加载 Iris 数据集
    iris = load_iris()

    # 将数据转换为 DataFrame 格式
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # 添加目标列（即鸢尾花的种类）
    iris_df['Species'] = iris.target_names[iris.target]

   # 特征缩放（标准化）
    X = iris_df.drop('Species', axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 类别编码
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(iris_df['Species'])

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    # 训练模型
    model.fit(X_train_pca, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 打印分类报告和准确率
    print(f"Classification Report for {model.__class__.__name__}:\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # PCA降维并可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 绘制降维后的数据点和分类边界
    plt.figure(figsize=(8, 6))

    # 绘制数据点，使用目标标签进行着色
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', edgecolors='k', s=50)

    # 设置标题和标签
    plt.title(f'{model.__class__.__name__} on Iris Dataset (PCA Reduced)', fontsize=14)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # 绘制分类边界
    h = .02  # 网格步长
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(pca.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # 显示图形
    plt.colorbar()
    plt.show()
```
