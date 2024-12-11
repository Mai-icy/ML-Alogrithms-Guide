# 二元分类任务对比

## 数据集介绍

乳腺癌数据集（Breast Cancer Wisconsin dataset）是一个非常著名的数据集，广泛用于机器学习和数据科学领域，特别是在分类问题中的应用。该数据集包含了与乳腺癌诊断相关的特征，主要用于预测肿瘤是良性（Benign）还是恶性（Malignant）。数据集的目标是帮助医生通过分析肿瘤的不同特征，做出准确的诊断。

### 数据集基本信息

- 数据集规模：该数据集包含 569 个样本，其中包括 30 个特征。
- 特征：每个样本由 30 个不同的特征描述，涵盖了肿瘤的各种属性，如半径、纹理、周长、面积、平滑度、紧凑性、对称性和分形维数等。
- 目标变量（类别）：每个样本的目标变量是肿瘤的类型，分为两类：
    - **0**：恶性肿瘤（Malignant）
    - **1**：良性肿瘤（Benign）

### 数据集预览

```python
from sklearn import datasets
import pandas as pd
import numpy as np

# 加载乳腺癌数据集
data = datasets.load_breast_cancer()

# 获取特征矩阵 X 和目标变量 y
X = data.data
y = data.target

# 输出数据集的基本信息
print("数据集的基本信息：")
print(f"特征矩阵的形状: {X.shape}")
print(f"类别标签的数量: {len(y)}")
print(f"类别标签（0: 恶性，1: 良性）: {data.target_names}")
print(f"特征名称: {data.feature_names}")

# 输出前几行数据样本（前 5 个样本）
print("\n前5个样本的特征值：")
print(pd.DataFrame(X, columns=data.feature_names).head())

# 输出前几行目标变量（目标标签）
print("\n前5个样本的目标标签（0: 恶性，1: 良性）：")
print(y[:5])

# 计算并输出一些统计信息（均值、标准差）
print("\n特征数据的统计信息：")
df = pd.DataFrame(X, columns=data.feature_names)
print(df.describe())

# 输出类别分布（恶性和良性肿瘤样本数）
print("\n类别分布：")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"{data.target_names[label]}: {count} 样本")
```

## 二元分类任务概述

模型只有一组 输入特征X 和 一个 输出特征 Y，其中输出特征为0或者1，代表其属于二元分类中的哪一类。

我们将数据集分成两份，一份是训练集，一份是测试集。

通过训练集输入给训练模型，训练模型形成其内部的参数，进行模型建立。

## **数据预处理**

由于便于二元分类的可视化效果，将数据集通过PCA算法降低到二维，通过二维数据训练模拟。

```python
data = datasets.load_breast_cancer()
X = data.data  # 使用所有特征
y = data.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用PCA降维到二维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

- **`X_train`** 和 **`y_train`**：训练集中的特征和目标变量。这部分数据用于训练模型，帮助模型学习输入和输出之间的关系。
- `X_test` 和 `y_test`：测试集中的特征和目标变量。这部分数据用于评估训练后的模型性能，确保模型在未知数据上的表现。
- `X_train_pca` 和 `x_test_pca`：测试集中的特征和目标变量的PCA降维后的二维数据。
- `test_size=0.3`：表示将数据集的 30% 用作测试集，剩余的 70% 用作训练集。
- `random_state=42`：是随机数种子，确保每次运行代码时，数据分割的方式是一致的。`42` 是一个常用的固定种子值（也可以换成其他任意整数）。这样可以保证你在不同时间、不同环境下运行代码时，得到相同的训练集和测试集划分。

## 模型统一训练和测试

```python
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")
```

## 绘制效果图片

```python
# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # 决策边界填充
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='coolwarm', s=100, edgecolor='k',
                marker='o', alpha=0.8)

# 绘制网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 标题和标签
plt.title(f'{model.__class__.__name__} - 决策边界 (PCA降维)', fontsize=16)
plt.xlabel('主成分 1', fontsize=12)
plt.ylabel('主成分 2', fontsize=12)

# 添加颜色条
plt.colorbar()

# 设置图例
plt.legend(title='预测标签', loc='upper right', fontsize=12)

plt.show()
```

通过颜色来表明分类结果，横轴表示PCA降维后的主成分1，纵轴降维后的主成分2。绘制决策边界可视化。

## 测试函数

总结以上步骤得出以下函数，通过输入模型就可以观察到分类结果

```python
def train_and_visualize_model(model):
    # 加载乳腺癌数据集
    data = datasets.load_breast_cancer()
    X = data.data  # 使用所有特征
    y = data.target

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用PCA降维到二维
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 训练模型
    model.fit(X_train_pca, y_train)

    # 预测
    y_pred = model.predict(X_test_pca)

    # 打印准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy * 100:.2f}%")

    # 绘制决策边界
    h = .02  # 网格点的间隔
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # 决策边界填充
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='coolwarm', s=100, edgecolor='k',
                    marker='o', alpha=0.8)

    # 绘制网格线
    plt.grid(True, linestyle='--', alpha=0.5)

    # 标题和标签
    plt.title(f'{model.__class__.__name__} - 决策边界 (PCA降维)', fontsize=16)
    plt.xlabel('主成分 1', fontsize=12)
    plt.ylabel('主成分 2', fontsize=12)

    # 添加颜色条
    plt.colorbar()

    # 设置图例
    plt.legend(title='预测标签', loc='upper right', fontsize=12)

    plt.show()
```