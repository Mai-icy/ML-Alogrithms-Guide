# LLE

## 概述

无监督学习中的一项重要任务是将结构复杂的数据转化为更简单的形式。

LLE（Locally Linear Embedding，局部线性嵌入）可以将以弯曲或扭曲的状态埋藏在高维空间中的结构简单地表示在低维空间中。

LLE被称为流形学习，它的目标是对具有非线性结构的数据进行降维。

对于原始数据，是典型的瑞士卷数据集图像。LLE和PCA的降维结果如下所示。PCA更像简单的压扁，由于PCA更适合变量之间有一定关联性的数据，所以在降维时，LLE更适合。

![image.png](images/1.png)

## 算法说明

LLE 是以降维后依旧保持原始高维空间中的局部线性组合关系作为核心

LLE算法要求数据点由其近邻点的线性组合来表示

对于数据点 x1，以最接近 x1 的两个点 x2 和 x3 的线性组合来表示它。

$x_1=w_{12} x_2 + w_{13} x_3$

可以得到两个权值w12，w13.

目的是通过降维后，依旧保持这个线性组合关系，如下图所示

LLE将高维的曲折关系转化为邻近点的组合关系。

![image.png](images/2.png)

对于近邻点数量，是超参数，设定为k个。

步骤：

- 找到数据点 xi 的 k 个近邻点
- 求出由 k 个近邻点线性组合的权值
- 使用权值计算出低维度的 y_i

在确定近邻点数量后，首先，为了求出权重 wij，我们将 xi 和 其近邻点的线性组合 的误差表示为

$x_i - \sum_j w_{ij} x_j$

随着 wij 值的变化，这个误差会增大或减少。

通过计算所有  xi 和线性组合的差的平方和，我们将权重 wij  与误差之间的关系表示为以下误差函数： $e(W) = \sum_i \left| x_i - \sum_j w_{ij} x_j \right|^2$

除了 wij 近邻点之外，表达式中的值都是 0。另外还有一个约束条件：对于某个 i，有

$\sum_j w_{ij} = 1$

我们可以认为权重 wij 表示数据点 xi 及其近邻点之间的关系，而 LLE 通过这种关系在低维空间中也得以保持。在计算完权重后，我们计算低维空间中表示数据点的 y_i：

$\Phi(y) = \sum_i \left| y_i - \sum_j w_{ij} y_j \right|^2$

前面求出使误差最小的权重 wij，现在要做的是利用刚刚求出的 wij，求使误差最小的 y

## 示例代码

```python
from sklearn.datasets import samples_generator
from sklearn.manifold import LocallyLinearEmbedding
data, color = samples_generator.make_swiss_roll(n_samples=1500)
n_neighbors = 12  # 近邻点的数量
n_components = 2  # 降维后的维度
model = LocallyLinearEmbedding(n_neighbors=n_neighbors,
n_components=n_components)
model.fit(data)
print(model.transform(data))  # 变换后的数
```

## 详细说明

### 流形学习

流形指的是可以将局部看做没有弯曲的空间，类似于将地球的局部绘制平面地图

“所谓的流形就是从局部来看是低维空间的结构被埋藏在高维空间里”

### 近邻点数量

近邻点数量是超参数，对于不同的数量有不同的降维结果。如下图

![image.png](images/3.png)

当近邻点设置为5时，LLT没有连贯的结构，分布狭窄，反应信息少。

当近邻点设置为50时，不同颜色的点距离很近，无法把握局部结构。

这个参数十分重要，影响很大。