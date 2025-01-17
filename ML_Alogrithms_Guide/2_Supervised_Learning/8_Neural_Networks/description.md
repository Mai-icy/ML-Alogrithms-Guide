## 算法说明

### 简单感知机

简单感知机由输入层和输出层构成，是将非线性函数应用于对特征值加权后的结果并进行识别的模型。它的工作原理基于加权求和和阶跃函数

举例：某特征维度为2，输入特征值为$(x_1,x_2)$，使用非线性函数 f 计算概率 y ：  
$y=f(w_0+w_1x_1+w_2x_2)$

- 加权求和：特征值的系数w1和w2称为 权重，常数项w0称为偏置。
- 激活函数：例如Sigmoid 可以将加权值转为一个概率值，通常激活函数后的输出通常是一个连续值，Sigmoid适合二值分类，如果要多分类，Softmax函数将更适合。

![image.png](images/5.png)  

图中是简单感知机的示意图，右图为简化图。

对于感知机的权值确定，感知机的权重在理想情况下在多次训练后会逐渐收敛到一个能够完美分割数据的解，前提是训练数据是线性可分的。如果数据是线性可分的，感知机算法保证最终会收敛。

### 神经网路

神经网络（Neural Network）可以看作是由多个感知机（Perceptron）通过分层构建而成的。

简单感知机不能很好学习某些数据的决策边界，如下图，典型的数据不是线性可分。

![image.png](images/6.png)

于是我们需要借用多个感知机，并进行一些处理。进行一个分层：

对这个例子 设置两个中间层

- 区分右上角的点和其他点的层
- 区分左下角的点和其他点的层

然后，设置综合这两个输出结果，同样利用简单感知机生成最终决定的层。通过这种做法，我们就可以根据数据是否进入被两条直线夹住的地方来分类了。示意图如下。

![image.png](images/7.png)

通过调节中间层的数量及层的深度可以学习更复杂的边界。如图。

![image.png](images/8.png)
