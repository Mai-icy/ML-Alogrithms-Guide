## 算法说明

逻辑回归根据数据x和表示其所属类别的标签y进行学习，计算概率。

如果标签是二元分类，则可以使用前面的y=0, 1这种二元数值表示。

### 与线性回归进行比较：

相同点：基本思想，对数据x乘以权重向量w，再加上偏置w0，计算wT x+w0的值

不同点：逻辑回归的输出范围限制在01之间，使用了Sigmoid函数：

σ(z)=1/[1+exp(-z)]

对输入数据x使用Sigmoid函数，p=σ(wT x+w0) 得到标签为y的概率p。（二元分类使用0.5作为阈值）

误差函数使用逻辑损失。逻辑损失在分类失败时返回大值，在分类成功时为小值。

与在误差回归中引入的均方误差不同的是，我们无法通过式子变形来计算逻辑损失的最小值，因此需要采用梯度下降法通过数值计算来求解。（机器学习中经常会通过数值计算来近似求解）
