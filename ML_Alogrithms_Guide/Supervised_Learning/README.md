
# 有监督学习

有监督学习是一种通过标注数据来训练模型的学习方法。它的目标是从输入数据中学习一个映射关系，然后用于预测新数据的标签。该章节介绍了常见的有监督学习算法，包括回归、分类及其变种。

## 包含的算法

1. **线性回归 (Linear Regression)**  
   - 用途：回归分析
   - 简介：线性回归是一种基本的回归分析方法，通过拟合一条最佳的直线来预测连续变量。

2. **正则化 (Regularization)**  
   - 用途：避免过拟合
   - 简介：正则化技术通过在损失函数中增加惩罚项（如L1或L2正则化），帮助减少模型的复杂度，防止过拟合。

3. **逻辑回归 (Logistic Regression)**  
   - 用途：二分类问题
   - 简介：逻辑回归用于预测二分类问题，通过使用Sigmoid函数将线性回归的输出映射到0到1之间的概率值。

4. **支持向量机 (Support Vector Machine, SVM)**  
   - 用途：分类
   - 简介：支持向量机通过寻找最佳的分割超平面来进行分类，能够处理高维数据和非线性分类问题。

5. **支持向量机(核方法) (SVM with Kernel)**  
   - 用途：非线性分类
   - 简介：核方法将数据映射到高维空间，使用SVM在高维空间中找到超平面，以便处理非线性可分的数据。

6. **朴素贝叶斯 (Naive Bayes)**  
   - 用途：分类
   - 简介：朴素贝叶斯是一种基于贝叶斯定理的分类方法，假设特征之间是独立的，广泛应用于文本分类等领域。

7. **随机森林 (Random Forest)**  
   - 用途：分类与回归
   - 简介：随机森林是一种集成学习方法，通过训练多个决策树并结合其预测结果来提高分类或回归的准确性。

8. **神经网络 (Neural Networks)**  
   - 用途：分类与回归
   - 简介：神经网络模仿人脑神经元的工作原理，具有强大的表达能力，广泛应用于图像识别、语音处理等领域。

9. **KNN算法 (K-Nearest Neighbors)**  
   - 用途：分类与回归
   - 简介：KNN是一种基于实例的学习方法，通过计算样本间的距离来进行分类或回归，简单易懂，适用于小规模数据集。

## 链接导向

- [线性回归 (Linear Regression)](./1_Linear_Regression/README.md)
- [正则化 (Regularization)](./2_Regularization/README.md)
- [逻辑回归 (Logistic Regression)](./3_Logistic_Regression/README.md)
- [支持向量机 (SVM)](./4_Support_Vector_Machine/README.md)
- [支持向量机(核方法) (SVM with Kernel)](./5_Support_Vector_Machine_Kernel/README.md)
- [朴素贝叶斯 (Naive Bayes)](./6_Naive_Bayes/README.md)
- [随机森林 (Random Forest)](./7_Random_Forest/README.md)
- [神经网络 (Neural Networks)](./8_Neural_Networks/README.md)
- [KNN算法 (K-Nearest Neighbors)](./9_KNN/README.md)

