## 详细说明

以下是对同一数据使用混合高斯分布和K-means进行聚类。

![image.png](images/4.png)

我们可以看到，混合高斯分布的数据会更优，因为高斯分布对于椭圆形数据更优，而k-means对重心开始呈圆形分布的数据会更好，因此椭圆成为了k-means的软肋，更该使用混合高斯分布。