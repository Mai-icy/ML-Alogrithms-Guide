## 示例代码

下面是对鸢尾花数据集应用k-means算法的代码。

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
data = load_iris()
n_clusters = 3  # 将簇的数量设置为3
model = KMeans(n_clusters=n_clusters)
model.fit(data.data)
print(model.labels_)  # 各数据点所属的簇
print(model.cluster_centers_)  # 通过fit()计算得到的重心
```
