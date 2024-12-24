## 示例代码

```python
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
data = load_iris()
n_components = 3  # 高斯分布的数量
model = GaussianMixture(n_components=n_components)
model.fit(data.data)
print(model.predict(data.data))  # 预测类别
print(model.means_)  # 各高斯分布的均值
print(model.covariances_)  # 各高斯分布的方差
```
