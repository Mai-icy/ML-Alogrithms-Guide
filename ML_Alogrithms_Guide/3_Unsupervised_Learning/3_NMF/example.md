## 示例代码

```python
from sklearn.decomposition import NMF
from sklearn.datasets.samples_generator import make_blobs
centers = [[5, 10, 5], [10, 4, 10], [6, 8, 8]]
V, _ = make_blobs(centers=centers)  # 以centers为中心生成数据
n_components = 2  # 潜在变量的个数
model = NMF(n_components=n_components)
model.fit(V)
W = model.transform(V) # 分解后的矩阵
H = model.components_
print(W)
print(H)
```
