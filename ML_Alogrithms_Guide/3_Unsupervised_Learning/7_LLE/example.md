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
