## 示例代码

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
data = load_digits()
n_components = 2  # 设置降维后的维度为2
model = TSNE(n_components=n_components)
print(model.fit_transform(data.data))
```
