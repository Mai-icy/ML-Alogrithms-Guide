## 示例代码

假设一个使用8个变量（=单词的个数）表示的数据集，现用2个潜在变量去表示它。

```python
from sklearn.decomposition import TruncatedSVD
data = [[1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]]
n_components = 2  # 潜在变量的个数
model = TruncatedSVD(n_components=n_components)
model.fit(data)
print(model.transform(data))  # 变换后的数据
print(model.explained_variance_ratio_)  # 贡献率
print(sum(model.explained_variance_ratio_))  # 累计贡献率
```

另外，与PCA一样，我们也可以检查LSA变换后的矩阵中包含多少原始信息。使用了scikit-learn的上述代码输出的累计贡献率约为0.67，表明这2个变量包含了约67%的原始数据的信息。
