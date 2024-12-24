## 示例代码

以下代码就是对之前温度和积雪预测的实例，最后输出了各种概率。

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
X_train = np.r_[np.random.normal(3, 1, size=50),
        np.random.normal(-1, 1, size=50)].reshape((100, -1))
y_train = np.r_[np.ones(50), np.zeros(50)]
model = LogisticRegression()
model.fit(X_train, y_train)
model.predict_proba([[0], [1], [2]])[:, 1]
#  array([ 0.12082515,  0.50296844,  0.88167486])
```
