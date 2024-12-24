## 示例代码

样本数据呈曲线分布，最近邻点k的数量采用默认值5.

``` python

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 生成数据
X, y = make_moons(noise=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = KNeighborsClassifier()
model.fit(X_train, y_train)  # 训练
y_pred = model.predict(X_test)
accuracy_score(y_pred, y_test)  # 评估

```
