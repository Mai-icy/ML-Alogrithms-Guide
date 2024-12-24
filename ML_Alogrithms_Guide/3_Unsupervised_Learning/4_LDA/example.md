## 示例代码

``` python

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# 使用remove去除正文以外的信息
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
max_features = 1000
# 将文本数据变换为向量
tf_vectorizer = CountVectorizer(max_features=max_features,
stop_words='english')
tf = tf_vectorizer.fit_transform(data.data)
n_topics = 20
model = LatentDirichletAllocation(n_components=n_topics)
model.fit(tf)                                
print(model.components_)  # 各主题包含的单词的分布
print(model.transform(tf))  # 使用主题描述的文本

```

使用scikit-learn实现基于LDA的主题模型的创建。使用了一个名为20 Newsgroups的
数据集，这个数据集是20个主题的新闻组文本的集合，每个文本属于一个主题。
