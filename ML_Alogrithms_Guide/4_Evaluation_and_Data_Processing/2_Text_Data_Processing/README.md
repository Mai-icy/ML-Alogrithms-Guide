# 文本数据的转换处理

对于自然语言处理，要对文本数据进行一定的处理。

## 基于单词出现次数的转换

对每个文本的每个单词进行计数。

建立一个表格，每一列表示一个单词，每一行表示一个文本。

对于一个单元格来说，代表的是这个文本中单词出现的个数。

以上方法是只考虑单词的出现次数，所有的单词并没有权重

### 基于tf-idf 的转换

- tf（term frequency，词频）：在文本中出现的频率
- idf（inverse document frequency，逆文本频率指数）：是一个包含该单词的文本越多，值就越小的值
- tf-idf：这两个指标相乘得到的结果

例如对于

文本1：“This is my car” 文本2：“This is my friend” 文本3：“This is my English book“

频率表：

|        | This | is   | my   | car  | friend | English | book |
| ------ | ---- | ---- | ---- | ---- | ------ | ------- | ---- |
| 文本 1 | 1    | 1    | 1    | 1    | 0      | 0       | 0    |
| 文本 2 | 1    | 1    | 1    | 0    | 1      | 0       | 0    |
| 文本 3 | 1    | 1    | 1    | 0    | 0      | 1       | 1    |

tf-idf转换后：

|        | This | is   | my   | car  | friend | English | book |
| ------ | ---- | ---- | ---- | ---- | ------ | ------- | ---- |
| 文本 1 | 0.41 | 0.41 | 0.41 | 0.70 | 0.00   | 0.00    | 0.00 |
| 文本 2 | 0.41 | 0.41 | 0.41 | 0.00 | 0.70   | 0.00    | 0.00 |
| 文本 3 | 0.34 | 0.34 | 0.34 | 0.00 | 0.00   | 0.57    | 0.57 |

行业术语和专有名词等只在特定文本中出现的词，往往具有较大的tf-idf值，有时能够很好地表示包含这些单词的文本。This 和 my等单词在每个文本中都出现了，所以它们的tf-idf值较小。

## 应用于机器学习模型

### 应用于机器学习模型

下面基于单词出现次数和 tf-idf 将文本数据转换为表格形式的特征数据，并应用于机器学习模型。

scikit-learn 的 `CountVectorizer` 可以计算单词出现次数，`TfidfVectorizer` 可以进行 tf-idf 的转换。

另外，通过 `fetch_20newsgroups` 获取使用的文本数据，机器学习模型则采用 `LinearSVC`。

### 示例代码

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups

categories = ['misc.forsale', 'rec.autos', 'comp.graphics', 'sci.med']
remove = ('headers', 'footers', 'quotes')
twenty_train = fetch_20newsgroups(subset='train',
                                  remove=remove,
                                  categories=categories)  # 训练数据
twenty_test = fetch_20newsgroups(subset='test',
                                 remove=remove,
                                 categories=categories)  # 验证数据
```

这里使用的是前面出现过的 20 Newsgroups 数据集，代码通过 `categories` 变量明确指定了 4 个主题的数据。

首先将文本数据转换为单词出现次数，然后使用 `LinearSVC` 学习和预测。得到的模型对验证数据的正确率约为 0.794。也就是说，我们成功地将文本数据转换为了表格形式的特征，并进行了机器学习。

### 示例代码

```python
count_vect = CountVectorizer()  # 单词出现次数
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_test_counts = count_vect.transform(twenty_test.data)

model = LinearSVC()
model.fit(X_train_counts, twenty_train.target)
predicted = model.predict(X_test_counts)
np.mean(predicted == twenty_test.target)
#  0.7937619350732018
```

接下来使用tf-idf进行转换，然后以同样的方式进行训练和预测。此时得到的模型的正确率约为0.87，比基于单词出现次数进行转换的方法的正确率高。通过tf-idf，看起来我们很好地抓住了文本数据的特点。

```python
tf_vec = TfidfVectorizer()  # tf-idf
X_train_tfidf = tf_vec.fit_transform(twenty_train.data)
X_test_tfidf = tf_vec.transform(twenty_test.data)
model = LinearSVC()
model.fit(X_train_tfidf, twenty_train.target)
predicted = model.predict(X_test_tfidf)
np.mean(predicted == twenty_test.target)
# 0.8701464035646085
```