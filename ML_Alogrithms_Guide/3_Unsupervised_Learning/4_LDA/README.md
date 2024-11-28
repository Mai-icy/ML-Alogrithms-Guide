# LDA

## 概述

LDA（Latent Dirichlet Allocation）是一种降维的用于文本建模的算法。可根据文本中的单词找出潜在的主题并分类。

举个例子 

*  We go to school on weekdays.
*  I like playing sports.
*  They enjoyed playing sports in school.
*  Did she go there after school?
*  He read the sports columns yesterday

假设这些例句主题数为2，将其应用于LDA算法。
以下为主题A和主题B单词的概率分布：

![1.png](images/1.png)

school是主题A的代表性单词，sports是主题B的代表性单词

具体做法如下：

* 基于文本的主题分布为单词分配主题
* 基于分配的主题的单词分布确定单词
* 对所有文本中包含的单词执行步骤1和步骤2的操作

## 算法说明

LDA通过以下步骤计算主题分布和单词分布。

1. 为各文本的单词随机分配主题。
2. 基于为单词分配的主题，计算每个文本的主题概率。
3. 基于为单词分配的主题，计算每个主题的单词概率。
4. 计算步骤2和步骤 3中的概率的乘积，基于得到的概率，再次为各文本的单词分配主题。
5. 重复步骤2 ~ 步骤4的计算，直到收敛。

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

## 详细说明

### 使用主题描述文本

一些主题能直观地通过包含的单词来概述文本，比如 game team year games season play hockey players league win teams 是关于体育的。

一些只包含数值或概括性不强的单词的主题则通过停用词（为了提高精度而排除在外的单词）来进行改进。

![2.png](images/2.png)

以上是文本的主题分布图，可以明显直观地看出是主题18的文本。