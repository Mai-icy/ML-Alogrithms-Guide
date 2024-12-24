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
