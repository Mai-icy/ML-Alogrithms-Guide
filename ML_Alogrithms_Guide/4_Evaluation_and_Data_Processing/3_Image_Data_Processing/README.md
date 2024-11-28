# 图像数据的转换处理

本节将说明如何将图像数据作为机器学习的输入数据。

## 直接将像素信息作为数值使用

- 灰度图像：是图像中的每个像素只表示亮度的图像

以灰度图形作为例子，以下是将图像数据转换为向量数据的例子。如果是简单的图像识别问题，就可以使用这种简单的转换建立一个具有一定精度的模型。

### 转换为一维的数据

将灰度按照顺序记录，得到的向量就是图片数据

![image.png](images/1.png)

这种转化丢失了二维的图像关系，失去了输入图像的重要信息

### 转化为二维的数据

有些模型会在保留图像的二维关系的前提下直接将其作为输入数据进行处理，比如在图像识别中常用的深度学习使用的就是像素的近邻像素的信息。 下面是将图像数据转换为向量数据的示例代码。代码中使用第三方Python包Pillow将图像（png）数据转换为了向量数据。

```python
from PIL import Image
import numpy as np
img = Image.open('mlzukan-img.png').convert('L')
width, height = img.size
img_pixels = []
for y in range(height):
    for x in range(width):
        # 通过getpixel获取指定位置的像素值
	      img_pixels.append(img.getpixel((x,y)))
print(img_pixels)
```

## 数据使用

以下是一个图像转化为向量数据后输入给机器学习模型的例子：

这里我们使用灰度手写数字数据建立一个模型，用于预测从0到9的10个数字。

```python
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
model = RandomForestClassifier()
model.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
expected = digits.target[n_samples // 2:]
predicted = model.predict(data[n_samples // 2:])
print(metrics.classification_report(expected, predicted))
```

从预测结果来看，使用了向量数据的模型成功地进行了高精度的预测。