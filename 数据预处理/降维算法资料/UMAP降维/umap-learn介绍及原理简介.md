
# [UMAP:探索数据之源,发现Python世界中的无限可能](https://www.toutiao.com/article/7344054861623034377/?app=news_article&timestamp=1710062827&use_new_style=1&req_id=202403101727062252C5F7B7C5CD108877&group_id=7344054861623034377&share_token=7BD0C70B-2603-42F0-BD00-3D1AD00A15B3&tt_from=weixin_moments&utm_source=weixin_moments&utm_medium=toutiao_ios&utm_campaign=client_share&wxshare_count=1&source=m_redirect)


* UMAP（Uniform Manifold Approximation and Projection）是一种基于流形学习的非线性降维技术.它主要依据以下两个思想:

1. 局部连通性:在高维空间中,数据点之间的局部关系更为重要.UMAP 试图保留这些局部结构.
2. 均匀度:UMAP 还尝试保持数据在降维后的空间中的均匀分布.

# UMAP 的工作流程包括以下几个步骤:

1. 邻近度计算:对给定数据集中的每个数据点,计算其相邻点.这一步类似于 K 近邻算法.
2. 局部结构建模:通过高维数据的局部结构来表示数据点.这有助于捕获数据的非线性特征.
3. 优化降维:将高维数据映射到低维空间,使得数据点在该空间中保持局部连通性和均匀度.

**UMAP 的公式:**

* UMAP 使用了一种称为 Fuzzy-Simplicial Set（模糊单纯集）的数学概念来描述数据点之间的连接关系.具体来说,UMAP 中使用了一种称为联合最近邻图（UMAP graph）的数据结构,其中数据点之间的距离表示它们在高维空间中的局部相似性.
* UMAP 的优化目标是最小化一个称为 “cross-entropy” 的损失函数,用于衡量高维空间中数据点之间的相似度与低维空间中对应数据点之间的相似度之间的差异.这个损失函数定义如下:

$ [ CE = \sum_{i=1}^{N} \sum_{j=1}^{N} w_{ij} (\log(\frac{w_{ij}}{q_{ij}}) - w_{ij} + q_{ij}) ] $

```
其中:
• ( N ) 是数据点的数量
• ( w_{ij} ) 表示高维空间中数据点 ( i ) 和 ( j ) 之间的相似度
• ( q_{ij} ) 表示低维空间中对应数据点 ( i ) 和 ( j ) 之间的相似度
```

* UMAP 通过最小化这个损失函数来实现将高维数据映射到低维空间并保持数据点之间的局部结构.
* 假设我们有一个高维数据集,如手写数字数据集 MNIST,我们可以使用 UMAP 将其降维到2维,并进行可视化展示.这有助于我们理解数据的结构和模式.

```python
以下是一个简单的示例代码:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from umap import UMAP

# 加载手写数字数据集
digits = load_digits()
data = digits.data
labels = digits.target

# 使用 UMAP 进行降维
umap = UMAP(n_components=2)
data_umap = umap.fit_transform(data)

# 绘制降维后的数据
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.scatter(data_umap[labels == i, 0], data_umap[labels == i, 1], label=str(i), alpha=0.5)
plt.title("UMAP Projection of Digits Dataset")
plt.legend()
plt.show()
```

**输出：**

![](https://p3-sign.toutiaoimg.com/tos-cn-i-axegupay5k/b040a6cb5f694aa288a3402c4e21e649~noop.image?_iz=58558&from=article.pc_detail&lk3s=953192f4&x-expires=1711515732&x-signature=gPcr225YRrSB2B8YG3mKsYbcoSg%3D)

* 这段示例代码演示了如何使用 UMAP 对手写数字数据集进行降维和可视化.

```
安装库:
要安装 UMAP 库,可以使用 pip 命令:
pip install umap-learn 
和
pip install matplotlib
```

* 确保你的 Python 环境已经安装了 pip,并且运行上述命令将会下载并安装 UMAP 库和其依赖项.
* 安装完成后,你就可以在 Python 中导入并使用 UMAP 库来处理数据降维和可视化任务.

**UMAP 的优点:**

1. 保留局部结构:UMAP能够很好地捕捉数据的局部结构,尤其适用于非线性数据.
2. 高效性:相对于传统的降维方法如 t-SNE,UMAP 计算速度更快,尤其适合处理大型数据集.
3. 可扩展性:UMAP可以很容易地应用于各种领域,包括图像、文本和其他类型的数据.

**UMAP 的缺点:**

1. 参数调整:UMAP有一些参数需要调整,例如邻近点数和最小距离,不同的数据集可能需要不同的参数设置.
2. 随机性:UMAP的结果具有一定的随机性,每次运行可能会得到稍微不同的结果.
3. 理解复杂度:UMAP的原理相对复杂,不太容易理解其背后的数学原理,这可能对某些用户造成困扰.

* 总体而言,UMAP是一种功能强大的降维和数据可视化工具,特别适合处理非线性数据和大规模数据集.它在许多实际应用中已经证明了其有效性,并被广泛应用于各种领域.
