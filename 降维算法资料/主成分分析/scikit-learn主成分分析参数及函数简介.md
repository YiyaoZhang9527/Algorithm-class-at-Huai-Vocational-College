#### [视频！用最直观的方式告诉你：什么是主成分分析PCA](https://www.bilibili.com/video/BV1E5411E71z/?buvid=Z649C66007D32BFB470485890A98DF7DDD30&is_story_h5=false&mid=uL3MQa8LoOiDbD2ugbClFg%3D%3D&p=1&plat_id=116&share_from=ugc&share_medium=iphone&share_plat=ios&share_session_id=4890E34B-0B3C-4E95-A5A3-B87241EA63B6&share_source=COPY&share_tag=s_i&timestamp=1695213004&unique_k=nPVDTco&up_id=3051484)

[代码实现！主成分分析（PCA）详解](https://blog.csdn.net/weixin_60737527/article/details/125144416)

#### 参数说明

```javascript
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
```

复制

##### **n_components**

**int, float, None or str**

代表返回的主成分的个数，即数据降到几维

* n_components=2 代表返回前2个主成分
* 0 < n_components < 1 代表满足最低的主成分方差累计贡献率
* n_components=0.98，指返回满足主成分方差累计贡献率达到98%的主成分
* n_components=None，返回所有主成分
* n_components=‘mle’，将自动选取主成分个数n，使得满足所要求的方差百分比

##### **copy**

**bool类型, False/True 默认是True**

 **在运行的过程中，是否将原数据复制** 。降维过程中，数据会变动。copy主要影响：调用显示降维后的数据的方法不同。

* copy=True时，直接 fit_transform(X)，就能够显示出降维后的数据
* copy=False时，需要 fit(X).transform(X) ，才能够显示出降维后的数据

##### whiten

**bool类型，False/True 默认是False**

白化是一种重要的预处理过程，其目的是降低输入数据的冗余性，使得经过白化处理的输入数据具有如下性质：

* 特征之间相关性较低
* 所有特征具有相同的方差

##### **svd_solver**

**str类型，str {‘auto’, ‘full’, ‘arpack’, ‘randomized’}**

意义：定奇异值分解 SVD 的方法

* auto：自动选择
* full：传统意义上的SVD
* arpack：直接使用scipy库的sparse SVD实现
* randomized：适用于数据量大，维度多，且主成分比例低的PCA降维

#### 属性atttibutes

* components_：返回最大方差的主成分。
* explained_variance_：它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
* explained_variance_ratio_：它代表降维后的各主成分的方差值占总方差值的比例，比例越大，则越是重要的主成分。（主成分方差贡献率）
* singular_values_：返回所被选主成分的奇异值。实现降维的过程中，有两个方法:
* 特征值分解（需要是方阵，限制多，计算量大）
* 奇异值分解（任意矩阵，计算量小，PCA默认）
* mean_：每个特征的经验平均值，由训练集估计。
* n_features_：训练数据中的特征数。
* n_samples_：训练数据中的样本数量。
* noise_variance_：噪声协方差

#### 方法Methods

##### fit(self,X,Y=None)

模型训练，PCA是无监督学习，没有标签，所以Y是None

##### fit_transform(self,X,Y=None)

将模型和X进行训练，并对X进行降维处理，返回的是降维后的数据

##### get_covariance(self)

获得协方差数据

##### get_params(self,deep=True)

返回的是模型的参数

##### inverse_transform(self,X)

将降维后的数据转成原始数据，不一定完全相同

##### transform(X)

将数据X转成降维后的数据。当模型训练好后，对于新输入的数据，可以直接用transform方法来降维。

### 案例分析

```javascript
import numpy as np
import matplotlib.pyplot as  plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

# 导入数据
from sklearn.datasets.samples_generator import make_blobs
```

复制

#### 导入数据作图

学习如何利用sklearn自带的数据

```javascript
# 生成数据集
X,y = make_blobs(n_samples=10000,
                   n_features=3,
                   centers=[[3,3,3],[0,0,0],[1,1,1,],[2,2,2]],
                   cluster_std=[0.2,0.1,0.2,0.2],
                   random_state=9)
```

复制

```javascript
X
```

复制

```javascript
array([[ 2.38526096,  2.1109917 ,  2.23765695],
       [ 0.05761939, -0.0117989 , -0.03393958],
       [ 3.08207073,  3.19904227,  3.08774759],
       ...,
       [ 3.13804869,  2.86955308,  2.86443838],
       [ 2.95413001,  3.42508432,  3.28296407],
       [ 2.79195132,  2.94196066,  2.71457256]])
```

复制

```javascript
y
```

复制

```javascript
array([3, 1, 0, ..., 0, 0, 0])
```

复制

```javascript
fig = plt.figure()
ax = Axes3D(fig,rect=[0,0,1,1],elev=30,azim=20)
plt.scatter(X[:,0],X[:,1],X[:,2],marker='o')
```

复制

![NFIftU.png](https://ask.qcloudimg.com/http-save/yehe-5430171/x8phmglp1a.png)NFIftU.png

**使用的数据有4个簇**

#### **查看方差分布（不降维** ）

不降维，只对数据进行投影，保留3个属性

```javascript
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
```

复制

```javascript
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
```

复制

#### 2个属性

查看两个重要的属性：

* 各个主成分的方差占比
* 主成分的方差

```javascript
print(pca.explained_variance_ratio_)  # 降维后的各主成分的方差值占总方差值的比例
print(pca.explained_variance_)  # 降维后的各主成分的方差值
```

复制

```javascript
[0.98318212 0.00850037 0.00831751]   # 第一个特征占据绝大多数
[3.78521638 0.03272613 0.03202212]
```

复制

**结论：特征1占据绝大多数**

#### 降维处理

从3维降到2维，指定n_components=2.

```javascript
pca = PCA(n_components=2)  # 降到2维
pca.fit(X)

# 查看属性
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
```

复制

```javascript
[0.98318212 0.00850037]
[3.78521638 0.03272613]
```

复制

![NFTkrR.png](https://ask.qcloudimg.com/http-save/yehe-5430171/4f6etqkvwx.png)NFTkrR.png

```javascript
# 查看转化后的数据分布
new_X = pca.transform(X)   # 转化后的数据
plt.scatter(new_X[:,0],new_X[:,1],marker="o")
plt.show()  # 显示的仍然是4个簇
```

复制

![NFIb0x.png](https://ask.qcloudimg.com/http-save/yehe-5430171/2h751ibbwx.png)NFIb0x.png

**将数据投影到平面上仍然是4个簇**

#### 指定主成分占比

比如，想看主成分的占比在99%以上的特征

```javascript
pca = PCA(n_components=0.99)  # 指定阈值占比
pca.fit(X)
```

复制

```javascript
PCA(copy=True, iterated_power='auto', n_components=0.99, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
```

复制

```javascript
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

复制

```javascript
[0.98318212 0.00850037]
[3.78521638 0.03272613]
2
```

复制

第1个特征 和第2个特征之和为：0.9831+0.0085已经超过99%

#### MLE算法自行选择降维维度

```javascript
pca = PCA(n_components='mle')
pca.fit(X)
```

复制

```javascript
PCA(copy=True, iterated_power='auto', n_components='mle', random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
```

复制

```javascript
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

复制

```javascript
[0.98318212]
[3.78521638]
1
```

复制

第一个特征的占比已经高达98.31%，所以算法只保留了第1个特征
