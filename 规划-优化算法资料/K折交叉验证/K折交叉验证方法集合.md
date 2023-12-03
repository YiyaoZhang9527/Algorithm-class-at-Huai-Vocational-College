[模型评估_方法_交叉验证法](https://zhuanlan.zhihu.com/p/441133806)


# 【模型评估_方法_交叉验证法】

[![Wsilenceyezi](https://pic1.zhimg.com/v2-abed1a8c04700ba7d72b45195223e0ff_l.jpg?source=32738c0c)](https://www.zhihu.com/people/di-xie-50)

[Wsilenceyezi](https://www.zhihu.com/people/di-xie-50)

数据分析

4 人赞同了该文章

目录

收起

1、交叉验证

2、交叉验证分类

3、代码实现

1）KFold（k折交叉验证）

2) StratifiedKFold（分层采样k折交叉验证）
3) ShuffleSplit（随机分组交叉验证）
4) GroupKFold（分组k折交叉验证）
5) RepeatedKFold（重复k折交叉验证）
6) 留p法
7) 其他

11）cross_val_score

12）cross_validate

13）GridSearchCV

上篇文章我们学习了模型评估的方法：留出法、自助法，接下来我们学习交叉验证法~

## **1、交叉验证**

先将数据集D划分为k个大小相似的互斥子集，

![](https://pic2.zhimg.com/80/v2-ecbb730bbd1df542d708b3ba2b6c76b1_720w.webp)

然后，每次用k-1个子集的并集作为训练集，剩下的一个子集作为测试集，这样就可以获得k组训练/测试集，从而可进行k次训练和测试，最终返回这k个测试结果的均值。k常用取值是10，即10折交叉验证，下面是10折示意图：

![](https://pic1.zhimg.com/80/v2-3d2be5efa12c751df4e841c7bcf49eec_720w.webp)

**注：**

* **分布一致性：** 每个子集都尽可能保持数据分布一致（分层采样）；
* **多次随机、重复实验：** 由于数据集D划分为k个子集与留出法一样，同样存在多种划分方式，为了减少这种因样本划分不同而引入的误差，通常要随机划分p次数据集，最终结果是这p次k折交叉验证结果的均值；
* **留一法：** k=1，此时不受数据的随机划分的影响，但是当样本量较大时，计算成本会非常大。

## **2、交叉验证分类**

![](https://pic2.zhimg.com/80/v2-62ad291ee9e931e3ab10d8213316e4cd_720w.webp)

## **3、代码实现**

### **1）KFold（k折交叉验证）**

```python3
##提取数据
from pandas import DataFrame as df
from sklearn.datasets import load_breast_cancer
data_ori = df(load_breast_cancer().data, columns=load_breast_cancer().feature_names)[-12:].reset_index()
data_ori['target'] = load_breast_cancer().target[-12:]
########## KFold（K折交叉验证）
from sklearn.model_selection import KFold 
kf = KFold(n_splits=6, shuffle=True)
'''
n_splits=6（默认5）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果
shuffle=True（默认False）：每次划分前数据重新洗牌，每次的运行结果不同；shuffle=False：每次运行结果相同，相当于random_state=整数
  可以单独设置shuffle，但是设置random_state必须设置shuffle=True
random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
'''
for train_index,test_index in kf.split(data_ori):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
```

![](https://pic4.zhimg.com/80/v2-8b068d13d1feb0362da57bfab13eb59b_720w.webp)

### **2) StratifiedKFold（分层采样k折交叉验证）**

```text
########## StratifiedKFold
from sklearn.model_selection import StratifiedKFold
strkf = StratifiedKFold(n_splits=6, shuffle=False)
'''
n_splits=6（默认5）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果

shuffle=True（默认False）：每次划分前数据重新洗牌，每次的运行结果不同；shuffle=False：每次运行结果相同，相当于random_state=整数
random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
'''
for train_index,test_index in strkf.split(data_ori,data_ori['target']):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
    ## 数据集、训练集、测试集数据分布
    data_pct = data_ori['target'].sum()/len(data_ori['target'])
    y_train_pct = y_train.sum()/len(y_train)
    y_test_pct = y_test.sum()/len(y_test)
    print(data_pct, y_train_pct, y_test_pct)
```

![](https://pic1.zhimg.com/80/v2-81ecf627a7b3a84418387dcc4cad65ac_720w.webp)

### **3) ShuffleSplit（随机分组交叉验证）**

```text
from sklearn.model_selection import ShuffleSplit
ShuffleS = ShuffleSplit(n_splits=6, test_size=0.4, random_state=1)
'''
将数据顺序打散之后再进行交叉验证
n_splits=6（默认10）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果
test_size=0.4：测试集占比为40%
random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
'''
for train_index,test_index in ShuffleS.split(data_ori):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
```

![](https://pic4.zhimg.com/80/v2-3d525af6030119f26c9e30587dde9f5b_720w.webp)

### **4) GroupKFold（分组k折交叉验证）**

可以按照自己的需求来进行分组分组

```text
########## GroupKFold（分组k折交叉验证）
from sklearn.model_selection import GroupKFold
groupkf = GroupKFold(n_splits=3)
groups = [1,1,1,1,1,1,2,2,2,2,3,3]
'''
* 同一组的样本不可能同时出现在同一折的测试集和训练集中。
* groups的分组数必须大于等于n_splits的值
'''
for train_index,test_index in groupkf.split(data_ori, groups=groups):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
```

![](https://pic2.zhimg.com/80/v2-6491b8703421a3b1acdea6146d0def2d_720w.webp)

### **5) RepeatedKFold（重复k折交叉验证）**

```text
########## RepeatedKFold（重复n次k折交叉验证 ，每次重复有不同的随机性）
from sklearn.model_selection import RepeatedKFold
rptkf = RepeatedKFold(n_splits=6, n_repeats=2, random_state=1)
''' 
n_splits=6（默认5）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果n_repeats=2(默认10)：重复2次10折交叉验证
random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
'''
for train_index,test_index in rptkf.split(data_ori):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
```

![](https://pic2.zhimg.com/80/v2-5ec9a126b567090a47d835239f5657c5_720w.webp)

**6) RepeatedStratifiedKFold（重复分层采样k折交叉验证）**

```text
########## RepeatedStratifiedKFold（重复n次分层采样的k折交叉验证 ，每次重复有不同的随机性）
from sklearn.model_selection import RepeatedStratifiedKFold
rptstrkf = RepeatedStratifiedKFold(n_splits=6, n_repeats=2, random_state=1)
''' 
n_splits=6（默认5）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果

n_repeats=2(默认10)：重复2次10折交叉验证
random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
'''
for train_index,test_index in rptstrkf.split(data_ori, data_ori['target']):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
    ## 数据集、训练集、测试集数据分布
    data_pct = data_ori['target'].sum()/len(data_ori['target'])
    y_train_pct = y_train.sum()/len(y_train)
    y_test_pct = y_test.sum()/len(y_test)
    print(data_pct, y_train_pct, y_test_pct)  
```

![](https://pic4.zhimg.com/80/v2-92f37ddc4c22a111dc6d235581136a27_720w.webp)

**6) 留一法**

```text
########## 留一法
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index,test_index in loo.split(data_ori):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
```

![](https://pic2.zhimg.com/80/v2-5c1a6504c3d7dadaa97c43f94b0d92e1_720w.webp)

### 7) **留p法**

```text
########## 留P法
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=1)
# p=1 :留1
for train_index,test_index in lpo.split(data_ori):
    print(train_index,test_index)
    x_train = data_ori.iloc[train_index,:]
    x_test = data_ori.iloc[test_index,:]
    y_train = data_ori['target'][train_index]
    y_test = data_ori['target'][test_index]
```

### **8) 其他**

GroupShuffleSplit、 StratifiedShuffleSplit、 StratifiedGroupKFold、 LeaveOneGroupOut、 LeavePGroupsOut性质与上面类似，这里不再详细介绍啦~

**以上方法只是划分了数据集，而cross_val_score是根据模型计算交叉验证的结果，可以理解为cross_val_score中调用了KFold进行数据集划分**

---

### **11）cross_val_score**

```text
########## cross_val_score
## 提取数据
from pandas import DataFrame as df
from sklearn.datasets import load_breast_cancer
data_ori = df(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
data_ori['target'] = load_breast_cancer().target

## 建模
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=1)

##>> cv 为数值
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(estimator=lr, X=data_ori.iloc[:,:-1], y=data_ori['target'], cv=3, scoring='accuracy') #scoring（默认accuracy）：评价函数
print(scores, scores.mean()) # [0.93157895 0.96842105 0.92592593] 0.9419753086419753
   

##>> cv 为交叉验证方法
# KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(estimator=lr, X=data_ori.iloc[:,:-1], y=data_ori['target'], cv=kf, scoring='accuracy') 
print(scores, scores.mean()) # [0.94210526 0.93157895 0.96825397] 0.9473127262600948

# ShuffleSplit
from sklearn.model_selection import ShuffleSplit
ShuffleS = ShuffleSplit(n_splits=3, random_state=1)
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(estimator=lr, X=data_ori.iloc[:,:-1], y=data_ori['target'], cv=ShuffleS, scoring='accuracy') 
print(scores, scores.mean()) # [1.         0.96491228 0.96491228] 0.9766081871345028
```

### 12）**cross_validate**

```text
########## cross_validate（多个评价指标）
from sklearn.model_selection import cross_validate
scoring = ['accuracy','f1'] 
scores = cross_validate(estimator=lr, X=data_ori.iloc[:,:-1], y=data_ori['target'], cv=3, scoring=scoring)
print(scores['test_accuracy'], scores['test_f1'])  # [0.93157895 0.96842105 0.92592593] [0.94693878 0.97520661 0.93965517]
print(scores['test_accuracy'].mean(), scores['test_f1'].mean()) # 0.9419753086419753 0.953933519831415
```

**以上两种交叉验证评估指标区别：**

![](https://pic3.zhimg.com/80/v2-13a01eb3dc015aeefc8623dc8a4abfde_720w.webp)

### **13）GridSearchCV**

```text
########## GridSearchCV
##提取数据
import numpy as np
from pandas import DataFrame as df
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

##提取数据
data_ori = df(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
data_ori['target'] = load_breast_cancer().target
x_train, x_test, y_train, y_test = train_test_split(data_ori.iloc[:,:-1], data_ori['target'], test_size=0.2, random_state=1)
 
## GridSearchCV
DT = DecisionTreeClassifier()
param = {'max_depth':[3,4,5], 'min_samples_leaf':[2,3,4], 'min_impurity_decrease':[0.1,0.2]}
DTGrid = GridSearchCV(estimator=DT, param_grid=param, cv=3, scoring='accuracy')
DTGrid.fit(x_train,y_train)

## 交叉结果表
means = DTGrid.cv_results_['mean_test_score']
stds = DTGrid.cv_results_['std_test_score']
params = DTGrid.cv_results_['params']
for i, j,k in zip(means,stds,params):
    print('mean:{:.6f}'.format(i), 'std:{:.6f}'.format(j), 'param:',k)
  
## 最优参数及分数
print(DTGrid.best_params_) # 最优参数{'max_depth': 3, 'min_impurity_decrease': 0.1, 'min_samples_leaf': 2}
print(DTGrid.best_score_) # 最优参数对应的分数（默认使用的accuracy）0.9033490182409666
```

![](https://pic1.zhimg.com/80/v2-f14900c730bd1762aae042eab806c28c_720w.webp)
