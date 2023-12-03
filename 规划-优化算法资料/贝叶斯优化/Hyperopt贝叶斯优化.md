# [贝叶斯优化-Hyperopt](https://daihuidai.github.io/2019/09/16/%E8%B4%9D%E5%8F%B6%E6%96%AF%E4%BC%98%E5%8C%96-Hyperopt/)

 **发表于** 2019-09-16 **|**  **分类于** [机器学习 ](https://daihuidai.github.io/categories/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/)|  [0 ](https://daihuidai.github.io/2019/09/16/%E8%B4%9D%E5%8F%B6%E6%96%AF%E4%BC%98%E5%8C%96-Hyperopt/#comments)|  **阅读量** **152**次

*万壑松风知客来，摇扇抚琴待留声*

#### 1. 文起

本篇文章记录通过 Python 调用第三方库，从而调用使用了贝叶斯优化原理的 Hyperopt 方法来进行超参数的优化选择。具体贝叶斯优化原理与相关介绍将在下一次文章中做较为详细的描述，可以参考这里。

Hyperopt 是 Python 的几个贝叶斯优化库中的一个。它使用 Tree Parzen Estimator（TPE），其它 Python 库还包括了 Spearmint（高斯过程代理）和 SMAC（随机森林回归）。贝叶斯优化问题有四个部分：

1. 目标函数：使用的机器学习模型调用该组超参数在验证集上的损失。
2. 域空间：类似于网格搜索，传入超参数的搜索范围。
3. 参数搜索：构造替代函数并选择下一个超参数值进行评估的方法。
4. 存储结果：最小化函数在评估每组测试后的最优超参数存储结果。

#### 2. Hyperopt 简单样例

> 说明：最简单的流程，实现以 XGBoost 作为调参模型，通过 hyperopt 完成上述贝叶斯优化的四个部分。

**一：定义目标函数**

 |

```python
# 准备数据
train_X, test_X, train_y, test_y = train_test_split(df_scaler, df_y, test_size=0.3, random_state=999)
data_train =xgb.DMatrix(train_X, train_y, silent=False)

# 定义目标函数
def objective(params, n_folds=10):
    cv_results =xgb.cv(params, data_train, num_boost_round=1000, nfold=n_folds, stratified=False, shuffle=True, metrics='mae', early_stopping_rounds=10)
    mae = max(cv_results['test-mae-mean'])
    loss = mae
    return loss
```

|  |  |
| - | - |

objective() 是目标函数（黑盒函数），作用是返回一个损失值，Hyperopt 也是根据这个损失值来逐步选择超参数的。后续的 fmin() 函数会最小化这个损失，所以你可以根据是最大化还是最小化问题来改变这个返回值。此处的目标函数十分基础并没有做过多的调整，可根据实际情况来做修改目标函数。

**二：设置域空间**

```python
from hyperopt import hp
space = {
    'learning_rate': hp.loguniform('learning_rate',np.log(0.01),np.log(0.4)),
    'max_depth': hp.choice('max_depth',range(1,8,1)),
    'min_child_weight': hp.choice('min_child_weight',range(1,5,1)),
    'reg_alpha': hp.uniform('reg_alpha',0.0,1.0),
    'subsample': hp.uniform('subsample',0.5,1.0),
    'colsample_bytree': hp.uniform('colsample_bytree',0.6,1.0)
}
```

|  |  |
| - | - |

域空间，也就是给定超参数搜索的范围。这个可以通过 hyperopt 的 hp 方法实现，hp 方法有很多个参数选择的方式如：hp.loguniform（对数空间搜索）、hp.lognormal（对数正态分布）、hp.normal（正态分布）、hp.choice（列表选项搜索）、hp.uniform（连续均匀分布搜索）、hp.quniform（连续均匀分布整数搜索）。

这里也只是简单的设置域空间，比如还可以通过 choice 方法构建列表从而实现对多个不同模型的筛选，这个在后续会有介绍。

> 强调：如果使用参数名相同会报错 “hyperopt.exceptions.DuplicateLabel”，例如域空间中定义了 XGBoost、LightGBM 两种模型，由于它们的部分参数名相同所以如果使用同名参数将报错。[解决方法](https://github.com/hyperopt/hyperopt/issues/380)：可以修改内层名称，这个在后续代码中会说明。

**三：参数搜索**


```python
# 参数搜索算法
from hyperopt import tpe
tpe_algorithm = tpe.suggest

# 寻找目标函数的最小值
from hyperopt import fmin
MAX_EVALS = 500
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS)
```

|  |  |
| - | - |

Hyperopt 中可以使用几种搜索算法：tpe（tpe.suggest）、随机 rand（rand.suggest）、模拟退火 anneal（anneal.suggest）。同时定义用来寻找目标函数返回最小损失值的 fmin() 函数，其中 max_evals 参数表示迭代次数。

**四：存储结果**


```python
# 优化结果参数
print(best)
```

|  |  |
| - | - |

best 就是 Hyperopt 返回最佳结果的参数。

#### 3. Hyperopt 可以变复杂

上面使用几处简单的代码，实现了 Python 中调用 Hyperopt 来完成超参数选择的功能。任何方法都可以变得复杂，更何况 Hyperopt ，所以在上面代码的基础上稍加改变，实现一种稍微复杂点的 Hyperopt，从而完成较为强大的功能。

> 说明：以下代码将实现，从 SVM、XGBoost、LightGBM、KNN、Linear 等多个不同模型中选择超参数，最终找到 Hyperopt 认为的最优参数。其中损失值为偏离度，并可以查看黑盒函数中的运行过程。

**一：定义目标函数**


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from hyperopt import STATUS_OK
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
features = df.iloc[:,:-1]
target = df.iloc[:,-1]
scaler = StandardScaler()
features = scaler.fit_transform(features)
train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.25, random_state=999)

# 计算偏离度
def dod(real_values, pred_values):
    dod_result = np.around((np.absolute(real_values - pred_values) / real_values).mean(), 4)
    return dod_result

# 计算传入模型的损失
def objective(params):
    model_name = params['model_name']
    del params['model_name']
    if model_name == 'svm':
        clf = SVR(**params)
    elif model_name == 'xgboost':
        clf = xgb.XGBRegressor(**params)
    elif model_name == 'lightgbm':
        clf = lgb.LGBMRegressor(**params)
    elif model_name == 'knn':
        clf = KNeighborsRegressor(**params)
    elif model_name == 'linear':
        clf = LinearRegression(**params)
    else:
        return 0
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    loss = dod(test_y, pred)
#    偏离度是自己定义的方法，当然可以返回交叉验证的结果（loss是负值需要注意）
#    loss = cross_val_score(clf, train_X, train_y, cv=10, scoring='neg_mean_absolute_error').mean()
    return {'loss':loss, 'params':params, 'status':STATUS_OK}

# 定义总的目标函数
count = 0
best_score = np.inf
model_name = None
def fn(params):
    global model_name, best_score, count
    count +=1
    score = objective(params.copy())
    loss = score['loss']
    if loss < best_score:
        best_score = loss
        model_name = params['model_name']
    if count % 50 == 0:
        print('iters:{0}, score:{1}'.format(count, score))
    return loss
```

|  |  |
| - | - |

**二：设置域空间**

> 说明：两个树模型有相同的参数名，所以在设置参数时需要区别参数名。

```python
from hyperopt import hp

# 设置域空间
space = hp.choice('regressor_type',[
    {
        'model_name': 'svm',
        'C': hp.uniform('C',0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0)
    },
    {
        'model_name': 'xgboost',
        'n_estimators': hp.choice('xgb_n_estimators', range(50,501,2)),
        'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
        'max_depth': hp.choice('xgb_max_depth', range(2,8,1)),
        'min_child_weight': hp.choice('xgb_min_child_weight', range(1,5,1)),
        'reg_alpha': hp.uniform('xgb_reg_alpha', 0, 1.0),
        'subsample': hp.uniform('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.6, 1.0)
    },
    {
        'model_name': 'lightgbm',
        'n_estimators': hp.choice('lgb_n_estimators', range(50,501,2)),
        'learning_rate': hp.uniform('lgb_learning_rate', 0.01, 0.3),
        'max_depth': hp.choice('lgb_max_depth', range(2,8,1)),
        'num_leaves': hp.choice('lgb_num_leaves', range(20, 50, 1)),
        'min_child_weight': hp.choice('lgb_min_child_weight', [0.001,0.005,0.01,0.05,0.1]),
        'min_child_samples': hp.choice('lgb_min_child_samples', range(5,51,5)),
        'subsample': hp.uniform('lgb_subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('lgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha': hp.uniform('lgb_reg_alpha', 0, 1.0)
    },
    {
        'model_name': 'knn',
        'n_neighbors': hp.choice('n_neighbors', range(2,11)),
        'algorithm': hp.choice('algorithm', ['auto','ball_tree','kd_tree','brute'])
    },
    {
        'model_name': 'linear',
        'normalize': hp.choice('normalize', [False, True])
    }
])
```

|  |  |
| - | - |

**三：参数搜索**


```python
# 设定搜索算法
from hyperopt import tpe

# 如果有有必要，可以设置查看黑盒函数fn中的搜索情况（每次选择参数等）
from hyperopt import Trials
trials = Trials()

# 定义搜索目标函数最小值函数
from hyperopt import fmin
MAX_EVALS = 1500
best_params = fmin(fn=fn, space=space, algo=tpe.suggest, max_evals=MAX_EVALS,trials=trials)
print('model_name: {0}, best_score: {1}'.format(model_name, best_score))
```

|  |  |
| - | - |

**四：存储结果**

```python
# 最佳参数
print(best_params)

# 查看黑盒函数中的变化
print(trials)	# 可以通过循环依次打印结果
```

|  |  |
| - | - |

#### 4. 文末

Hyperopt 通过不同的设置可以变得更加完善和准确，从而在搜索参数上取得进一步成功。

你需要注意一点，Hyperopt 并不是一个完全没有缺陷的方法，它也可能从一开始就陷入局部最优，所以相比于暴力的网格搜索它只是一种具有数学理论支撑的高级暴力搜索方法。
