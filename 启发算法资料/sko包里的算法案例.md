首先导入库

```python
import sko
import matplotlib.pyplot as plt
import pandas as pd
```

定义一个函数

```python
import numpy as np
def pin_func(p):
    '''
    一个针状函数，取值范围为 0 <= x, y <= 100
    有一个最大值为1.1512在坐标(50,50)
    '''
    # 解析 p 向量，即 (x, y)
    x, y = p
    # 计算针状函数的结果
    r = np.square( (x-50)** 2 + (y-50)** 2 )+np.exp(1)
    t = np.sin(r) / r + 1
    # 转化为求最小值
    return -1 * t 

#一样的函数，返回值没有负号
def pin_fun(p):
    # 解析 p 向量，即 (x, y)
    x, y = p
    r = np.square( (x-50)** 2 + (y-50)** 2 )+np.exp(1)
    t = np.sin(r) / r + 1
    return t 
```

遗传算法

```python
from sko.GA import GA

#求目标函数最小值
o_GA = sko.GA.GA(func=pin_func,#目标函数 
                 n_dim=2,#函数中的参数维度 
                 size_pop=50,#种群初始规模
                 max_iter=800,#最大迭代次数 
                 prob_mut=0.001,#变异概率
                  lb=[0, 0,],#函数参数上限 
                 ub=[100, 100,],#函数参数下限 
                 precision=1e-7)#精度要求
#另外还有constraint_eq等式约束条件，适用于非线性约束，constraint_ueq不等式约束条件，适用于非线性约束

A_GA_x, A_GA_y = o_GA.run()#运行算法
print('X:', A_GA_x, '\n', 'Y', A_GA_y)#打印


#求目标函数最大值
B_GA = sko.GA.GA(func=pin_fun, n_dim=2, size_pop=50, max_iter=800, prob_mut=0.001,
                  lb=[0, 0,], ub=[100, 100,], precision=1e-7)
B_GA_x, B_GA_y = B_GA.run()
print('x:', B_GA_x, '\n', 'y:', B_GA_y)
```

粒子群算法

```python
导入粒子群算法
from sko.PSO import PSO

#默认求最大值
A_pso=PSO(func=pin_func,
          n_dim=2,
          pop=400,#粒子群规模
          max_iter=200,
          lb=[0,0,],
          ub=[100,100,],
          w=1,#惯性因子
          c1=2,#个体学习因子
          c2=2)#社会学习因子

A_pos_x,A_pos_y = A_pso.run()#运行算法
print('x:', A_pos_x, '\n', 'y:', A_pos_y)
```

差分进化算法

```python
#导入差分进化算法
from sko.DE import DE

A_DE = DE(func=pin_func, 
          n_dim=2, 
          size_pop=50, #种群初始规模
          max_iter=800,
          prob_mut=0.3#变异率
          lb=[0, 0,], 
          ub=[100, 100,])

A_DE_x, A_DE_y = A_DE.run()#运行算法
print('x:', A_DE_x, '\n', 'y:', A_DE_y)
```

模拟退火算法

```python
#导入模拟退火算法
from sko.SA import SA


A_sa = SA(func=pin_func, 
          x0=[1, 1], #目标函数的初始点
          T_max=1, #初始温度
          T_min=1e-9,#最小温度 
          L=300, #内循环迭代次数
          max_stay_counter=150,#最大停留次数
          lb=[0,0,],
          ub=[100,100,])

A_sa_x, A_sa_y = A_sa.run()#运行算法
print('x:', A_sa_x, '\n', 'y:', A_sa_y)
```

鱼群算法

```python
#导入鱼群算法
from sko.AFSA import AFSA

afsa = AFSA(func=pin_func, 
            n_dim=2, 
            size_pop=50,#种群初始规模 
            max_iter=300,
            max_try_num=100,#当某一代的最优解无法更新时，最多重复尝试的次数 
            step=5, #步长
            visual=0.3,#可视化界面的比例
            q=0.98, #温度调节的参数，用于控制降温速度
            delta=1)#更新解的相关参数，控制上一次最优解和当前最优解的距离

best_x, best_y = afsa.run()#运行算法
print(best_x, best_y)
```
