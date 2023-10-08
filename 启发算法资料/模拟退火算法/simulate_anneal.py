# -*- encoding: utf-8 -*-
'''
@File    :   simulate_anneal.py
@Time    :   2020/10/28 12:45:28
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''

# here put the import lib

import numpy as np
import matplotlib.pyplot as plt


# ### $ 根据热力学的原理，在温度为T时，出现能量差为dE的降温的概率为P(dE)，表示为：$
# # $ P(dE) = exp( dE/(k*T) ) $
# ### $ 其中k是一个常数，exp表示自然指数，且dE<0(温度总是降低的)。这条公式指明了 $


# # 单变量退火
def PDE(DE, t, k=1):
    '''

    Args:
        DE:
        t:
        k:

    Returns:

    '''
    return np.exp((DE) / (k * t))


def DE_function(new, old):
    '''

    Args:
        new:
        old:

    Returns:

    '''
    return new - old


def jump(DE, T, k=1):
    '''

    Args:
        DE:
        T:
        k:

    Returns:

    '''
    return PDE(DE, T, k) > np.random.rand() and 0 or 1


def simulate_anneal(func,
                    parameter={
                        "T": 1,
                        "T_min": 0,
                        "r": 0.0001,
                        "expr": 0,
                        "jump_max": np.inf
                    }):
    '''

    Args:
        func:
        parameter:

    Returns:

    '''
    path, funcpath = [], []
    T = parameter["T"]  # 系统温度，初时应在高温
    T_min = parameter["T_min"]  # 最小温度值
    r = parameter["r"]  # 降温速率
    counter = 0
    expr = parameter["expr"]  # 假设初解
    jump_max = parameter["jump_max"]  # 最大冷却值
    jump_counter = 0
    while T > T_min:
        counter += 1
        new_expr = func.__next__()  # 迭代新解
        funcpath.append(new_expr)
        DE = new_expr - expr
        if DE <= 0:
            expr = new_expr
            jump_counter = 0
        elif DE > 0:
            expr = expr
            if jump(DE, T):
                T *= r
                jump_counter += 1
                if jump_counter > jump_max:
                    print("最大回炉冷却次数:", jump_counter)
                    return expr, path, funcpath
        path.append(expr)
        print("{}{}{}{}{}{}{}{}".format('系统温度:', T, ' 新状态:', expr, ' 迭代轮次:',
                                        counter, ' DE:', DE))

    return expr, path, funcpath


if __name__ == "__main__":

    def f():  # 待优化最小函数
        '''

        Returns:

        '''
        for x in np.random.randn(1000):
            yield x


    expr, path, funcpath = simulate_anneal(f(),
                                           parameter={
                                               "T": 1,
                                               "T_min": 0,
                                               "r": 0.11,
                                               "expr": 0,
                                               "jump_max": 1000
                                           })
    print(expr)
    plt.figure(figsize=(16, 9))  # %%
    plt.plot(path, c='g')
    plt.plot(funcpath, c='r')
    plt.show()
    plt.close()
