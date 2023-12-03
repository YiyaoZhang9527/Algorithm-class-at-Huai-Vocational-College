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
def PDE(DE, T, k=1):
    '''

    Args:
        DE:
        t:
        k:

    Returns:

    '''
    return np.exp((DE) / (k * T))


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
                        "T": 1, #系统的温度，系统初始应该要处于一个高温的状态 初始温度越高，且马尔科夫链越长，算法搜索越充分，得到全局最优解的可能性越大，但这也意味着需要耗费更多的计算时间
                        "T_min": 0, #温度的下限，若温度T达到T_min，则停止搜索
                        "r": 0.0001, #用于控制降温的快慢 值越小T更新越快，退出越快
                        "expr": 0, #初始解
                        "jump_max": np.inf, #最大回炉停留次数
                        "k":1 # k越小越不容易退出
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
    k = parameter["k"]
    while T > T_min:
        counter += 1
        new_expr = func.__next__()  # 迭代新解
        funcpath.append(new_expr)
        DE = DE_function(new_expr , expr)
        if DE <= 0:
            # 如果新解比假设初解或者上一个达标解要小，就更新解
            expr = new_expr
            # 跳出域值更新为0 
            jump_counter = 0
        elif DE > 0:
            # 如果新解比假设初解或者上一个达标解要大，就不更新解
            expr = expr
            if jump(DE, T,k):
                # 每更新一次T更新一次
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
                                               "r": 0.4,
                                               "expr": 0,
                                               "jump_max": 1000,
                                               "k":0.000001
                                           })
    print(expr)
    plt.figure(figsize=(16, 9))  # %%
    plt.plot(path, c='g')
    plt.plot(funcpath, c='r')
    plt.show()
    plt.close()
