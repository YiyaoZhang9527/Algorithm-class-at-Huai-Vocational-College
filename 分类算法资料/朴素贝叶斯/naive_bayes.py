# -*- encoding: utf-8 -*-
'''
@File    :   naive_bayes.py
@Time    :   2020/09/06 16:44:37
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
'''
# here put the import lib


import jieba
import numpy as np
import pandas as pd

def probability(A,a):
    n=A.shape[0]
    nm = np.cumproduct(np.array(A.shape))[0]
    return (A==a).T.dot(np.ones(n))/n

def set_conditions(data,*conditions):
    PAVector = [probability(data,conditions[i])[i] for i in range(len(conditions))]
    return np.array(PAVector)

def bayes(data,*conditions,**cond):
    '''
    data 输入数据矩阵
    *conditions 输入条件列
    **cond 输入判定条件
    返回贝叶斯推断概率
    '''
    m,n = data.shape
    conditions_lenght = len(conditions)
    ybool = data[:,-1]==cond['cond']
    result = 0
    for i in range((conditions_lenght)):
        tmp = ((data[:,i]==conditions[i])&ybool)[ybool].sum()/m
        result+=tmp
    return result

def naive_bayes(data,*conditions,**conds):
    '''
    data 输入数据矩阵
    *conditions 输入条件列
    **cond 输入判定条件
    返回：分类结果 ，贝叶斯推断概率向量
    '''
    conds = np.array(conds['conds'])
    expr = np.array([bayes(data,*conditions,cond=cond) for cond in conds])
    return conds[expr==expr.max()][0],expr


if __name__ == "__main__":
    table1 = pd.DataFrame({
    '身高':['高','高','中','中','矮','矮','矮','中']
    ,'体重':['重','重','中','中','轻','轻','中','中']
    ,'鞋码':['大','大','大','中','小','小','中','中']
    ,'性别':['男','男','男','男','女','女','女','女']
}) 
    data = table1.to_numpy()
    print(naive_bayes(data,'高','大','小',conds=['男','女']))

