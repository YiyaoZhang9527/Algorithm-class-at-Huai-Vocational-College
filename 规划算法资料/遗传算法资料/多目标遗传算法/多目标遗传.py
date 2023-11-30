#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: yudengwu(余登武）
# @Date  : 2021/10/26
#@email:1344732766@qq.com
#============导入相关包========
import numpy as np
import platform
import matplotlib.pyplot as plt
system = platform.system()
if system == "Linux":
    plt.rcParams['font.sans-serif'] = ["AR PL UKai CN"] #["Noto Sans CJK JP"]
elif system == "Darwin":
    plt.rcParams['font.sans-serif'] = ["Kaiti SC"]
plt.rcParams['axes.unicode_minus'] = False
# import matplotlib as mpl
import matplotlib; matplotlib.use('TkAgg')
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

#==========定义两个目标函数=============
# 定义函数1
def function1(x):
    value = -(x+2) ** 2+2*x
    return value
# 定义函数2
def function2(x):
    value = -(x - 2) ** 2
    return value

#=========定义群体，并绘制初始解的分布图=====
pop_size = 10
max_gen = 100
# 迭代次数
#Initialization
min_x=-10
max_x=10
np.random.seed(10)#固定随机数种子，使每次生成的初始解集一样
solution=np.random.uniform(min_x,max_x,pop_size) #生成的初始解集


#函数1 对应的初始目标函数值
values1=map(function1,solution) #python中的map用法，可以对遍历一个列表如solution，然后将列表元素传入到函数function1。得到解。得到的格式为对象
values1=[i for i in values1] #因为上一步得到是对象格式，需要转换成列表格式

#函数2 对应的初始目标函数值
values2=map(function2,solution) #python中的map用法，可以对遍历一个列表如solution，然后将列表元素传入到函数function2。得到解。得到的格式为对象
values2=[i for i in values2] #因为上一步得到是对象格式，需要转换成列表格式

plt.scatter(values1,values2, s=20, marker='o')
for i in range(pop_size):
    plt.annotate(i, xy=(values1[i], values2[i]), xytext=(values1[i] - 0.05, values2[i] - 0.05),fontsize=18)
plt.xlabel('function1')
plt.ylabel('function2')
plt.title('解的分布示意图')
plt.show()

#=================快速非支配排序==============
'''

1.n[p]=0 s[p]=[] 
2.对所有个体进行非支配判断，若p支配q，则将q加入到S[p]中，并将q的层级提升一级。
  若q支配p，n[p]+1.
3.找出种群中np=0的个体，即最优解，并找到最优解的支配解集合。存放到front[0]中
4 i==0
5.判断front是否为空，若不为空，将front中所有的个体sp中对应的被支配个体数减去1，(存放np==0的解序号进front[i+1]);i=i+1,跳到2;
  若为空，则表明得到了所有非支配集合，程序结束
'''

values=[values1,values2] #解集【目标函数1解集，目标函数2解集...】
def fast_non_dominated_sort(values):
    """
    优化问题一般是求最小值
    :param values: 解集【目标函数1解集，目标函数2解集...】
    :return:返回解的各层分布集合序号。类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]] 其中[1]表示Pareto 最优解对应的序号
    """
    values11=values[0]#函数1解集
    S = [[] for i in range(0, len(values11))]#存放 每个个体支配解的集合。
    front = [[]] #存放群体的级别集合，一个级别对应一个[]
    n = [0 for i in range(0, len(values11))]#每个个体被支配解的个数 。即针对每个解，存放有多少好于这个解的个数
    rank = [np.inf for i in range(0, len(values11))]#存放每个个体的级别



    for p in range(0, len(values11)):#遍历每一个个体
        # ====得到各个个体 的被支配解个数 和支配解集合====
        S[p] = [] #该个体支配解的集合 。即存放差于该解的解
        n[p] = 0  #该个体被支配的解的个数初始化为0  即找到有多少好于该解的 解的个数
        for q in range(0, len(values11)):#遍历每一个个体
            less = 0 #的目标函数值小于p个体的目标函数值数目
            equal = 0 #的目标函数值等于p个体的目标函数值数目
            greater = 0 #的目标函数值大于p个体的目标函数值数目
            for k in range(len(values)):  # 遍历每一个目标函数
                if values[k][p] > values[k][q]:  # 目标函数k时，q个体值 小于p个体
                    less = less + 1  # q比p 好
                if values[k][p] == values[k][q]:  # 目标函数k时，p个体值 等于于q个体
                    equal = equal + 1
                if values[k][p] < values[k][q]:  # 目标函数k时，q个体值 大于p个体
                    greater = greater + 1  # q比p 差

            if (less + equal == len(values)) and (equal != len(values)):
                n[p] = n[p] + 1  # q比p,  比p好的个体个数加1

            elif (greater + equal == len(values)) and (equal != len(values)):
                S[p].append(q)  # q比p差，存放比p差的个体解序号

        #=====找出Pareto 最优解，即n[p]===0 的 个体p序号。=====
        if n[p]==0:
            rank[p] = 0 #序号为p的个体，等级为0即最优
            if p not in front[0]:
                # 如果p不在第0层中
                # 将其追加到第0层中
                front[0].append(p) #存放Pareto 最优解序号

    # =======划分各层解========

    """
    #示例，假设解的分布情况如下，由上面程序得到 front[0] 存放的是序号1
    个体序号    被支配个数   支配解序号   front
    1          0            2,3,4,5    0
    2,         1，          3,4,5
    3，        1，           4,5
    4，        3，           5
    5          4，           0

    #首先 遍历序号1的支配解，将对应支配解[2,3,4,5] ，的被支配个数-1（1-1,1-1,3-1,4-1）
    得到
    表
    个体序号    被支配个数   支配解序号   front
    1          0            2,3,4,5    0
    2,         0，          3,4,5
    3，        0，           4,5
    4，        2，           5
    5          2，           0

    #再令 被支配个数==0 的序号 对应的front 等级+1
    得到新表...
    """
    i = 0
    while (front[i] != []):  # 如果分层集合为不为空
        Q = []
        for p in front[i]:  # 遍历当前分层集合的各个个体p
            for q in S[p]:  # 遍历p 个体 的每个支配解q
                n[q] = n[q] - 1  # 则将fk中所有给对应的个体np-1
                if (n[q] == 0):
                    # 如果nq==0
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)  # 存放front=i+1 的个体序号

        i = i + 1  # front 等级+1
        front.append(Q)

    del front[len(front) - 1]  # 删除循环退出 时 i+1产生的[]

    return front #返回各层 的解序号集合 # 类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]]

front=fast_non_dominated_sort(values)


#=================打印结果=======================
#遍历各层
for i in range(len(front)):
    print('第%d层,解的序号为%s'%(i,front[i]))
    jie=[]
    for j in front[i]:#遍历第i层各个解
        jie.append(solution[j])
    print('第%d层,解为%s'%(i,jie))

