import numpy as np
import matplotlib.pyplot as plt
## github 原地址 https://github.com/MorvanZhou/Evolutionary-Algorithm/blob/master/tutorial-contents/Genetic%20Algorithm/Genetic%20Algorithm%20Basic.py

DNA_SIZE = 10 #DNA length
POP_SIZE = 100 #population size 种群多少个样本
CROSS_RATE = 0.8 #mating probability (DNA crossover) 交叉配对的比例
MUTATION_RATE = 0.003 # mutation probability 变异的强度比例
N_GENERATIONS = 200 # 迭代循环次数
X_BOUND = [0,5] # x upper and lower bounds x轴取值

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pred):
    #适应度
    """
    +1e-3是为了总和不等于0
    """

    return pred + 1e-3 - np.min(pred)

def translateDNA(pop):
    """
    """
    # return  pop.dot(2**np.arange(DNA_SIZE[::-1])/(2**DNA_SIZE-1*X_BOUND[1]))
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]

def seletc(pop,fitness):
    """

    """
    #生存淘汰
    idx = np.random.choice(np.arange(POP_SIZE)
                           ,size=POP_SIZE
                           ,replace=True # replace 代表的意思是抽样之后还放不放回去，如果是False的话，那么通一次挑选出来的数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了。
                           ,p=fitness/fitness.sum()) #p 每一个元素出现的概率是多少 -> 这里用 
    return pop[idx]

# def crossover(parent,pop):
#     """
#     parent 父母
#     pop 种群
#     """
#     #DNA交叉配对
#     if np.random.rand() < CROSS_RATE:
#         i_ = np.random.randint(0,POP_SIZE,size=1)
#         # 选点
#         # cross_points = np.random.randint(0,2,size=DNA_SIZE).astype(np.bool)
#         cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool) 
#         parent[cross_points] = pop[i_,cross_points]
#         return parent

def crossover(parent, pop):     # mating process (genes crossover)
    """
    parent 父母
    pop 种群
    """
    #     #DNA交叉配对
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent

def mutate(child):
    #变异
    """
    child: 子类
    """
    for point in range(DNA_SIZE):
        # 如果随机概率小于变异强度
        if np.random.rand() < MUTATION_RATE:
            # 将原本是1 的地方变成0本来是0变成1
            child[point] = 1 if child[point] == 0 else 0
    return child

pop = np.random.randint(0,2,(1,DNA_SIZE)).repeat(POP_SIZE,axis=0)
# pop [[]
#      []
#      []
#      []]

for _ in range(N_GENERATIONS):
    print(f"\n --------------------- \n第{_}轮迭代")
    F_value = F(translateDNA(pop))
    print(f"F_value:{F_value}")
    # 计算适应度
    fitness = get_fitness(F_value)
    # 选择种群
    pop = seletc(pop,fitness)
    pop_copy = pop.copy()
    for parent in pop:
        print(f"parent:{parent}")
        # DNA 交叉配对
        child = crossover(parent,pop)
        print(f"child1:{child}")
        # 变异
        child = mutate(child)
        # 更新种群内容为孩子
        print(f"child2:{child}")
        parent[:] = child



plt.ioff(); plt.show()