"""
Visualize Genetic Algorithm to find a maximum point in a function.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds


def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): 
    # 适应度计算 就是每代留存的概率
    return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop): 
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    # nature selection wrt pop's fitness
    # 按照适者生存不适者淘汰 
    idx = np.random.choice(np.arange(POP_SIZE)
                           , size=POP_SIZE
                           , replace=True,
                           p=fitness/fitness.sum()
                           )

    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    # 交叉配对
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)   # choose crossover points
        # 一定概率将 parent的某些值 随机替换成 pop[i_]行的对应位置的值
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    # 变异
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

plt.ion()       # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    # 计算种群得到的值
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA

    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # GA part (evolution)
    # 计算适应度
    fitness = get_fitness(F_values)

    best_x =  pop[np.argmax(fitness),:]
    best_y = F_values[np.argmax(fitness)]
    print(f"Most fitted DNA/第{_+1}次迭代最适合的DNA: ", best_x)
    # 根据适应度选择留下的种群
    pop = select(pop, fitness)
    # 拷贝一份当前种群以迭代给下一次循环
    pop_copy = pop.copy()
    # 从当前种群中循环迭代每一行作为父节点
    for parent in pop:
        # 交叉一次。一定概率将 parent的某些值 随机替换成 pop[随机]行的对应[随机]位置的值
        child = crossover(parent, pop_copy)
        # 变异一次。
        child = mutate(child)
        parent = child       # parent is replaced by its child

print(best_x,best_y)
plt.ioff(); plt.show()