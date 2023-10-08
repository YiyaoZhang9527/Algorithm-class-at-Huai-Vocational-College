import numpy as np
import matplotlib.pyplot as plt

# 用来加载中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


def loadDataSet(filename):
    '''加载文件，将feature存在X中，y存在Y中'''
    X = []
    Y = []
    with open(filename, 'rb') as f:
        for idx, line in enumerate(f):
            line = line.decode('utf-8').strip()
            if not line:
                continue

            eles = line.split()
            if idx == 0:
                numFeature = len(eles)

            eles = list(map(float, eles))  # 将数据转换成float型

            X.append(eles[:-1])  # 除最后一列都是feature，append(list)
            Y.append([eles[-1]])  # 最后一列是实际值,同上

        return np.array(X), np.array(Y)  # 将X,Y列表转化成矩阵

def h(theta, X):
    '''
    定义模型函数
    :param theta:
    :param X:
    :return:
    '''
    print('h',X,theta)
    return np.dot(X, theta)  # 此时的X为处理后的X

def J(theta, X, Y):
    '''
    定义代价函数
    :param theta:
    :param X:
    :param Y:
    :return:
    '''
    m = len(X)
    return np.sum(np.dot((h(theta,X)-Y).T, (h(theta,X)-Y))/(2 * m))

def bgd(alpha # 学习率控制步长
        , maxloop # 最大迭代次数
        , epsilon # 阈值控制迭代
        , X #训练数据
        , Y #结果集
         ):
    '''
    定义梯度下降公式，其中alpha为学习率控制步长，maxloop为最大迭代次数，epsilon为阈值控制迭代（判断收敛）
    :param alpha:   学习率控制步长
    :param maxloop:     最大迭代次数
    :param epsilon:     阈值控制迭代
    :param X:   训练数据
    :param Y:   结果集
    :return: 求出误差最小参数 theta, 历史代价向量 costs
    '''
    m, n = X.shape  # m为样本数，n为特征数，在这里为2

    # 初始化参数为零
    theta = np.zeros((2, 1))

    count = 0  # 记录迭代次数
    converged = False  # 是否收敛标志
    cost = np.inf  # 初始化代价为无穷大
    costs = []  # 记录每一次迭代的代价值
    thetas = {0: [theta[0, 0]], 1: [theta[1, 0]]}  # 记录每一轮theta的更新

    while count <= maxloop: #当学习次数小于最大学习次数限制时，循环可以运行
        if converged: #如果收敛成个，则返回
            break
        # 更新theta
        count = count + 1 #每次+1计算学习次数

        # 单独计算,求两个参数的偏导数，进行梯度下降
        '''
        theta0 = 参数0 - 学习率 * (1 / 数据总长度的值) * sum(当前参数0，1代入假设函数临时得到的直线 - 真实结果的值)
        '''
        theta0 = theta[0, 0] - alpha / m * (h(theta, X) - Y).sum()
        theta1 = theta[1, 0] - alpha / m * (
            np.dot(X[:, 1][:, np.newaxis].T, (h(theta, X) - Y)) #这里的是矩阵对向量的叉乘
        ).sum()  # 重点注意一下
        # 同步更新
        theta[0, 0] = theta0
        theta[1, 0] = theta1
        thetas[0].append(theta0)
        thetas[1].append(theta1)

        '''
        print('待求参数(theta):\n', theta, '\n'
              , '学习率（alpha）:\n', alpha, '\n'
              , '长度（m）:\n', m, '\n'
              # ,'X:\n', X[0:5] ,'\n'
              # ,'Y:\n',Y[0:5] ,'\n'
              , '是否收敛(converged):\n', converged, '\n'
              , 'X^i:', X[:, 1][:, np.newaxis].T[0:5]
              , 'theta0:',theta[0, 0]
              , 'theta1:',theta[1, 0])
              '''

        # 更新当前代价变量的值cost
        cost = J(theta, X, Y)
        costs.append(cost)

        # 如果收敛，则不再迭代
        if cost < epsilon:
            converged = True
    return theta, costs, thetas


def liner(X  #预期值
          ,Y #真实值
          ,alpha = 0.000000002 #学习率
          ,maxloop = 3500 # 最大迭代次数
          ,epsilon = 0.001 # 收敛判断条件
     ):
    m, n = X.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)  # 将第一列为1的矩阵，与原X相连

    '''#调用梯度下降算法，求出误差最小参数 theta, 历史代价向量 costs , '''
    resault = bgd(alpha, maxloop, epsilon, X, Y)
    theta, costs, thetas = resault  # 最优参数保存在theta中，costs保存每次迭代的代价值，thetas保存每次迭代更新的theta值
    XCopy = X.copy()
    XCopy.sort(0)  # axis=0 表示列内排序

    '''调用假设函数求出线性拟合结果'''
    yHat = h(theta, XCopy)
    print('结果')
    return np.c_[X[:,1],np.squeeze(yHat)],thetas,costs,theta

def liner_plot(X #预期值
          ,Y #真实值
          ,alpha = 0.000000002 #学习率
          ,maxloop = 3500 # 最大迭代次数
          ,epsilon = 0.001 # 收敛判断条件
     ):
    plt.figure(figsize=(16, 9))
    resault = liner(X, Y , alpha , maxloop , epsilon)
    res = resault[0]
    plt.scatter(X, Y,c='g')
    plt.plot(res[:, 0:1], res[:, -1:],c='r')
    plt.show()
    plt.close()
    return resault

if __name__ == '__main__':
    X, Y = np.random.randn(100)[:, None], np.random.randn(100)[:, None]  # loadDataSet('./data/ex1.txt')
    print(X.shape)
    print(Y.shape)
    print(liner_plot(X,Y,alpha=0.000001,maxloop=1000,epsilon=0.1))
