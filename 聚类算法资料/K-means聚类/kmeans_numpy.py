# -*- encoding: utf-8 -*-
'''
@File    :   kmeans_numpy.py
@Time    :   2020/11/26 14:18:16
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
@Desc    :   None
'''

# here put the import lib
import numpy as np
import matplotlib.pyplot as plt


def norm(vector, p=6):
    '''
    向量的范数
    '''
    return (vector ** p).dot(np.ones_like(vector)) ** (1 / p)


def difference_in_norms(vector1, vector2, p=3):
    """
    闵氏距离
    当p=1时，就是曼哈顿距离
    当p=2时，就是欧氏距离
    当p→∞时，就是切比雪夫距离
    :param vec1:
    :param vec2:
    :param p:
    :return
    """
    # print(np.linalg.norm(vec1 - vec2, ord=p))
    return norm(vector1 - vector2, p)
    # return sum((x - y) ** p for (x, y) in zip(vec1, vec2)) ** (1 / p)
    # return np.linalg.norm(vec1 - vec2, ord=p)


def centroids_init(train_x=None, k=8):
    """
    初始化质心
    """
    '''
    m,n = train_x.shape # 训练集的维度大小
    max_row = train_x.max(axis=0) #行的最大值
    min_row = train_x.min(axis=0) #行的最小值
    return np.array([np.random.randint(min_row[i],max_row[i],k) for i in range(n)]).T #生成指定的中心点
    '''
    return train_x[np.random.choice(a=np.arange(train_x.shape[0]), size=k, replace=False, p=None)]


def closest_centroid(centroid, x, p=2):
    '''
    计算最近的质心，拿到距离最近的k的索引
    '''
    m, n = centroid.shape
    closest_dist = np.inf
    for _ in range(m):
        distance = difference_in_norms(centroid[_], x, p)
        if distance < closest_dist:
            closest_dist = distance
            index_ = _
    return index_


def clustering(centroid, train_x, p=2):
    """
    单步聚类,拿到每一类x的索引
    """
    m, n = train_x.shape
    classification = dict()
    for x_i in range(m):
        k_i = closest_centroid(centroid, train_x[x_i])
        if k_i in classification:
            classification[k_i].append(x_i)
        else:
            classification.update({k_i: [x_i]})
    return classification


def rearrange_centroids(train_x, clusters, p=2):
    '''
    重新计算质心
    '''
    m, n = train_x.shape
    n_clusters = (len(clusters))
    init_new_centroids = np.zeros((n_clusters, n))
    for kNo, i in zip(clusters, range(n_clusters)):
        clusters_index = clusters[kNo]
        old_clusters = train_x[clusters_index]
        n_of_this_clusters = old_clusters.shape[0]
        new_clusters = old_clusters.T.dot(np.ones(n_of_this_clusters)) / n_of_this_clusters
        init_new_centroids[i] = new_clusters
    return init_new_centroids


def Kmeans(train_x, k, p=2, max_iter=10):
    centroids = centroids_init(train_x, k)
    print("初始质心:", centroids.shape)
    rl = []
    for _ in range(max_iter):
        clusters = clustering(centroids, train_x, p)
        print("聚类后质心:", len(clusters), "迭代整体轮次:", _)
        previous = centroids
        centroids = rearrange_centroids(train_x, clusters, p)
        rl.append(centroids.sum() - previous.sum())
        if rl[-1] == 0:
            break
    print(_)
    return {"index": [clusters[i] for i in clusters], 'loss': rl, "time": _, "centroids": centroids,
            "centroids_size": centroids.shape[0]}


if __name__ == "__main__":
    train_x = np.random.normal(0, 1, 100).reshape(50, 2)
    print(Kmeans(train_x, k=4))

    index_of_kmearn = dict()
    for i in range(2):
        temp = Kmeans(train_x, k=12, p=3)
        index_s, loss, time, centroids, centroids_size = temp["index"], temp["loss"], temp["time"], temp["centroids"], \
                                                         temp["centroids_size"]
        if centroids_size in index_of_kmearn:
            index_of_kmearn[centroids_size].append(index_s)
        else:
            index_of_kmearn.update({centroids_size: index_s})
        plt.figure(figsize=(16, 9))
        for index_ in index_s:
            class_x = train_x[index_]
            x, y = class_x[:, 0], class_x[:, -1]
            plt.scatter(x, y)
        plt.show()
        plt.close()
