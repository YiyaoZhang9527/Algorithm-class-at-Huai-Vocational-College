# -*- encoding: utf-8 -*-
'''
@File    :   kmeans_cupy.py
@Time    :   2020/11/26 14:32:35
@Author  :   DataMagician 
@Version :   1.0
@Contact :   408903228@qq.com
@Desc    :   None
'''

# here put the import lib


import cupy as cp


def norm_cuda(vector, p=2):
    '''
    向量的范数
    '''
    return (vector ** p).dot(cp.ones_like(vector)) ** (1 / p)


def difference_in_norms_cuda(vector1, vector2, p=3):
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
    return norm_cuda(vector1 - vector2, p=3)
    # return sum((x - y) ** p for (x, y) in zip(vec1, vec2)) ** (1 / p)
    # return np.linalg.norm(vec1 - vec2, ord=p)


def centroids_init_cuda(train_x,k=8):
    """
    初始化质心
    """
    '''
    m,n = train_x.shape # 训练集的维度大小
    max_row = train_x.max(axis=0).tolist() #行的最大值
    min_row = train_x.min(axis=0).tolist() #行的最小值
    centroids = cp.zeros((n,k))
    #print(centroids,max_row)
    for i in range(n):
        centroids[i] = cp.random.randint(int(min_row[i]),int(max_row[i]),k)
        '''

    return train_x[cp.random.choice(a=cp.arange(train_x.shape[0]), size=k, replace=False, p=None)]  # 生成指定的中心点




def closest_centroid_cuda(centroid, x, p=2):
    '''
    计算最近的质心，拿到距离最近的k的索引
    '''
    m, n = centroid.shape
    closest_dist = float("inf")

    for _ in range(m):
        vector = centroid[_] - x
        distance = (vector ** p).dot(cp.ones_like(vector)) ** (1 / p)  # difference_in_norms_cuda(centroid[_],x,p)
        if distance < closest_dist:
            closest_dist = distance
            index_ = _
        # print(distance)
    return index_


# centroid = centroids_init_cuda()
# print(closest_centroid_cuda(centroid,train_x[100]))


def clustering_cuda(centroid, train_x, p=2):
    """
    单步聚类,拿到每一类x的索引
    """
    m, n = train_x.shape
    classification = dict()
    for x_i in range(m):
        k_i = closest_centroid_cuda(centroid, train_x[x_i])
        if k_i in classification:
            classification[k_i].append(x_i)
        else:
            classification.update({k_i: [x_i]})
    return classification


# ks = centroids_init_cuda()
# print(clustering_cuda(ks,train_x[:30]))

def rearrange_centroids_cuda(train_x, clusters, p=2):
    '''
    重新计算质心
    '''
    m, n = train_x.shape
    n_clusters = (len(clusters))
    init_new_centroids_cuda = cp.zeros((n_clusters, n))

    for kNo, i in zip(clusters, range(n_clusters)):
        clusters_index = clusters[kNo]
        old_clusters = train_x[clusters_index]
        n_of_this_clusters = old_clusters.shape[0]
        new_clusters = old_clusters.T.dot(cp.ones(n_of_this_clusters)) / n_of_this_clusters
        init_new_centroids_cuda[i] = new_clusters
    return init_new_centroids_cuda


# ks = centroids_init_cuda()
# print(rearrange_centroids_cuda(train_x,clustering_cuda(ks,train_x)))


def Kmeans_cuda(train_x, y_lab, k, p=2, max_iter=10,centroids=False):
    if centroids != True:
        centroids = centroids_init_cuda(train_x,k)
    print("初始质心:", centroids.shape)
    rl = []
    for _ in range(max_iter):
        clusters = clustering_cuda(centroids, train_x, p)
        print("聚类后质心:", len(clusters), "迭代整体轮次:", _)
        previous = centroids
        centroids = rearrange_centroids_cuda(train_x, clusters, p)
        rl.append(centroids.sum() - previous.sum())
        if rl[-1] == 0:
            break
    print(_)
    return clusters#[y_lab[clusters[i]] for i in clusters], rl, _
