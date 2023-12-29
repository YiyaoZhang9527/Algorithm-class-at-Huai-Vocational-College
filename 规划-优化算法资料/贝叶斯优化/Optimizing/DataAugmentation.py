from scipy import spatial
from sko.GA import GA_TSP
import numpy as np
from sklearn.metrics import precision_recall_curve,auc,f1_score
import pandas as pd
from sklearn.model_selection import train_test_split


class SeqEqWeightedDifference():
    def __init__(self,positive_x,ylab=1
                 ,metric_str='cosine'
                 ,p=3,SIZE_POP=50,MAX_ITER=500,PROB_MUT=1):
        self.p = p
        self.ylab = ylab
        self.metric = metric_str
        self.SIZE_POP = SIZE_POP
        self.MAX_ITER = MAX_ITER
        self.PROB_MUT = PROB_MUT
        self.positive_x = positive_x
        self.m,self.n = self.positive_x.shape
        self.num_points = positive_x.shape[0]
        if self.metric == "minkowski":
            self.distance_matrix = spatial.distance.cdist(positive_x, positive_x,metric=self.metric,p=self.p)
        else:
            self.distance_matrix = spatial.distance.cdist(positive_x, positive_x,metric=self.metric)
        self.new_positive = self.SequentialEqualWeightedDifference(positive_x)
        self.new_x,self.new_y,self.best_points, self.best_distance ,self.ga_tsp= self.new_positive
    
    def cal_total_distance(self,routine):
        '''The objective function. input routine, return total distance.
        cal_total_distance(np.arange(num_points))
        '''
        num_points, = routine.shape
        return sum([self.distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])



    def SequentialEqualWeightedDifference(self,positive_x):
        ga_tsp = GA_TSP(func=self.cal_total_distance
                        , n_dim=self.num_points
                        , size_pop=self.SIZE_POP
                        , max_iter=self.MAX_ITER
                        , prob_mut=self.PROB_MUT)
        print("运行向量排序")
        best_points, best_distance = ga_tsp.run()
        print("排序完成")
        max_iter = best_points.size
        y_lab = np.full(max_iter-1,self.ylab)
        data_augmentation_matrix = np.zeros((max_iter-1,self.n))
        for i in range(max_iter-1):
            left_vector,right_vector = positive_x[best_points[i]], positive_x[best_points[i+1]]
            generated_x = (left_vector+right_vector)/2
            data_augmentation_matrix[i]=generated_x
        return data_augmentation_matrix,y_lab,best_points, best_distance,ga_tsp


if __name__ == '__main__':
    new_data = pd.read_csv('data/房地产-select.csv')
    y = new_data['label']
    del new_data['label']
    x = new_data
    # data = pd.concat([x,y],axis=1)

    x = pd.get_dummies(x)
    # 去除数据全部相同的列
    x = x.drop(x.columns[x.std()==0],axis=1)
    # 如果存在重复列或者字符列则无法填充
    Xfillna = x.fillna(x.mean())   
    #Xfillna = x.fillna(0)  
    XfillnaMean = Xfillna.mean()
    XfillnaStd = Xfillna.std()
    Xfillna = (Xfillna - XfillnaMean)/XfillnaStd
    names = list(Xfillna.keys())
    Xfillna = np.array(Xfillna)
    x = Xfillna
    m,n = x.shape
    y = np.array(y,dtype=int)
    
    
    xy = np.c_[x,y]
    x,y = xy[:,0:n],xy[:,n:n+1]
    y = np.ravel(y)
    positive_numpy = xy[np.ravel(xy[:,n:n+1]==1)]
    positive_x = positive_numpy[:,:n]
    positive_y = positive_numpy[:,n]
    
    ###TODO：插入正例数据进行正例插值
    sqwd = SeqEqWeightedDifference(positive_x
                                   ,metric_str='euclidean',ylab=1
                                   )
    new_positive = np.c_[sqwd.new_x,sqwd.new_y]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42) 
    train_x = np.r_[train_x,sqwd.new_x]
    train_y = np.r_[train_y,sqwd.new_y]