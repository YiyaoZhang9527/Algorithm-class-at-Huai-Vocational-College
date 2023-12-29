

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

from DataAugmentation import SeqEqWeightedDifference
from BayesOptLightGBMClass import BayesOptLightGBM
from BayesOptGBDTClass import BayesOptGBDT
from MyLogColor import  log,LogLevel
from sklearn import datasets
# from MyMetrics import MyMetrics


cancer=datasets.load_breast_cancer()
x=cancer.data
y=cancer.target

def Rollover(x):
    x = x.astype(bool)
    x = ~x
    x = x.astype(int)
    return x
####TODO:将少数变成正例
y = Rollover(y)


# ###TODO：插入正例数据进行正例插值
# from DataAugmentation import SeqEqWeightedDifference
# sqwd = SeqEqWeightedDifference(positive_x
#                                ,metric_str='minkowski',ylab=1,p=3
#                                )
# new_positive = np.c_[sqwd.new_x,sqwd.new_y]
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42) 
# train_x = np.r_[train_x,sqwd.new_x]
# train_y = np.r_[train_y,sqwd.new_y]
# bol = BayesOptLightGBM(x,y,MAX_SHUFFLE=200
#                        ,Folds=0
#                        ,metrics_class="pr-auc"
#                        ,x_train=train_x,x_val=test_x
#                        ,y_train=train_y,y_val=test_y)
# # bol = BayesOptLightGBM(x,y,MAX_SHUFFLE=200)
# bol.run()
# log(f"所有搜索相关记录：{bol.Trials.trials[0]}",LogLevel.INFO)
# bol.test_params()

# ####TODO：不插值计算
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42) 
# bol = BayesOptLightGBM(x,y,MAX_SHUFFLE=200
#                         ,Folds=0
#                         ,metrics_class="roc-auc"
#                         ,EARLY_STOP_BAYES=300
#                         ,x_train=train_x,x_val=test_x
#                         ,y_train=train_y,y_val=test_y)
# # bol = BayesOptLightGBM(x,y,MAX_SHUFFLE=200)
# bol.run()
# log(f"所有搜索相关记录：{bol.Trials.trials[0]}",LogLevel.INFO)
# bol.test_params()

###TODO:权重寻优
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42) 
# bol = BayesOptLightGBM(x,y,MAX_SHUFFLE=100
#                         ,Folds=0
#                         ,metrics_class="all"
#                         #all=[1.pr_auc,2.roc_auc,3.accuracy,4.precision,5.recall,6.false_alarm,7.miss_rate,8.specificity,9.f1score]
#                         ,metrics_weight=[0,0.5,0.5,0,0,0,0,0,0]
#                         ,EARLY_STOP_BAYES=200
#                         ,NUM_EVALS=1000
#                         ,min_recall=0
#                         ,cost_wight=1
#                         # ,x_train=train_x,x_val=test_x
#                         # ,y_train=train_y,y_val=test_y
#                         )
# bol.run()
# log(f"所有搜索相关记录：{bol.Trials.trials[0]}",LogLevel.INFO)
# bol.test_params()

# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42) 
boG = BayesOptGBDT(x,y
                    ,MAX_SHUFFLE=100
                    ,Folds=0
                    ,metrics_class="all"
                    #all=[1.pr_auc,2.roc_auc,3.accuracy,4.precision,5.recall,6.false_alarm,7.miss_rate,8.specificity,9.f1score]
                    ,metrics_weight=[0,0.5,0.5,0,0,0,0,0,0]
                    ,EARLY_STOP_BAYES=200
                    ,NUM_EVALS=1000
                    ,min_recall=0
                    ,cost_wight=1
#                   # ,x_train=train_x,x_val=test_x
#                   # ,y_train=train_y,y_val=test_y
                    )
boG.run()
log(f"所有搜索相关记录：{boG.Trials.trials[0]}",LogLevel.INFO)
boG.test_params()