import pandas as pd
import numpy as np
import xgboost as xgb
from sko.GA import GA
from sko.tools import set_run_mode
from sklearn.model_selection import train_test_split
from sklearn import metrics
from log_color import log,LogLevel
from tqdm import tqdm
import time
import datetime
import os
from sklearn import datasets


cancer=datasets.load_breast_cancer()
x=cancer.data
y=cancer.target

# from matplotlib import pyplot as plt
# train_df = pd.read_csv('./train_v2.csv', index_col=0)
# test_df =pd.read_csv('./test_v2.csv', index_col=0)
# print(train_df)
# x = train_df.drop(['user_id','merchant_id','label'],axis=1)
# y = train_df['label']


# ####TODO: 混合打乱数据
# y_to_numpy = y.to_numpy()
# x_to_numpy = x.to_numpy()
# m,n = x_to_numpy.shape
# y_to_numpy = y_to_numpy.reshape((m,1))
# xy = np.c_[x_to_numpy,y_to_numpy]
# for i in tqdm(range(50),desc="随机打乱数据"):
#     np.random.shuffle(xy)
# x,y = xy[:,0:13],xy[:,13:14]
# log(f"x.shape:{x.shape},y.shape:{y.shape}",LogLevel.INFO)

#### TODO:分割数据
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state = 42)
gamma = 0

####TODO:自动计算alpha值的取值范围 取负例的比例
# train_Y = y_train 
train_positive = (y_train==1).sum()
train_negative = (y_train==0).sum()
train_y_counter = y_train.size
alpha = train_negative/train_y_counter

log(f"""训练数据中,正例有【{train_positive}】个占比【{train_positive/train_y_counter}】
    ，负例有【{train_negative}】个占比【{train_negative/train_y_counter}】
    ，alpha值为【{alpha}】，""",LogLevel.INFO)

test_positive = (y_val==1).sum()
test_negative = (y_val==0).sum()
test_y_counter = y_val.size
log(f"""测试数据中,正例有【{test_positive}】个占比【{test_positive/test_y_counter }】
    ，负例有【{test_negative}】个占比【{test_negative/test_y_counter }】
    ，alpha值为【{test_negative/test_y_counter}】，""",LogLevel.INFO)


def XGBoostAUC(p):
    func_start = time.time()
    etas = [0.0001,0.001,0.01,0.1]
    sampling_methods = ["uniform","gradient_based"]
    # tree_methods = ["auto","exact","approx","hist"]
    w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16 = p
    params = { 
        "objective":"binary:logistic"   
        ,"learning_rate":w1 #0.1
        , "n_estimators":int(w2)#11 #即基评估器的数量。这个参数对随机森林模型的精确性影响是单调的，n_estimators越 大，模型的效果往往越好。但是相应的，任何模型都有决策边  n_estimators达到一定的程度之后，随机森林的 精确性往往不在上升或开始波动，并且，n_estimators越大，需要的计算量和内存也越大，训练的时间也会越来越 长。对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡。
        , "max_depth":int(w3) #构建树的深度，越大越容易过拟合
        , "min_child_weight":w4 #控制划分子节点的参数， #越大min_child_weight，算法越保守。范围：[0,无穷大] 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
#                , "num_class ":1#类别数，与 multisoftmax 并用
        , "gamma":w5 #损失下降多少才进行分裂， 控制叶子节点的个数
        , "subsample":w6#0.8 #随机采样训练样本
        #, "colsample_bytree":1 #生成树时进行的列采样
#                , "objective":'binary:logistic' # {'binary:logistic'}是二分类的问题，{'multi:softmax',}是多分类的问题 这个是优化目标，必须得有，因为xgboost里面有求一阶导数和二阶导数，其实就是这个。
        , "nthread":5 #cpu 线程数
        , "scale_pos_weight":train_negative/train_positive #负样本总数/正样本总数 。若训练负样本总数是500 ，正样本总数100，那么设置 scale_pos_weigh为 5
        , "lambda":w7#2 # 正则化参数
        , "eta":etas[int(w8)] #0.001 # 如同学习率
        , "verbosity":1 # 打印消息的详细程度。有效值为 0（静默）、1（警告）、2（信息）、3（调试）。
        , "eval_metric":"auc"
        , "seed":int(w9)
        , "max_delta_step":w10 #范围：[0,无穷大] ,我们允许每个叶子输出的最大增量步长。如果该值设置为0，则表示没有约束。如果将其设置为正值，则可以帮助使更新步骤更加保守。通常不需要此参数，但当类别极度不平衡时，它可能有助于逻辑回归。将其设置为 1-10 的值可能有助于控制更新。
        # ,"subsample":w11
        ,"sampling_method":sampling_methods[int(w11)]
        ,'colsample_bytree':w12 # 每次迭代中随机选择特征的比例
        , 'colsample_bylevel':w13 
        , 'colsample_bynode':w14
        ,"gpu_id":0 # 使用GPU的参数1
        ,"tree_method":"gpu_hist"#tree_methods[int(w16)]# #使用GPU的参数2 GPU参数做直方图排序
        ,"max_leaves":int(w15) #要添加的最大节点数。不被树方法使用
        ,"num_parallel_tree":int(w16) #每次迭代期间构建的并行树的数量。此选项用于支持增强随机森林
        }
    dtrain = xgb.DMatrix(x_train,label=y_train)
    clf = xgb.train(params=params
                    ,dtrain=dtrain
                    ,num_boost_round=100
                    ,evals=[(dtrain,"train")]
                    ,verbose_eval=False # 不显示训练信息就改False
                    #,obj=logistic_obj
                    )
    dtest = xgb.DMatrix(x_val,label=y_val)
    lr_proba = clf.predict(dtest)
    lr_proba = np.nan_to_num(lr_proba,0)
    fpr,tpr,threshold = metrics.roc_curve(y_val, lr_proba)
    roc_auc = metrics.auc(fpr,tpr)
    dtrain=None
    clf = None
    dtest = None
    lr_proba = None
    fpr,tpr,threshold = None,None,None

    func_end = time.time()
    ### 写入优化参数日志
    params_log_path = "./xgboost_opt_params_log_binary.csv"
    params_log = {k:[v] for k,v in params.items()}
    params_log.update({
        "roc_auc":[roc_auc]
    })
    params_log = pd.DataFrame(params_log)
    if os.path.exists(params_log_path):
        params_log.to_csv(params_log_path,mode='a', header=False, index=None)
    else:
        params_log.to_csv(params_log_path,index=None)

    global NOW_FUC_RUN_ITER
    # global SIZE_POP
    # global MAX_ITER
    NOW_FUC_RUN_ITER += 1
    log(f"""本次迭代AUC分数为:[{roc_auc}],
        用时:[{func_end-func_start}]秒,
        当前优化第:[{NOW_FUC_RUN_ITER}]次,
        已运行:[{NOW_FUC_RUN_ITER}]次，
        已运行:[{(NOW_FUC_RUN_ITER/((SIZE_POP+1)*MAX_ITER))*100}%]
        当前是:[{i+1}]第代种群
        用时总计:[{datetime.timedelta(seconds=(func_end-GA_start_time))}]秒,
        本次参数为:[{params}]""",LogLevel.PASS)

    return -roc_auc

#### TODO:设置遗传算法参数表
GA_params_table = pd.read_csv("GA_params_for_Xgboost.csv")
lbs = GA_params_table["lb"].to_list()
ubs = GA_params_table["ub"].to_list()
precisions = GA_params_table["precision"].to_list()
log(f"自变量上限:{lbs}，自变量下限:{ubs}，自变量精度:{precisions}",LogLevel.PASS)


GA_start_time = time.time()
NOW_FUC_RUN_ITER = 0
SIZE_POP = 100
MAX_ITER = 2

# #### TODO:矢量化，多线程和多进程，显卡内存不足，无法同时运行多个实例
# mode = 'vectorization','multithreading','multiprocessing'
# set_run_mode(XGBoostAUC,mode[-2])
ga = GA(func=XGBoostAUC
        , n_dim=16 #待求解的自变量数量
        , size_pop=SIZE_POP #种群初始化个体数量 
        , max_iter=MAX_ITER # 进化迭代次数
        , prob_mut=0.01 #变异概率
        , lb=lbs # 自变量下限
        ,ub=ubs # 自变量上限
        ,precision=precisions #精度
        #,early_stop = True # 当出现两个相同值时是否早停退出
       )
for i in range(MAX_ITER):
    best_x,best_y = ga.run(1)
    print('best_x:', best_x,'\n','best_y:',best_y)
    ### TODO:保存最佳xgboost参数表
    # best_x,best_y =  (0.339,35,6,55.7184,0.66666,0.8,57.869,0.1,59,6.6141,1,0.53333,1.0,1.0,30,26),[0.65]
    # w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17 = best_x
    w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16 = best_x
    etas = [0.0001,0.001,0.01,0.1]
    sampling_methods = ["uniform","gradient_based"]
    params = { 
        "objective":"binary:logistic"   
        ,"learning_rate":w1 #0.1
        , "n_estimators":int(w2)#11 #即基评估器的数量。这个参数对随机森林模型的精确性影响是单调的，n_estimators越 大，模型的效果往往越好。但是相应的，任何模型都有决策边  n_estimators达到一定的程度之后，随机森林的 精确性往往不在上升或开始波动，并且，n_estimators越大，需要的计算量和内存也越大，训练的时间也会越来越 长。对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡。
        , "max_depth":int(w3) #构建树的深度，越大越容易过拟合
        , "min_child_weight":w4 #0.8 #越大min_child_weight，算法越保守。范围：[0,无穷大] 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
    #                , "num_class ":1#类别数，与 multisoftmax 并用
        , "gamma":w5 #损失下降多少才进行分裂， 控制叶子节点的个数
        , "subsample":w6#0.8 #随机采样训练样本
        #, "colsample_bytree":1 #生成树时进行的列采样
    #                , "objective":'binary:logistic' # {'binary:logistic'}是二分类的问题，{'multi:softmax',}是多分类的问题 这个是优化目标，必须得有，因为xgboost里面有求一阶导数和二阶导数，其实就是这个。
        , "nthread":5 #cpu 线程数
        , "scale_pos_weight":train_negative/train_positive #负样本总数/正样本总数 。若训练负样本总数是500 ，正样本总数100，那么设置 scale_pos_weigh为 5
        , "lambda":w7#2 # 正则化参数
        , "eta":etas[int(w8)] #0.001 # 如同学习率
        , "verbosity":1 # 打印消息的详细程度。有效值为 0（静默）、1（警告）、2（信息）、3（调试）。
        , "eval_metric":"auc"
        , "seed":int(w9)
        , "max_delta_step":w10 #范围：[0,无穷大] ,我们允许每个叶子输出的最大增量步长。如果该值设置为0，则表示没有约束。如果将其设置为正值，则可以帮助使更新步骤更加保守。通常不需要此参数，但当类别极度不平衡时，它可能有助于逻辑回归。将其设置为 1-10 的值可能有助于控制更新。
        # ,"subsample":w11
        ,"sampling_method":sampling_methods[int(w11)]
        ,'colsample_bytree':w12
        , 'colsample_bylevel':w13
        , 'colsample_bynode':w14
        ,"gpu_id":0 # 使用GPU的参数1
        ,"tree_method":"gpu_hist"#tree_methods[int(w16)]# #使用GPU的参数2
        ,"max_leaves":int(w15) #要添加的最大节点数。不被树方法使用
        ,"num_parallel_tree":int(w16) #每次迭代期间构建的并行树的数量。此选项用于支持增强随机森林
        }
    best_xgboost_params_binary_path = "best_xgboost_params_binary.csv"
    params.update({"best_auc":best_y[0]})
    best_params_table = pd.DataFrame({k:[v] for k,v in params.items()})
    if os.path.exists(best_xgboost_params_binary_path):
        best_params_table.to_csv(best_xgboost_params_binary_path,mode='a', header=False, index=None)
    else:
        best_params_table.to_csv(best_xgboost_params_binary_path,index=False)



#### TODO:测试最优参数是否可用
best_xgboost_params_table = pd.read_csv("best_xgboost_params_binary.csv")
best_xgboost_params_table.drop(['best_auc'], axis=1, inplace=True)
xgboost_param_names = best_xgboost_params_table.columns
xgboost_param_values = best_xgboost_params_table.iloc[0].tolist()
xgboost_best_params = dict(zip(xgboost_param_names,xgboost_param_values)) 
dtrain = xgb.DMatrix(x_train,label=y_train)
clf = xgb.train(params=xgboost_best_params
                ,dtrain=dtrain
                ,num_boost_round=100
                ,evals=[(dtrain,"train")]
                ,verbose_eval=False # 不显示训练信息就改False
                # ,obj=logistic_obj
                )
dtest = xgb.DMatrix(x_val,label=y_val)
lr_proba = clf.predict(dtest)
lr_proba = np.nan_to_num(lr_proba,0)
fpr,tpr,threshold = metrics.roc_curve(y_val, lr_proba)
roc_auc = metrics.auc(fpr,tpr)
dtrain=None
clf = None
dtest = None
lr_proba = None
fpr,tpr,threshold = None,None,None
log(f"测试的最优参数AUC分数为:[{roc_auc}],本次参数为:[{xgboost_best_params}]",LogLevel.PASS)


