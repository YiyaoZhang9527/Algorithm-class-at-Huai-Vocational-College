from hyperopt import hp
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# from OptMetrics import MyMetric
# from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import datasets
from MyLogColor import log,LogLevel
import time
from sklearn.metrics import precision_recall_curve,auc,f1_score,roc_curve,auc

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

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state = 42)


def ROC_AUC(test_y, proba):
    fpr,tpr,threshold = roc_curve(test_y, proba)
    roc_auc_ = auc(fpr,tpr)
    return roc_auc_


historical_metrics = []
historical_params = {}

boosters = ['gbtree','gblinear','dart']
sampling_methods = ['uniform','gradient_based']
tree_methods = ["auto","exact","approx","hist"]
refresh_leafs = [0,1]
# process_types = ["default","update"]#,"refresh","prune"]
grow_policys = ["depthwise","lossguide"]
# sample_types = ["uniform","weighted"]
normalize_types = ["tree","forest"]
rate_drops = []



param_grid_hp = {
    'boosters':hp.choice('boosters',boosters)
    ,"n_estimators":hp.quniform("n_estimators",50,1000,1)
    ,"learning_rate":hp.uniform("learning_rate",1e-5,1)
    ,"gamma":hp.quniform("gamma",0,100,1)
    ,"max_depth":hp.quniform("max_depth",6,200,1)
    ,"min_child_weight":hp.quniform("min_child_weight",0,100,1)
    ,"max_delta_step":hp.quniform("max_delta_step",0,100,1)
    ,"subsample":hp.uniform("subsample",0,1)
    # ,"sampling_method":hp.choice("sampling_method",sampling_methods)
    ,"colsample_bytree":hp.uniform("colsample_bytree",0,1)
    ,"colsample_bylevel":hp.uniform("colsample_bylevel",0,1)
    ,"colsample_bynode":hp.uniform("colsample_bynode",0,1)
    ,"lambda":hp.quniform("lambda",0,200,1)
    ,"alpha":hp.quniform("alpha",0,200,1)
    ,"tree_method":hp.choice("tree_method",tree_methods)
    # ,"scale_pos_weight":hp.uniform("scale_pos_weight",0,1000)
    ,"refresh_leaf":hp.choice("refresh_leaf",refresh_leafs)
    # ,"process_type":hp.choice("process_type",process_types)
    ,"grow_policy":hp.choice("grow_policy",grow_policys)
    ,"max_leaves":hp.quniform("max_leaves",0,10000,1)
    ,"max_bin":hp.quniform("max_bin",256,1000,1)
    ,"num_parallel_tree":hp.quniform("num_parallel_tree",1,100,1)   
}
# booster_dart_params = {
#     "sample_type":hp.choice("sample_type",sample_types)
#     ,"normalize_type":hp.choice("normalize_type",normalize_types)
#     ,"rate_drop":hp.uniform("rate_drop",0,1)
#     ,"one_drop":hp.quniform("one_drop",0,1000,1)
#     ,"skip_drop":hp.uniform("skip_drop",0,1)
# }

booster_gblinear_params = {
    
}

def PR_AUC(test_y,proba,pred):
    precision,recall,_ = precision_recall_curve(test_y,proba)
    f1 ,pr_auc = f1_score(test_y,pred),auc(recall,precision)
    return pr_auc

def hyperopt_objective(hyperopt_params): 
    params = {
        "objective":"binary:logistic"
        ,'boosters':hyperopt_params['boosters']
        ,"n_estimators":int(hyperopt_params["n_estimators"])
        ,"learning_rate":hyperopt_params["learning_rate"]
        ,"gamma":hyperopt_params["gamma"]
        ,"max_depth":int(hyperopt_params["max_depth"])
        ,"min_child_weight":int(hyperopt_params["min_child_weight"])
        ,"max_delta_step":int(hyperopt_params["max_delta_step"])
        ,"subsample":hyperopt_params["subsample"]
        ,"verbosity":0
        # ,"sampling_method":hyperopt_params["sampling_method"]
        ,"colsample_bytree":hyperopt_params["colsample_bytree"]
        ,"colsample_bylevel":hyperopt_params["colsample_bylevel"]
        ,"colsample_bynode":hyperopt_params["colsample_bynode"]
        ,"lambda":int(hyperopt_params["lambda"])
        ,"alpha":int(hyperopt_params["alpha"])
        ,"tree_method":hyperopt_params["tree_method"]
        ,"scale_pos_weight":(y_train==0).sum()/(y_train==1).sum()
        ,"refresh_leaf":hyperopt_params["refresh_leaf"]
        # ,"process_type":hyperopt_params["process_type"]
        ,"grow_policy":hyperopt_params["grow_policy"]
        ,"max_leaves":int(hyperopt_params["max_leaves"])
        ,"max_bin":int(hyperopt_params["max_bin"])
        ,"num_parallel_tree":int(hyperopt_params["num_parallel_tree"])   
    }
    # booster_dart_params = {
    #     "sample_type":hyperopt_params["sample_type"]
    #     ,"normalize_type":hp.choice("normalize_type",normalize_types)
    #     ,"rate_drop":hyperopt_params["rate_drop"]
    #     ,"one_drop":int(hyperopt_params["one_drop"])
    #     ,"skip_drop":hyperopt_params["skip_drop"]
    # }
    dtrain = xgb.DMatrix(x_train,label=y_train)
    clf = xgb.train(params=params
                   ,dtrain=dtrain
                   ,num_boost_round=100
                   ,evals=[(dtrain,"train")]
                   ,verbose_eval=False # 不显示训练信息就改False
                   # ,obj=logistic_obj
                   )
    dtest = xgb.DMatrix(x_val,label=y_val)
    xgboost_proba = clf.predict(dtest)
    # xgbosst_proba = np.nan_to_num(xgboost_proba,0)
    
    global NOW_FUC_RUN_ITER
    NOW_FUC_RUN_ITER += 1
    metric = ROC_AUC(y_val,xgboost_proba)
    historical_metrics.append(metric)
    historical_params.update({NOW_FUC_RUN_ITER-1:params})
    return - metric

def param_hyperopt(max_evals=100):

    #保存迭代过程
    trials = Trials()

    #设置提前停止
    early_stop_fn = no_progress_loss(100)

    #定义代理模型
    #algo = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=50)
    params_best = fmin(hyperopt_objective #目标函数
                       , space = param_grid_hp #参数空间
                       , algo = tpe.suggest #代理模型
                       #, algo = algo
                       , max_evals = max_evals #允许的迭代次数
                       , verbose=True
                       , trials = trials
                       , early_stop_fn = early_stop_fn
                      )

    #打印最优参数，fmin会自动打印最佳分数
    print("\n","\n","best params: ", params_best,
          "\n")
    return params_best, trials

NOW_FUC_RUN_ITER = 0
PARAMS_BEST, Trials = param_hyperopt(600)

historical_metrics = np.array(historical_metrics)
idx = np.argmax(historical_metrics)
params = historical_params[idx]
dtrain = xgb.DMatrix(x_train,label=y_train)
clf = xgb.train(params=params
               ,dtrain=dtrain
               ,num_boost_round=100
               ,evals=[(dtrain,"train")]
               ,verbose_eval=False # 不显示训练信息就改False
               # ,obj=logistic_obj
               )
dtest = xgb.DMatrix(x_val,label=y_val)
xgboost_proba = clf.predict(dtest)
print("测试优化结果",ROC_AUC(y_val,xgboost_proba))