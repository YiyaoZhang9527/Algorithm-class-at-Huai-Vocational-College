from hyperopt import hp
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss
# from OptMetrics import MyMetric
from sklearn.ensemble import GradientBoostingClassifier
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

losss = [ 'deviance', 'exponential']
criterions = ['friedman_mse', 'squared_error', 'absolute_error']
Max_features = ['auto', 'sqrt', 'log2'
                # ,hp.randint("max_features_int",0,x_train.shape[-1])
                ,hp.uniform("max_features_float",0,1)]
warm_starts = [True,False]
param_grid_hp = {
    "loss":hp.choice("loss",losss)
    ,"learning_rate":hp.uniform("learning_rate",0,1)
    ,'n_estimators': hp.quniform("n_estimators",10,1000,1)
    ,'subsample':hp.uniform("subsample",0,1)
    ,"criterion":hp.choice("criterion",criterions)
    ,"min_samples_leaf":hp.uniform("min_samples_leaf",0,0.5)
    ,"min_samples_split":hp.uniform("min_samples_split",0,1)
    ,"min_weight_fraction_leaf":hp.uniform("min_weight_fraction_leaf",0,0.5)
    ,"max_depth":hp.quniform("max_depth",1,1000,1)
    ,"min_impurity_decrease":hp.uniform("min_impurity_decrease",0,1)
    # ,"min_impurity_split":hp.uniform("min_impurity_split",0,1)
    ,"random_state":hp.randint("random_state",100)
    ,"max_features":hp.choice("max_features",Max_features)
    ,"max_leaf_nodes":hp.quniform("max_leaf_nodes",2,1000,1)
    ,"warm_start":hp.choice("warm_start",warm_starts)
}

def PR_AUC(test_y,proba,pred):
    precision,recall,_ = precision_recall_curve(test_y,proba)
    f1 ,pr_auc = f1_score(test_y,pred),auc(recall,precision)
    return pr_auc

def hyperopt_objective(params):  
    # log(f"本次参数:{params}",LogLevel.INFO)
    rfc = GradientBoostingClassifier(
        loss = params["loss"]
        ,learning_rate = params["learning_rate"]
        ,n_estimators= int(params["n_estimators"])
        ,subsample = params["subsample"]
        ,criterion = params["criterion"]
        ,min_samples_leaf = params["min_samples_leaf"]
        ,min_samples_split = params["min_samples_split"]
        ,min_weight_fraction_leaf = params["min_weight_fraction_leaf"]
        ,max_depth = int(params["max_depth"])
        ,min_impurity_decrease = params["min_impurity_decrease"]
        # ,min_impurity_split = params["min_impurity_split"]
        ,random_state = params["random_state"]
        ,max_features = params["max_features"]
        ,max_leaf_nodes = int(params["max_leaf_nodes"])
        ,warm_start = params["warm_start"]
    )
    rfc =rfc.fit(x_train, y_train)
    rfc_proba = rfc.predict_proba(x_test)[:,1]
    rfc_pred = rfc.predict(x_test)
    metric = PR_AUC(y_test,rfc_proba,rfc_pred)
    return -metric

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

def parsing_bayes_params_for_RandomForest(params):
    if int(params["max_features"]) - params["max_features"]== 0:
        max_features_params = Max_features[int(params["max_features"])]
    else:
        max_features_params = params["max_features"]
    
    return {"loss":losss[int(params["loss"])]
        ,"learning_rate":params["learning_rate"]
        ,"n_estimators": int(params["n_estimators"])
        ,"subsample": params["subsample"]
        ,"criterion" : criterions[int(params["criterion"])]
        ,"min_samples_leaf" : params["min_samples_leaf"]
        ,"min_samples_split" :  params["min_samples_split"]
        ,"min_weight_fraction_leaf" : params["min_weight_fraction_leaf"]
        ,"max_depth" : int(params["max_depth"])
        ,"min_impurity_decrease": params["min_impurity_decrease"]
        # ,min_impurity_split = params["min_impurity_split"]
        ,"random_state" : params["random_state"]
        ,"max_features ": max_features_params
        ,"max_leaf_nodes ": int(params["max_leaf_nodes"])
        ,"warm_start" : warm_starts[params["warm_start"]]
        }
    
    
    

def hyperopt_validation(params):
    if int(params["max_features"]) - params["max_features"]== 0:
        max_features_params = Max_features[int(params["max_features"])]
    else:
        max_features_params = params["max_features"]
        
    rfc = GradientBoostingClassifier(
        loss = losss[int(params["loss"])]
        ,learning_rate = params["learning_rate"]
        ,n_estimators= int(params["n_estimators"])
        ,subsample = params["subsample"]
        ,criterion = criterions[int(params["criterion"])]
        ,min_samples_leaf = params["min_samples_leaf"]
        ,min_samples_split = params["min_samples_split"]
        ,min_weight_fraction_leaf = params["min_weight_fraction_leaf"]
        ,max_depth = int(params["max_depth"])
        ,min_impurity_decrease = params["min_impurity_decrease"]
        # ,min_impurity_split = params["min_impurity_split"]
        ,random_state = params["random_state"]
        ,max_features = max_features_params
        ,max_leaf_nodes = int(params["max_leaf_nodes"])
        ,warm_start = warm_starts[params["warm_start"]]
    )
    rfc =rfc.fit(x_train, y_train)
    rfc_proba = rfc.predict_proba(x_test)[:,1]
    rfc_pred = rfc.predict(x_test)
    metric = PR_AUC(y_test,rfc_proba,rfc_pred)
    
    log(f"解析最优参数:{parsing_bayes_params_for_RandomForest(params)}",LogLevel.SUCCESS)
    log(f"metric:{metric}",LogLevel.SUCCESS)
    return metric



def optimized_param_search_and_report(num_evals):
    start_time = time.time()

    # 进行贝叶斯优化
    params_best, trials = param_hyperopt(num_evals)

    # 打印最佳参数验证结果
    hyperopt_validation(params_best)

    # 打印所有搜索相关的记录
    print("All search records:")
    print(trials.trials[0])


    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # 转换为分钟
    print(f"Optimization completed in {elapsed_time} minutes.")

# 执行优化
optimized_param_search_and_report(300)