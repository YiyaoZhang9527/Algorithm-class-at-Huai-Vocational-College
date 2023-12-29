from hyperopt import hp
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss
# from OptMetrics import MyMetric
from sklearn.ensemble import RandomForestClassifier
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


criterions = ["gini", "entropy"]
max_depths = [None,hp.randint("max_depth_int",0,1000)]
Max_features = ['auto', 'sqrt', 'log2',None,hp.quniform("max_features_int",1,1000,1),hp.uniform("max_features_float",0,1)]
Max_leaf_nodes = [None,hp.quniform("max_leaf_nodes_int",0,1000,1)]
bootstraps = [True,False]
oob_scores = [True,False]
warm_starts = [True,False]
class_weights = [None,'balanced']
param_grid_hp = {
    "n_estimators":hp.quniform("n_estimators",10,1000,1)
    ,"criterion":hp.choice("criterion",criterions)
    ,"max_depth":hp.choice("max_depth",max_depths)
    ,"min_samples_split":hp.quniform("min_samples_split",2,1000,1)
    ,"min_samples_leaf":hp.quniform("min_samples_leaf",1,1000,1)
    ,"min_weight_fraction_leaf":hp.uniform("min_weight_fraction_leaf",0,0.5)
    ,"max_features":hp.choice("max_features",Max_features)
    ,"max_leaf_nodes":hp.choice("max_leaf_nodes",Max_leaf_nodes)
    ,"min_impurity_decrease":hp.uniform("min_impurity_decrease",0,1)
    ,"bootstrap":hp.choice("bootstrap",bootstraps)
    ,"oob_score":hp.choice("oob_score",oob_scores)
    ,"random_state":hp.randint("random_state",100)
    # ,"verbose":hp.quniform("verbose",0,1000,1)
    ,"warm_start":hp.choice("warm_start",warm_starts)
    ,"class_weight":hp.choice("class_weight",class_weights)
    ,"max_samples":hp.uniform("max_samples",0,1)
}

def PR_AUC(test_y,proba,pred):
    precision,recall,_ = precision_recall_curve(test_y,proba)
    f1 ,pr_auc = f1_score(test_y,pred),auc(recall,precision)
    return pr_auc

def hyperopt_objective(hyperopt_params):  
    # log(f"本次参数:{params}",LogLevel.INFO)
    try:
        params = {
            "n_estimators":int(hyperopt_params["n_estimators"])
            ,"criterion":hyperopt_params["criterion"]
            ,"max_depth":hyperopt_params["max_depth"]
            ,"min_samples_split":int(hyperopt_params["min_samples_split"])
            ,"min_samples_leaf":int(hyperopt_params["min_samples_leaf"])
            ,"min_weight_fraction_leaf":hyperopt_params["min_weight_fraction_leaf"]
            ,"max_features":hyperopt_params["max_features"]
            ,"max_leaf_nodes":type(hyperopt_params["max_leaf_nodes"])==float and int(hyperopt_params["max_leaf_nodes"]) or None
            ,"min_impurity_decrease":hyperopt_params["min_impurity_decrease"]
            ,"bootstrap":hyperopt_params["bootstrap"]
            ,"oob_score":hyperopt_params["oob_score"]
            ,"random_state":int(hyperopt_params["random_state"])
            # ,"verbose":int(hyperopt_params["verbose"])
            ,"warm_start":hyperopt_params["warm_start"]
            ,"class_weight":hyperopt_params["class_weight"]
            ,"max_samples":hyperopt_params["max_samples"]
            ,'n_jobs':-1
        }

        rfc = RandomForestClassifier(**params)
        rfc =rfc.fit(x_train, y_train)
        rfc_proba = rfc.predict_proba(x_test)[:,1]
        rfc_pred = rfc.predict(x_test)
        metric = PR_AUC(y_test,rfc_proba,rfc_pred)
    except Exception as e:
        print(e)
        metric = 0
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

params_best, trials = param_hyperopt(300)
print(params_best)