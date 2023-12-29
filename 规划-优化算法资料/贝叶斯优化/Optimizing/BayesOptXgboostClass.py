import platform
import matplotlib.pyplot as plt
system = platform.system()
if system == "Linux":
    plt.rcParams['font.sans-serif'] = ["AR PL UKai CN"] #["Noto Sans CJK JP"]
elif system == "Darwin":
    plt.rcParams['font.sans-serif'] = ["Kaiti SC"]
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import xgboost as xgb
from hyperopt import hp, fmin, tpe, Trials, partial
from MyLogColor import  log,LogLevel
import time
import datetime
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from OptMetrics import MyMetric


class BayesOptXgboost(MyMetric):
    
    def __init__(self,x,y
                 ,Folds=6 #如果采用分层k折交叉验证的时候则写出分多少折
                 ,TEST_SPLIT=0.2 #测试集的比例
                 ,EARLY_STOP_BAYES =100 #当参数优化多少次没有进步的时候就停止搜索
                 ,NUM_EVALS=600 #最大优化参数的搜索次数
                 ,MAX_SHUFFLE=100 #数据洗牌的次数
                 ,x_train=[], x_val=[], y_train=[], y_val=[]
                 ,metrics_class = "pr-auc" #"[pr-auc],[roc-auc],[f1-score],[recall],[precision],[accuracy],[roc-auc-recall],[roc-auc-accuracy]"
                 # all=[pr_auc,roc_auc,accuracy,precision,recall,false_alarm,miss_rate,specificity,f1score]
                 ,metrics_weight = [0.5,0.5]
                 ,min_recall = 0.5 #召回率的最小值
                 ,cost_wight = 0.1 #对召回率不满足的情况下权重值的惩罚值
                 ,StratifiedKFoldShuffle=True
                #  ,is_verbose = False # 是否查看详细信息
                ):
        self.Folds = Folds
        self.TEST_SPLIT =TEST_SPLIT
        self.EARLY_STOP_BAYES = EARLY_STOP_BAYES
        self.NUM_EVALS = NUM_EVALS
        self.MAX_SHUFFLE = MAX_SHUFFLE
        self.metrics_class = metrics_class
        self.metrics_weight = np.array(metrics_weight)
        self.metrics_weight = self.metrics_weight/self.metrics_weight.sum()
        self.min_recall = min_recall
        self.cost_wight = cost_wight
        self.StratifiedKFoldShuffle = StratifiedKFoldShuffle
        
        self.Bayes_start_time = None
        self.NOW_FUC_RUN_ITER = 0
        self.Trials = None
        self.bayes_opt_parser = None
        self.PARAMS_BEST = None
        self.historical_metrics = np.zeros(self.NUM_EVALS)
        self.historical_params = {}
        
        self.all_metrice_names = ['pr_auc',
                            'roc_auc',
                            'accuracy',
                            'precision',
                            'recall',
                            'false_alarm',
                            'miss_rate',
                            'specificity',
                            'f1score']
        
        
        self.boosters = ['gbtree','gblinear','dart']
        # self.sampling_methods = ['uniform','gradient_based']
        self.tree_methods = ["auto","exact","approx","hist"]
        self.refresh_leafs = [0,1]
        self.grow_policys = ["depthwise","lossguide"]
        # self.normalize_types = ["tree","forest"]
        self.param_grid_hp = {
            'boosters':hp.choice('boosters',self.boosters)
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
            ,"tree_method":hp.choice("tree_method",self.tree_methods)
            # ,"scale_pos_weight":hp.uniform("scale_pos_weight",0,1000)
            ,"refresh_leaf":hp.choice("refresh_leaf",self.refresh_leafs)
            # ,"process_type":hp.choice("process_type",process_types)
            ,"grow_policy":hp.choice("grow_policy",self.grow_policys)
            ,"max_leaves":hp.quniform("max_leaves",0,10000,1)
            ,"max_bin":hp.quniform("max_bin",256,1000,1)
            ,"num_parallel_tree":hp.quniform("num_parallel_tree",1,100,1)   
            }
        self.x = x
        self.y = y
        self.m,self.n = x.shape
        self.y = np.array(y,dtype=int)
        
        
        if len(x_train) and len(x_val) and len(y_train) and len(y_val):
            self.x_train, self.x_val, self.y_train, self.y_val = x_train,x_val,y_train,y_val
        else:
            if self.MAX_SHUFFLE > 0:
                self.shuffle_x,self.shuffle_y = self.shuffle_data(x,y)
                self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.shuffle_x, self.shuffle_y, test_size=self.TEST_SPLIT, random_state = 42)
            else:
                self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=self.TEST_SPLIT, random_state = 42)
        self.train_positive = (self.y_train==1).sum()
        self.train_negative = (self.y_train==0).sum()
        self.train_y_counter = self.y_train.size
        self.alpha = self.train_negative/self.train_y_counter

        log(f"""训练数据中,正例有【{self.train_positive}】个占比【{self.train_positive/self.train_y_counter}】
            ，负例有【{self.train_negative}】个占比【{self.train_negative/self.train_y_counter}】
            ，alpha值为【{self.alpha}】，""",LogLevel.INFO)
        
        self.test_positive = (self.y_val==1).sum()
        self.test_negative = (self.y_val==0).sum()
        self.test_y_counter = self.y_val.size

        log(f"""测试数据中,正例有【{self.test_positive}】个占比【{self.test_positive/self.test_y_counter }】
            ，负例有【{self.test_negative}】个占比【{self.test_negative/self.test_y_counter }】
            ，alpha值为【{self.test_negative/self.test_y_counter}】，""",LogLevel.INFO)


    def shuffle_data(self,x,y):
        """
        数据洗牌
        """
        xy = np.c_[x,y]
        for i in tqdm(range(self.MAX_SHUFFLE),desc="数据洗牌"):
            np.random.shuffle(xy)
        x,y = xy[:,0:self.n],xy[:,self.n:self.n+1]
        y = np.ravel(y)
        return x,y
    
    def hyperopt_objective(self,hyperopt_params):
        
        func_start = time.time()
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
            ,"scale_pos_weight":(self.y_train==0).sum()/(self.y_train==1).sum()
            ,"refresh_leaf":hyperopt_params["refresh_leaf"]
            # ,"process_type":hyperopt_params["process_type"]
            ,"grow_policy":hyperopt_params["grow_policy"]
            ,"max_leaves":int(hyperopt_params["max_leaves"])
            ,"max_bin":int(hyperopt_params["max_bin"])
            ,"num_parallel_tree":int(hyperopt_params["num_parallel_tree"])   
        }
        log(f"本次参数:{params}",LogLevel.INFO) 
        try:
            if isinstance(self.Folds,(int,float)) and self.Folds > 0:
                strkf = StratifiedKFold(n_splits=self.Folds, shuffle=self.StratifiedKFoldShuffle)
                '''
                n_splits=6（默认5）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果

                shuffle=True（默认False）：每次划分前数据重新洗牌，每次的运行结果不同；shuffle=False：每次运行结果相同，相当于random_state=整数
                random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
                '''
                metrics_ = np.zeros(self.Folds)
                log(f"k={self.Folds}折分层交叉验证",LogLevel.PASS)
                for i,index_ in enumerate(strkf.split(self.x,self.y)):
                    train_index,test_index = index_
                    # print(train_index,test_index)
                    X_train_KFold, X_test_KFold = self.x[train_index],self.x[test_index]
                    y_train_KFold, y_test_KFold = self.y[train_index],self.y[test_index]
                    dtrain = xgb.DMatrix(X_train_KFold,label=y_train_KFold)
                    xgboostc = xgb.train(params=params
                                ,dtrain=dtrain
                                ,num_boost_round=100
                                ,evals=[(dtrain,"train")]
                                ,verbose_eval=False # 不显示训练信息就改False
                                # ,obj=logistic_obj
                                )
                    
                    dtest = xgb.DMatrix(self.x_val,label=self.y_val)
                    xgboost_proba = xgboostc.predict(X_test_KFold)
                    xgboost_pred = (xgboost_proba>0.5).astype(int)
                    
                    if self.metrics_class == "pr-auc":
                        metrics_[i] = self.PR_AUC(y_test_KFold, xgboost_proba,xgboost_pred)
                    elif self.metrics_class == "roc-auc":
                        metrics_[i] = self.ROC_AUC(y_test_KFold, xgboost_proba)
                    elif self.metrics_class == "f1-score":
                        metrics_[i] = f1_score(y_test_KFold,xgboost_pred) 
                    elif self.metrics_class == "recall":
                        metrics_[i] = self.TPRN_Score(y_test_KFold,xgboost_pred)["召回率|recall｜真阳率｜命中率"] 
                    elif self.metrics_class == "precision":
                        metrics_[i] = self.TPRN_Score(y_test_KFold,xgboost_pred)["精确率|precision"] 
                    elif self.metrics_class == "accuracy":
                        metrics_[i] = self.TPRN_Score(y_test_KFold,xgboost_pred)["ACC正确率|accuracy"]
                    elif self.metrics_class == "roc-auc-recall":
                        roc_auc = self.ROC_AUC(y_test_KFold, xgboost_proba)
                        recall = self.TPRN_Score(y_test_KFold,xgboost_pred)["召回率|recall｜真阳率｜命中率"]
                        metrics_[i] = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[-1]
                    elif self.metrics_class == "roc-auc-recall-accuracy":
                        roc_auc = self.ROC_AUC(y_test_KFold, xgboost_proba)
                        recall = self.TPRN_Score(y_test_KFold,xgboost_pred)["召回率|recall｜真阳率｜命中率"]
                        accuracy = self.TPRN_Score(y_test_KFold,xgboost_pred)["ACC正确率|accuracy"]
                        metrics_[i] = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[1]+accuracy*self.metrics_weight[-1] 
                    if self.metrics_class == "all":
                        TPRN = self.TPRN_Score(y_test_KFold,xgboost_pred)
                        pr_auc = self.PR_AUC(y_test_KFold, xgboost_proba,xgboost_pred)
                        roc_auc = self.ROC_AUC(y_test_KFold, xgboost_proba)
                        accuracy = TPRN["ACC正确率|accuracy"]
                        precision = TPRN['精确率|precision']
                        recall = TPRN["召回率|recall｜真阳率｜命中率"]
                        false_alarm = TPRN['误报率|false alarm｜假阳率｜虚警率｜误检率'] 
                        miss_rate = TPRN['漏报率|miss rate|也称为漏警率|漏检率']
                        specificity = TPRN['特异度|specificity']
                        f1score = f1_score(y_test_KFold,xgboost_pred) 
                        
                        metrics_values = np.array([pr_auc,roc_auc,accuracy,precision,recall,-false_alarm,-miss_rate,specificity,f1score])
                        
                        metrics_values = np.nan_to_num(metrics_values,0)
                        if recall > self.min_recall:
                            metrics_weight = np.nan_to_num(self.metrics_weight,0)
                        else:
                            metrics_weight = np.nan_to_num(self.metrics_weight,0)*self.cost_wight
                        metrics_[i] = metrics_values.dot(metrics_weight)                    
                # roc_auc  = roc_aucs.mean()
                metrics_value = metrics_.mean()
                    
            else: 
                dtrain = xgb.DMatrix(self.x_train,label=self.y_train)
                xgboostc = xgb.train(params=params
                            ,dtrain=dtrain
                            ,num_boost_round=100
                            ,evals=[(dtrain,"train")]
                            ,verbose_eval=False # 不显示训练信息就改False
                            # ,obj=logistic_obj
                            )
                dtest = xgb.DMatrix(self.x_val,label=self.y_val)
                xgboost_proba = xgboostc.predict(dtest)
                xgboost_pred = (xgboost_proba>0.5).astype(int)
                if self.metrics_class == "pr-auc":
                    metrics_value = self.PR_AUC(self.y_val, xgboost_proba,xgboost_pred)
                elif self.metrics_class == "roc-auc":
                    metrics_value = self.ROC_AUC(self.y_val, xgboost_proba)
                elif self.metrics_class == "f1-score":
                    metrics_value = f1_score(self.y_val,xgboost_pred)
                elif self.metrics_class == "recall":
                    metrics_value = self.TPRN_Score(self.y_val,xgboost_pred)["召回率|recall｜真阳率｜命中率"] 
                elif self.metrics_class == "precision":
                    metrics_value = self.TPRN_Score(self.y_val,xgboost_pred)["精确率|precision"] 
                elif self.metrics_class == "accuracy":
                    metrics_value = self.TPRN_Score(self.y_val,xgboost_pred)["ACC正确率|accuracy"]
                elif self.metrics_class == "roc-auc-recall":
                    roc_auc = self.ROC_AUC(self.y_val, xgboost_proba)
                    recall = self.TPRN_Score(self.y_val,xgboost_pred)["召回率|recall｜真阳率｜命中率"]
                    metrics_value = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[-1]
                elif self.metrics_class == "roc-auc-recall-accuracy":
                    roc_auc = self.ROC_AUC(self.y_val, xgboost_proba)
                    recall = self.TPRN_Score(self.y_val,xgboost_pred)["召回率|recall｜真阳率｜命中率"]
                    accuracy = self.TPRN_Score(self.y_val,xgboost_pred)["ACC正确率|accuracy"]
                    metrics_value = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[1]+accuracy*self.metrics_weight[-1]                      
                elif self.metrics_class == "all":
                    TPRN = self.TPRN_Score(self.y_val, xgboost_pred)
                    pr_auc = self.PR_AUC(self.y_val, xgboost_proba,xgboost_pred)
                    roc_auc = self.ROC_AUC(self.y_val, xgboost_proba)
                    accuracy = TPRN["ACC正确率|accuracy"]
                    precision = TPRN['精确率|precision']
                    recall = TPRN["召回率|recall｜真阳率｜命中率"]
                    false_alarm = TPRN['误报率|false alarm｜假阳率｜虚警率｜误检率'] 
                    miss_rate = TPRN['漏报率|miss rate|也称为漏警率|漏检率']
                    specificity = TPRN['特异度|specificity']
                    f1score = f1_score(self.y_val,xgboost_pred) 
                    metrics_values = np.array([pr_auc,roc_auc,accuracy,precision,recall,-false_alarm,-miss_rate,specificity,f1score])
                    metrics_values = np.nan_to_num(metrics_values,0)
                    if recall > self.min_recall:
                        metrics_weight = np.nan_to_num(self.metrics_weight,0)
                    else:
                        metrics_weight = np.nan_to_num(self.metrics_weight,0)*self.cost_wight
                    metrics_value = metrics_values.dot(metrics_weight)
        except Exception as e:
            log(str(e),LogLevel.ERROR)
            metrics_value = self.historical_metrics.mean()   
        # metrics_value = MyMetric.PR_AUC(self.y_val,xgboost_proba,xgboost_pred)
        
        func_end = time.time()
        # global NOW_FUC_RUN_ITER
        self.NOW_FUC_RUN_ITER += 1
        self.historical_params.update({self.NOW_FUC_RUN_ITER-1:params})
        self.historical_metrics[self.NOW_FUC_RUN_ITER-1] = metrics_value
        
        log(f"""本次迭代{self.metrics_class}分数为:[{metrics_value}],
        用时:[{func_end-func_start}]秒,
        当前优化第:[{self.NOW_FUC_RUN_ITER}]次,
        已运行:[{self.NOW_FUC_RUN_ITER}]次，
        用时总计:[{datetime.timedelta(seconds=(func_end-self.Bayes_start_time))}]秒,
        """,LogLevel.PASS)
        return -metrics_value
    
    def param_hyperopt(self,max_evals=100):
        """
        """
        #保存迭代过程
        trials = Trials()

        #设置提前停止
        ## 如果损失没有增加，将在 X 次迭代后停止的停止函数
        early_stop_fn = no_progress_loss(self.EARLY_STOP_BAYES)

        #定义代理模型
        #algo = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=50)
        # global hyperopt_params
        # hyperopt_params = self.param_grid_hp
        params_best = fmin(self.hyperopt_objective #目标函数
                        , space = self.param_grid_hp #参数空间
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
    
    def run(self):
        t = 0
        for params_name,obj in self.param_grid_hp.items():
            t += 1
            log(f"已准备优化的第{t}个参数,名称:{params_name},类型:{obj}",LogLevel.PASS)

        self.Bayes_start_time = time.time()
        self.NOW_FUC_RUN_ITER = 0
        self.PARAMS_BEST, self.Trials = self.param_hyperopt(self.NUM_EVALS)
        idx = np.argmin([-self.historical_metrics])
        # self.historical_metrics[idx]
        self.bayes_opt_parser = self.historical_params[idx]
        log(f"解析参数得到结果:{self.bayes_opt_parser}",LogLevel.PASS)
        
    def test_params(self):
        params = self.bayes_opt_parser
        # gbdtc = RandomForestClassifier(**params)
        # gbdtc =gbdtc.fit(self.x_train, self.y_train)
        # xgboost_proba = gbdtc.predict_proba(self.x_val)[:,1]
        # xgboost_pred = gbdtc.predict(self.x_val)
        dtrain = xgb.DMatrix(self.x_train,label=self.y_train)
        xgboostc = xgb.train(params=params
                    ,dtrain=dtrain
                    ,num_boost_round=100
                    ,evals=[(dtrain,"train")]
                    ,verbose_eval=False # 不显示训练信息就改False
                    # ,obj=logistic_obj
                    )
        dtest = xgb.DMatrix(self.x_val,label=self.y_val)
        xgboost_proba = xgboostc.predict(dtest)
        xgboost_pred = (xgboost_proba>0.5).astype(int)
        # metric = MyMetric.PR_AUC(self.y_val,xgboost_proba,xgboost_pred)
        log(f'模型的评估报告1：\n,{classification_report(self.y_val, xgboost_pred)}\n',LogLevel.SUCCESS)
        log(f'模型的评估报告2：\n,{self.TPRN_Score(self.y_val, xgboost_pred)}\n',LogLevel.SUCCESS)
        
        # self.plot_roc(self.y_val, xgboost_proba[:,1])
        if self.metrics_class == "pr-auc":
            metrics = self.PR_AUC(self.y_val, xgboost_proba,xgboost_pred)
        elif self.metrics_class == "roc-auc":
            metrics = self.ROC_AUC(self.y_val, xgboost_proba)
        elif self.metrics_class == "f1-score":
            metrics = f1_score(self.y_val,xgboost_pred)
        elif self.metrics_class == "recall":
            metrics = self.TPRN_Score(self.y_val,xgboost_pred)["召回率|recall｜真阳率｜命中率"] 
        elif self.metrics_class == "precision":
            metrics = self.TPRN_Score(self.y_val,xgboost_pred)["精确率|precision"] 
        elif self.metrics_class == "accuracy":
            metrics = self.TPRN_Score(self.y_val,xgboost_pred)["ACC正确率|accuracy"]
        elif self.metrics_class == "roc-auc-recall":
            roc_auc = self.ROC_AUC(self.y_val, xgboost_proba)
            recall = self.TPRN_Score(self.y_val,xgboost_pred)["召回率|recall｜真阳率｜命中率"]
            metrics = roc_auc*self.metrics_weight[0]*recall*self.metrics_weight[-1]
        elif self.metrics_class == "roc-auc-recall-accuracy":
            roc_auc = self.ROC_AUC(self.y_val, xgboost_proba)
            recall = self.TPRN_Score(self.y_val,xgboost_pred)["召回率|recall｜真阳率｜命中率"]
            accuracy = self.TPRN_Score(self.y_val,xgboost_pred)["ACC正确率|accuracy"]
            metrics = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[1]+accuracy*self.metrics_weight[-1]                      
        elif self.metrics_class == "all":
            TPRN = self.TPRN_Score(self.y_val, xgboost_pred)
            pr_auc = self.PR_AUC(self.y_val, xgboost_proba,xgboost_pred)
            roc_auc = self.ROC_AUC(self.y_val, xgboost_proba)
            accuracy = TPRN["ACC正确率|accuracy"]
            precision = TPRN['精确率|precision']
            recall = TPRN["召回率|recall｜真阳率｜命中率"]
            false_alarm = TPRN['误报率|false alarm｜假阳率｜虚警率｜误检率'] 
            miss_rate = TPRN['漏报率|miss rate|也称为漏警率|漏检率']
            specificity = TPRN['特异度|specificity']
            f1score = f1_score(self.y_val,xgboost_pred) 
            metrics_values = np.array([pr_auc,roc_auc,accuracy,precision,recall,false_alarm,miss_rate,specificity,f1score])
            #metrics_weight = np.array(self.metrics_weight)
            metrics_values = np.nan_to_num(metrics_values,0)
            metrics_weight = np.nan_to_num(self.metrics_weight,0)
            for metrice_name,metrics_value,weight in zip(self.all_metrice_names,metrics_values,metrics_weight):
                log(f'测试{metrice_name}值为:{metrics_value},权重为{weight}',LogLevel.SUCCESS)
            metrics = metrics_values.dot(metrics_weight) 
        log(f'测试优化参数的:【{self.metrics_class}】得分为:【{metrics}】,优化过程中的最高分为:【{self.historical_metrics.max()}】',LogLevel.SUCCESS)
        return metrics
            
        
if __name__ == '__main__':
    from sklearn import datasets
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
    boxgboost = BayesOptXgboost(x,y
                       ,MAX_SHUFFLE=100
                        ,Folds=0
                        ,metrics_class="all"
                        #all=[1.pr_auc,2.roc_auc,3.accuracy,4.precision,5.recall,6.false_alarm,7.miss_rate,8.specificity,9.f1score]
                        ,metrics_weight=[0,0.5,0.5,0,0,0,0,0,0]
                        ,EARLY_STOP_BAYES=200
                        ,NUM_EVALS=1000
                        ,min_recall=0
                        ,cost_wight=1
                       )
    boxgboost.run()
    boxgboost.test_params()