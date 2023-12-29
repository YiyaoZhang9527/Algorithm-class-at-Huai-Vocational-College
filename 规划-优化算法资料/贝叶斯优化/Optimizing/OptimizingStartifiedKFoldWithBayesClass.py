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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, partial
from MyLogColor import  log,LogLevel
from sklearn import metrics
import time
import datetime
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve,auc,f1_score
# from functools import partial
# from uniplot import plot as default_plot


class BayesOptLightGBM():

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
        self.NUM_EVALS = NUM_EVALS
        self.EARLY_STOP_BAYES = EARLY_STOP_BAYES
        self.TEST_SPLIT =TEST_SPLIT
        self.MAX_SHUFFLE = MAX_SHUFFLE
        # print(self.MAX_SHUFFLE)
        self.metrics_class = metrics_class
        self.StratifiedKFoldShuffle = StratifiedKFoldShuffle
        # self.is_verbose = is_verbose
        
        self.all_metrice_names = ['pr_auc',
                                    'roc_auc',
                                    'accuracy',
                                    'precision',
                                    'recall',
                                    'false_alarm',
                                    'miss_rate',
                                    'specificity',
                                    'f1score']
        
        self.metrics_weight = np.array(metrics_weight)
        self.metrics_weight = self.metrics_weight/self.metrics_weight.sum()
        self.min_recall = min_recall
        self.cost_wight = cost_wight
        
        self.tree_learners = ["feature","serial","data","voting"]
        self.boostings = ["gbdt","rf"]
        self.is_unbalances = [True,False]
        self.param_grid_hp = {
            'min_data_in_leaf': hp.quniform('min_data_in_leaf',5,1000,1),
            'num_leaves': hp.quniform("num_leaves",20,1000,1), #一棵树的最大叶子数
            "boosting": hp.choice("boosting",self.boostings),#gbdt 传统的梯度提升决策树,rf随机森林,dart Dropout 遇到多个可加回归树,
            'n_estimators':hp.quniform('n_estimators',80,2000,1),#基学习器
            "tree_learner": hp.choice("tree_learner",self.tree_learners),#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
            'max_bin': hp.quniform("max_bin",20,1000,1),#50, #直方图分箱特征值的将被存储的最大箱数，少量的 bin 可能会降低训练准确性，但可能会增加处理过度拟合的能力
            "min_data_in_bin":hp.quniform("min_data_in_bin",3,100,1),#一个 bin 内的数据最少数量,使用它可以避免一数据一箱（潜在的过度拟合）
            'max_depth':hp.quniform('max_depth',5,100,1),#15, #限制树模型的最大深度
            #"min_data_in_leaf":hp.quniform("min_data_in_leaf",3,1000,1),#100,#一片叶子中的数据数量最少。可以用来处理过拟合
            "learning_rate": hp.uniform("learning_rate",0,1),#0.01,#学习率
            "bagging_fraction":hp.uniform("bagging_fraction",0.01,1),# 0.8,  # 每次迭代时用的数据比例，但这将随机选择部分数据而不重新采样，可用于加速训练可以用来处理过拟​​合
            "feature_fraction":hp.uniform("feature_fraction",0.01,1),#0.8, # 每次迭代中随机选择特征的比例，lightGBM 将在每次迭代（树）上随机选择特征子集1.0。例如，如果将其设置为0.8，LightGBM 将在训练每棵树之前选择 80% 的特征
            "lambda_l1":hp.randint("lambda_l1",1000),#20, #L1正则化 0-正无穷
            "lambda_l2":hp.randint("lambda_l2",1000),#20 ,
            'seed': hp.randint('seed',100),#42,
            'bagging_freq':hp.quniform("bagging_freq",0,100,1),#3,#0表示禁用装袋；k意味着在每次迭代时执行装袋k。每次k迭代，LightGBM 都会随机选择用于下一次迭代的数据
            'is_unbalance':hp.choice('is_unbalance',self.is_unbalances),#True, #是否为不平衡数据
            "early_stopping_rounds":hp.quniform("early_stopping_rounds",1,1000,1),#30,#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练

        }  #设置出参数
        self.Bayes_start_time = None
        self.NOW_FUC_RUN_ITER = 0
        self.Trials = None
        self.bayes_opt_parser = None
        self.PARAMS_BEST = None
        self.historical_metrics = np.zeros(self.NUM_EVALS)



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
        
        self.param_grid_hp = {
            'min_data_in_leaf': hp.quniform('min_data_in_leaf',5,1000,1),
            'num_leaves': hp.quniform("num_leaves",20,1000,1), #一棵树的最大叶子数
            "boosting": hp.choice("boosting",self.boostings),#gbdt 传统的梯度提升决策树,rf随机森林,dart Dropout 遇到多个可加回归树,
            'n_estimators':hp.quniform('n_estimators',80,2000,1),#基学习器
            "tree_learner": hp.choice("tree_learner",self.tree_learners),#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
            'max_bin': hp.quniform("max_bin",20,1000,1),#50, #直方图分箱特征值的将被存储的最大箱数，少量的 bin 可能会降低训练准确性，但可能会增加处理过度拟合的能力
            "min_data_in_bin":hp.quniform("min_data_in_bin",3,100,1),#一个 bin 内的数据最少数量,使用它可以避免一数据一箱（潜在的过度拟合）
            'max_depth':hp.quniform('max_depth',5,100,1),#15, #限制树模型的最大深度
            #"min_data_in_leaf":hp.quniform("min_data_in_leaf",3,1000,1),#100,#一片叶子中的数据数量最少。可以用来处理过拟合
            "learning_rate": hp.uniform("learning_rate",0,1),#0.01,#学习率
            "bagging_fraction":hp.uniform("bagging_fraction",0.01,1),# 0.8,  # 每次迭代时用的数据比例，但这将随机选择部分数据而不重新采样，可用于加速训练可以用来处理过拟​​合
            "feature_fraction":hp.uniform("feature_fraction",0.01,1),#0.8, # 每次迭代中随机选择特征的比例，lightGBM 将在每次迭代（树）上随机选择特征子集1.0。例如，如果将其设置为0.8，LightGBM 将在训练每棵树之前选择 80% 的特征
            "lambda_l1":hp.randint("lambda_l1",1000),#20, #L1正则化 0-正无穷
            "lambda_l2":hp.randint("lambda_l2",1000),#20 ,
            'seed': hp.randint('seed',100),#42,
            'bagging_freq':hp.quniform("bagging_freq",0,100,1),#3,#0表示禁用装袋；k意味着在每次迭代时执行装袋k。每次k迭代，LightGBM 都会随机选择用于下一次迭代的数据
            'is_unbalance':hp.choice('is_unbalance',self.is_unbalances),#True, #是否为不平衡数据
            "early_stopping_rounds":hp.quniform("early_stopping_rounds",1,1000,1),#30,#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练

        }  #设置出参数

        

    @staticmethod
    def ROC_AUC(y_lab, proba):
        fpr,tpr,threshold = metrics.roc_curve(y_lab, proba)
        roc_auc_ = metrics.auc(fpr,tpr)
        return roc_auc_
        
    @staticmethod
    def PR_AUC(test_y,gbm_proba,gbm_pred):
        gbm_precision,gbm_recall,_ = precision_recall_curve(test_y,gbm_proba)
        gbm_f1 ,gbm_auc = f1_score(test_y,gbm_pred),auc(gbm_recall,gbm_precision)
        return gbm_auc
    
    @staticmethod
    def TPRN_Score(test_y,pred_y):
        TP = ((test_y==1) * (pred_y==1)).sum()
        FN = ((test_y == 1) * (pred_y == 0)).sum()
        FP = ((test_y==0)*(pred_y==1)).sum()
        TN = ((test_y==0)*(pred_y==0)).sum()
        
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        precision = TP/(TP+FP)
        N = FP+TN
        P = TP+FN
        ACC = (TP+TN)/(TP+FN+FP+TN)
        FNR = FN/(TP+FN)
        
        # 两种F1 Score 计算结果是一样的
        F1Score = (2*TP)/(2*TP+FN+FP)
    # F1Score = (2*precision*TPR)/(precision+TPR)
    
        return {"混淆矩阵":np.matrix([[TP,FN],[FP,TN]])
            ,"ACC正确率|accuracy":ACC
            ,"精确率|precision":precision
            ,"召回率|recall｜真阳率｜命中率":TPR
            ,"误报率|false alarm｜假阳率｜虚警率｜误检率":FPR
            ,"漏报率|miss rate|也称为漏警率|漏检率":FNR
            ,"特异度|specificity":1-FPR
            ,"F1-score:":F1Score
            ,"真实正样本数":P,"真实负样本数":N}
            
    
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

    def plot_roc(self,y_test, y_score):
        fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
        roc_auc = metrics.auc(fpr,tpr)
        plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
        plt.plot(fpr, tpr, color='black', lw = 1)
        plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
        plt.text(0.5,0.3,'ROC curve (area = %0.10f)' % roc_auc)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.show()

    def hyperopt_objective(self,hyperopt_params):

        func_start = time.time()
        
        params = {
                'verbose':-1,
                'min_data_in_leaf': int(hyperopt_params['min_data_in_leaf']),#一片叶子中的数据数量最少。可以用来处理过拟​​合注意：这是基于 Hessian 的近似值，因此有时您可能会观察到分裂产生的叶节点的观测值少于这么多
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': int(hyperopt_params['num_leaves']), #一棵树的最大叶子数
                "boosting": hyperopt_params["boosting"],#"gbdt",#gbdt 传统的梯度提升决策树,rf随机森林,dartDropout 遇到多个可加回归树,
                'n_estimators':int(hyperopt_params['n_estimators']),#2000,#基学习器
                "tree_learner":hyperopt_params["tree_learner"],#"feature",#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
                'max_bin': int(hyperopt_params['max_bin']), #直方图分箱特征值的将被存储的最大箱数，少量的 bin 可能会降低训练准确性，但可能会增加处理过度拟合的能力
                "min_data_in_bin":int(hyperopt_params["min_data_in_bin"]), #一个 bin 内的数据最少数量.使用它可以避免一数据一箱（潜在的过度拟合)
                'max_depth':int(hyperopt_params['max_depth']), #限制树模型的最大深度
                #"min_data_in_leaf":int(hyperopt_params["min_data_in_leaf"]),#一片叶子中的数据数量最少。可以用来处理过拟合
                "learning_rate": hyperopt_params["learning_rate"],#学习率
                #"colsample_bytree": 0.8,  
                "bagging_fraction": hyperopt_params["bagging_fraction"],  # 每次迭代时用的数据比例，但这将随机选择部分数据而不重新采样，可用于加速训练可以用来处理过拟​​合
                "feature_fraction":hyperopt_params["feature_fraction"], # 每次迭代中随机选择特征的比例，lightGBM 将在每次迭代（树）上随机选择特征子集1.0。例如，如果将其设置为0.8，LightGBM 将在训练每棵树之前选择 80% 的特征
                "lambda_l1":hyperopt_params["lambda_l1"], #L1正则化 0-正无穷
                "lambda_l2":hyperopt_params["lambda_l2"],
                'n_jobs': -1,
                #'silent': 1,  # 信息输出设置成1则没有信息输出
                'seed': int(hyperopt_params['seed']),
                'bagging_freq':int(hyperopt_params['bagging_freq']),#装袋频率,0表示禁用装袋；k意味着在每次迭代时执行装袋k。每次k迭代，LightGBM 都会随机选择用于下一次迭代的数据bagging_fraction * 100 %k
                'is_unbalance':hyperopt_params['is_unbalance'], #是否为不平衡数据
                "early_stopping_rounds":int(hyperopt_params["early_stopping_rounds"]),#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练
                "device_type":"cpu"#"cuda"
                #'scale_pos_weight': wt
            }  #设置出参数
        log(f"本次参数为:[{params}]",LogLevel.INFO)
        try:
            ####TODO:分层k折交叉验证，以处理数据极度不平衡问题
            # self.Folds
            if isinstance(self.Folds,(int,float)) and self.Folds > 0:
                # print(self.Folds)
                # self.StratifiedKFoldShuffle
                strkf = StratifiedKFold(n_splits=self.Folds, shuffle=self.StratifiedKFoldShuffle)
                '''
                n_splits=6（默认5）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果

                shuffle=True（默认False）：每次划分前数据重新洗牌，每次的运行结果不同；shuffle=False：每次运行结果相同，相当于random_state=整数
                random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
                '''
                # roc_aucs = np.zeros(self.Folds)
                metrics_ = np.zeros(self.Folds)
                log(f"k={self.Folds}折分层交叉验证",LogLevel.PASS)
                for i,index_ in enumerate(strkf.split(self.x,self.y)):
                    train_index,test_index = index_
                    # print(train_index,test_index)
                    X_train_KFold, X_test_KFold = self.x[train_index],self.x[test_index]
                    y_train_KFold, y_test_KFold = self.y[train_index],self.y[test_index]
                    gbm = lgb.LGBMClassifier(**params)
                    gbm.fit(X_train_KFold, y_train_KFold, 
                            # verbose_eval=True ,
                            eval_metric='auc',
                            verbose = self.is_verbose, #
                            # verbose=False,
                            eval_set=[(X_train_KFold, y_train_KFold), (X_test_KFold, y_test_KFold)]
                            # ,early_stopping_rounds=30
                        )
                    gbm_pred = gbm.predict(X_test_KFold)
                    gbm_proba = gbm.predict_proba(X_test_KFold)[:,1]
                    if self.metrics_class == "pr-auc":
                        metrics_[i] = self.PR_AUC(y_test_KFold, gbm_proba,gbm_pred)
                    elif self.metrics_class == "roc-auc":
                        metrics_[i] = self.ROC_AUC(y_test_KFold, gbm_proba)
                    elif self.metrics_class == "f1-score":
                        metrics_[i] = f1_score(y_test_KFold,gbm_pred) 
                    elif self.metrics_class == "recall":
                        metrics_[i] = self.TPRN_Score(y_test_KFold,gbm_pred)["召回率|recall｜真阳率｜命中率"] 
                    elif self.metrics_class == "precision":
                        metrics_[i] = self.TPRN_Score(y_test_KFold,gbm_pred)["精确率|precision"] 
                    elif self.metrics_class == "accuracy":
                        metrics_[i] = self.TPRN_Score(y_test_KFold,gbm_pred)["ACC正确率|accuracy"]
                    elif self.metrics_class == "roc-auc-recall":
                        roc_auc = self.ROC_AUC(y_test_KFold, gbm_proba)
                        recall = self.TPRN_Score(y_test_KFold,gbm_pred)["召回率|recall｜真阳率｜命中率"]
                        metrics_[i] = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[-1]
                    elif self.metrics_class == "roc-auc-recall-accuracy":
                        roc_auc = self.ROC_AUC(y_test_KFold, gbm_proba)
                        recall = self.TPRN_Score(y_test_KFold,gbm_pred)["召回率|recall｜真阳率｜命中率"]
                        accuracy = self.TPRN_Score(y_test_KFold,gbm_pred)["ACC正确率|accuracy"]
                        metrics_[i] = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[1]+accuracy*self.metrics_weight[-1] 
                    if self.metrics_class == "all":
                        TPRN = self.TPRN_Score(y_test_KFold,gbm_pred)
                        pr_auc = self.PR_AUC(y_test_KFold, gbm_proba,gbm_pred)
                        roc_auc = self.ROC_AUC(y_test_KFold, gbm_proba)
                        accuracy = TPRN["ACC正确率|accuracy"]
                        precision = TPRN['精确率|precision']
                        recall = TPRN["召回率|recall｜真阳率｜命中率"]
                        false_alarm = TPRN['误报率|false alarm｜假阳率｜虚警率｜误检率'] 
                        miss_rate = TPRN['漏报率|miss rate|也称为漏警率|漏检率']
                        specificity = TPRN['特异度|specificity']
                        f1score = f1_score(y_test_KFold,gbm_pred) 
                        
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
                ###TODO:单次迭代
                # li.log_evaluation(1)
                # lgb.log_evaluation(0)
                gbm = lgb.LGBMClassifier(**params)
                gbm.fit(self.x_train, self.y_train, 
                    # verbose_eval=True ,
                        eval_metric='auc',
                        # verbose = self.is_verbose,
                eval_set=[(self.x_train, self.y_train), (self.x_val, self.y_val)]
                    # ,early_stopping_rounds=30
                )
                gbm_pred = gbm.predict(self.x_val)
                gbm_proba = gbm.predict_proba(self.x_val)[:,1]
                if self.metrics_class == "pr-auc":
                    metrics_value = self.PR_AUC(self.y_val, gbm_proba,gbm_pred)
                elif self.metrics_class == "roc-auc":
                    metrics_value = self.ROC_AUC(self.y_val, gbm_proba)
                elif self.metrics_class == "f1-score":
                    metrics_value = f1_score(self.y_val,gbm_pred)
                elif self.metrics_class == "recall":
                    metrics_value = self.TPRN_Score(self.y_val,gbm_pred)["召回率|recall｜真阳率｜命中率"] 
                elif self.metrics_class == "precision":
                    metrics_value = self.TPRN_Score(self.y_val,gbm_pred)["精确率|precision"] 
                elif self.metrics_class == "accuracy":
                    metrics_value = self.TPRN_Score(self.y_val,gbm_pred)["ACC正确率|accuracy"]
                elif self.metrics_class == "roc-auc-recall":
                    roc_auc = self.ROC_AUC(self.y_val, gbm_proba)
                    recall = self.TPRN_Score(self.y_val,gbm_pred)["召回率|recall｜真阳率｜命中率"]
                    metrics_value = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[-1]
                elif self.metrics_class == "roc-auc-recall-accuracy":
                    roc_auc = self.ROC_AUC(self.y_val, gbm_proba)
                    recall = self.TPRN_Score(self.y_val,gbm_pred)["召回率|recall｜真阳率｜命中率"]
                    accuracy = self.TPRN_Score(self.y_val,gbm_pred)["ACC正确率|accuracy"]
                    metrics_value = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[1]+accuracy*self.metrics_weight[-1]                      
                if self.metrics_class == "all":
                    TPRN = self.TPRN_Score(self.y_val, gbm_pred)
                    pr_auc = self.PR_AUC(self.y_val, gbm_proba,gbm_pred)
                    roc_auc = self.ROC_AUC(self.y_val, gbm_proba)
                    accuracy = TPRN["ACC正确率|accuracy"]
                    precision = TPRN['精确率|precision']
                    recall = TPRN["召回率|recall｜真阳率｜命中率"]
                    false_alarm = TPRN['误报率|false alarm｜假阳率｜虚警率｜误检率'] 
                    miss_rate = TPRN['漏报率|miss rate|也称为漏警率|漏检率']
                    specificity = TPRN['特异度|specificity']
                    f1score = f1_score(self.y_val,gbm_pred) 
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
                            
        func_end = time.time()
        # global NOW_FUC_RUN_ITER
        self.NOW_FUC_RUN_ITER += 1
        self.historical_metrics[self.NOW_FUC_RUN_ITER-1]=metrics_value

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
    
    def parsing_bayes_params_for_LightGBM(self,hyperopt_params):
        """
        解析估计出的beyes最优参数
        """
        params = {
            'verbose':-1,
            'min_data_in_leaf': int(hyperopt_params['min_data_in_leaf']),#一片叶子中的数据数量最少。可以用来处理过拟​​合注意：这是基于 Hessian 的近似值，因此有时您可能会观察到分裂产生的叶节点的观测值少于这么多
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': int(hyperopt_params['num_leaves']), #一棵树的最大叶子数
            "boosting": self.boostings[hyperopt_params["boosting"]],#"gbdt",#gbdt 传统的梯度提升决策树,rf随机森林,dartDropout 遇到多个可加回归树,
            'n_estimators':int(hyperopt_params['n_estimators']),#2000,#基学习器
            "tree_learner": self.tree_learners[hyperopt_params["tree_learner"]],#"feature",#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
            'max_bin': int(hyperopt_params['max_bin']), #直方图分箱特征值的将被存储的最大箱数，少量的 bin 可能会降低训练准确性，但可能会增加处理过度拟合的能力
            "min_data_in_bin":int(hyperopt_params["min_data_in_bin"]), #一个 bin 内的数据最少数量.使用它可以避免一数据一箱（潜在的过度拟合)
            'max_depth':int(hyperopt_params['max_depth']), #限制树模型的最大深度
            #"min_data_in_leaf":int(hyperopt_params["min_data_in_leaf"]),#一片叶子中的数据数量最少。可以用来处理过拟合
            "learning_rate": hyperopt_params["learning_rate"],#学习率
            #"colsample_bytree": 0.8,  
            "bagging_fraction": hyperopt_params["bagging_fraction"],  # 每次迭代时用的数据比例，但这将随机选择部分数据而不重新采样，可用于加速训练可以用来处理过拟​​合
            "feature_fraction":hyperopt_params["feature_fraction"], # 每次迭代中随机选择特征的比例，lightGBM 将在每次迭代（树）上随机选择特征子集1.0。例如，如果将其设置为0.8，LightGBM 将在训练每棵树之前选择 80% 的特征
            "lambda_l1":hyperopt_params["lambda_l1"], #L1正则化 0-正无穷
            "lambda_l2":hyperopt_params["lambda_l2"],
            'n_jobs': -1,
            #'silent': 1,  # 信息输出设置成1则没有信息输出
            'seed': int(hyperopt_params['seed']),
            'bagging_freq':int(hyperopt_params['bagging_freq']),#装袋频率,0表示禁用装袋；k意味着在每次迭代时执行装袋k。每次k迭代，LightGBM 都会随机选择用于下一次迭代的数据bagging_fraction * 100 %k
            'is_unbalance': self.is_unbalances[hyperopt_params['is_unbalance']], #是否为不平衡数据
            "early_stopping_rounds":int(hyperopt_params["early_stopping_rounds"]),#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练
            "device_type":"cpu"#"cuda"
            #'scale_pos_weight': wt
        }  #设置出参数
        return params
    
    def run(self):
        t = 0
        for params_name,obj in self.param_grid_hp.items():
            t += 1
            log(f"已准备优化的第{t}个参数,名称:{params_name},类型:{obj}",LogLevel.PASS)

        self.Bayes_start_time = time.time()
        self.NOW_FUC_RUN_ITER = 0
        self.PARAMS_BEST, self.Trials = self.param_hyperopt(self.NUM_EVALS)
        self.bayes_opt_parser = self.parsing_bayes_params_for_LightGBM(self.PARAMS_BEST)
        
        # plot = partial(default_plot, height=40, width=160)
        # plot(self.historical_metrics,title=f"贝叶斯优化参数的运行历史{self.metrics_class}得分记录",color=True)
        
        log(f"解析参数得到结果:{self.bayes_opt_parser}",LogLevel.PASS)

    def test_params(self):
        gbm = lgb.LGBMClassifier(**self.bayes_opt_parser)
        gbm.fit(self.x_train, self.y_train, 
                # verbose_eval=True ,
                    eval_metric='auc',
                    # verbose = self.is_verbose,
            eval_set=[(self.x_train, self.y_train), (self.x_val, self.y_val)]
                # ,early_stopping_rounds=30
            )
        gbm_pred = gbm.predict(self.x_val)
        gbm_proba = gbm.predict_proba(self.x_val)[:,1]
        log(f'模型的评估报告1：\n,{classification_report(self.y_val, gbm_pred)}\n',LogLevel.SUCCESS)
        log(f'模型的评估报告2：\n,{self.TPRN_Score(self.y_val, gbm_pred)}\n',LogLevel.SUCCESS)
        
        
        # self.plot_roc(self.y_val, gbm_proba[:,1])
        if self.metrics_class == "pr-auc":
            metrics = self.PR_AUC(self.y_val, gbm_proba,gbm_pred)
        elif self.metrics_class == "roc-auc":
            metrics = self.ROC_AUC(self.y_val, gbm_proba)
        elif self.metrics_class == "f1-score":
            metrics = f1_score(self.y_val,gbm_pred)
        elif self.metrics_class == "recall":
            metrics = self.TPRN_Score(self.y_val,gbm_pred)["召回率|recall｜真阳率｜命中率"] 
        elif self.metrics_class == "precision":
            metrics = self.TPRN_Score(self.y_val,gbm_pred)["精确率|precision"] 
        elif self.metrics_class == "accuracy":
            metrics = self.TPRN_Score(self.y_val,gbm_pred)["ACC正确率|accuracy"]
        elif self.metrics_class == "roc-auc-recall":
            roc_auc = self.ROC_AUC(self.y_val, gbm_proba)
            recall = self.TPRN_Score(self.y_val,gbm_pred)["召回率|recall｜真阳率｜命中率"]
            metrics = roc_auc*self.metrics_weight[0]*recall*self.metrics_weight[-1]
        elif self.metrics_class == "roc-auc-recall-accuracy":
            roc_auc = self.ROC_AUC(self.y_val, gbm_proba)
            recall = self.TPRN_Score(self.y_val,gbm_pred)["召回率|recall｜真阳率｜命中率"]
            accuracy = self.TPRN_Score(self.y_val,gbm_pred)["ACC正确率|accuracy"]
            metrics = roc_auc*self.metrics_weight[0]+recall*self.metrics_weight[1]+accuracy*self.metrics_weight[-1]                      
        if self.metrics_class == "all":
            TPRN = self.TPRN_Score(self.y_val, gbm_pred)
            pr_auc = self.PR_AUC(self.y_val, gbm_proba,gbm_pred)
            roc_auc = self.ROC_AUC(self.y_val, gbm_proba)
            accuracy = TPRN["ACC正确率|accuracy"]
            precision = TPRN['精确率|precision']
            recall = TPRN["召回率|recall｜真阳率｜命中率"]
            false_alarm = TPRN['误报率|false alarm｜假阳率｜虚警率｜误检率'] 
            miss_rate = TPRN['漏报率|miss rate|也称为漏警率|漏检率']
            specificity = TPRN['特异度|specificity']
            f1score = f1_score(self.y_val,gbm_pred) 
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
    
    ####TODO：不插值计算
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42) 
    bol = BayesOptLightGBM(x,y,MAX_SHUFFLE=200
                           ,Folds=0
                           ,metrics_class="f1-score"
                           ,x_train=train_x,x_val=test_x
                           ,y_train=train_y,y_val=test_y)
    # bol = BayesOptLightGBM(x,y,MAX_SHUFFLE=200)
    bol.run()
    log(f"所有搜索相关记录：{bol.Trials.trials[0]}",LogLevel.INFO)
    bol.test_params()