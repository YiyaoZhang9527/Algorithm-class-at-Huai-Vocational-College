# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

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
from log_color import log,LogLevel
from sklearn import metrics
import time
import datetime
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets


cancer=datasets.load_breast_cancer()
x=cancer.data
y=cancer.target


def shuffle_data(x,y):
    """
    数据洗牌
    """
    xy = np.c_[x,y]
    for i in tqdm(range(100),desc="随机打乱数据"):
        np.random.shuffle(xy)
    x,y = xy[:,0:n],xy[:,n:n+1]
    y = np.ravel(y)
    return x,y

m,n = x.shape
y = np.array(y,dtype=int)
x,y = shuffle_data(x,y)
log(f"x.shape:{x.shape},y.shape:{y.shape}",LogLevel.INFO) 
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state = 42)

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


def plot_roc(y_test, y_score):
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

def parsing_bayes_params_for_LightGBM(hyperopt_params):
    """
    解析估计出的beyes最优参数
    """
    boostings = ["gbdt","rf","dart"]
    tree_learners = ["serial","feature","data","voting"]
    is_unbalances = [True,False]
    params = {
        'verbose':-1,
        'min_data_in_leaf': int(hyperopt_params['min_data_in_leaf']),#一片叶子中的数据数量最少。可以用来处理过拟​​合注意：这是基于 Hessian 的近似值，因此有时您可能会观察到分裂产生的叶节点的观测值少于这么多
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': int(hyperopt_params['num_leaves']), #一棵树的最大叶子数
        "boosting": boostings[hyperopt_params["boosting"]],#"gbdt",#gbdt 传统的梯度提升决策树,rf随机森林,dartDropout 遇到多个可加回归树,
        'n_estimators':int(hyperopt_params['n_estimators']),#2000,#基学习器
        "tree_learner":tree_learners[hyperopt_params["tree_learner"]],#"feature",#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
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
        'is_unbalance':is_unbalances[hyperopt_params['is_unbalance']], #是否为不平衡数据
        "early_stopping_rounds":int(hyperopt_params["early_stopping_rounds"]),#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练
        "device_type":"cpu"#"cuda"
        #'scale_pos_weight': wt
    }  #设置出参数
    return params

def hyperopt_objective(hyperopt_params):

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

    ####TODO:分层k折交叉验证，以处理数据极度不平衡问题
    global Folds
    strkf = StratifiedKFold(n_splits=Folds, shuffle=True)
    '''
    n_splits=6（默认5）：将数据集分成6个互斥子集，每次用5个子集数据作为训练集，1个子集为测试集，得到6个结果

    shuffle=True（默认False）：每次划分前数据重新洗牌，每次的运行结果不同；shuffle=False：每次运行结果相同，相当于random_state=整数
    random_state=1（默认None）：随机数设置为1，使得每次运行的结果一致
    '''
    roc_aucs = np.zeros(Folds)
    for i,index_ in enumerate(strkf.split(x,y)):
        train_index,test_index = index_
        # print(train_index,test_index)
        X_train_KFold, X_test_KFold = x[train_index],x[test_index]
        y_train_KFold, y_test_KFold = y[train_index],y[test_index]
        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(X_train_KFold, y_train_KFold, 
                # verbose_eval=True ,
                eval_metric='auc',
                eval_set=[(X_train_KFold, y_train_KFold), (X_test_KFold, y_test_KFold)]
                # ,early_stopping_rounds=30
            )
        gbm_pred = gbm.predict(X_test_KFold)
        gbm_proba = gbm.predict_proba(X_test_KFold)[:,1]
        #print(gbm_proba.sum(),y_test_KFold.sum())
        fpr,tpr,threshold = metrics.roc_curve(y_test_KFold, gbm_proba)
        roc_auc_ = metrics.auc(fpr,tpr)
        roc_aucs[i]= roc_auc_

    roc_auc  = roc_aucs.mean()

    ####TODO:单次迭代
    # gbm = lgb.LGBMClassifier(**params)
    # gbm.fit(x_train, y_train, 
    #     # verbose_eval=True ,
    #         eval_metric='auc',
    # eval_set=[(x_train, y_train), (x_val, y_val)]
    #     # ,early_stopping_rounds=30
    # )
    # gbm_pred = gbm.predict(x_val)
    # gbm_proba = gbm.predict_proba(x_val)[:,1]
    # fpr,tpr,threshold = metrics.roc_curve(y_val, gbm_proba)
    # roc_auc = metrics.auc(fpr,tpr)


    func_end = time.time()
    global NOW_FUC_RUN_ITER
    NOW_FUC_RUN_ITER += 1

    log(f"""本次迭代AUC分数为:[{roc_auc}],
        用时:[{func_end-func_start}]秒,
        当前优化第:[{NOW_FUC_RUN_ITER}]次,
        已运行:[{NOW_FUC_RUN_ITER}]次，
        用时总计:[{datetime.timedelta(seconds=(func_end-Bayes_start_time))}]秒,
        """,LogLevel.PASS)
    return -roc_auc

##TODO: 域空间，也就是给定超参数搜索的范围。这个可以通过 hyperopt 的 hp 方法实现，hp 方法有很多个参数选择的方式如
# hp.loguniform（对数空间搜索）
# 、hp.lognormal（对数正态分布）
# 、hp.normal（正态分布）
# 、hp.choice（列表选项搜索）
# 、hp.uniform（连续均匀分布搜索）
# 、hp.quniform（连续均匀分布整数搜索）

tree_learners = ["feature","serial","data","voting"]
boostings = ["gbdt","rf"]
is_unbalances = [True,False]

param_grid_hp = {
    'min_data_in_leaf': hp.quniform('min_data_in_leaf',5,1000,1),
    'num_leaves': hp.quniform("num_leaves",20,1000,1), #一棵树的最大叶子数
    "boosting": hp.choice("boosting",boostings),#gbdt 传统的梯度提升决策树,rf随机森林,dart Dropout 遇到多个可加回归树,
    'n_estimators':hp.quniform('n_estimators',80,2000,1),#基学习器
    "tree_learner": hp.choice("tree_learner",tree_learners),#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
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
    'is_unbalance':hp.choice('is_unbalance',is_unbalances),#True, #是否为不平衡数据
    "early_stopping_rounds":hp.quniform("early_stopping_rounds",1,1000,1),#30,#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练

}  #设置出参数




def param_hyperopt(max_evals=100):

    #保存迭代过程
    trials = Trials()

    #设置提前停止
    ## 如果损失没有增加，将在 X 次迭代后停止的停止函数
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


t = 0
for params_name,obj in param_grid_hp.items():
    t += 1
    log(f"已准备优化的第{t}个参数,名称:{params_name},类型:{obj}",LogLevel.PASS)

Bayes_start_time = time.time()
NOW_FUC_RUN_ITER = 0
Folds = 6 #几折交叉验证
NUM_EVALS = 600
params_best, trials = param_hyperopt(NUM_EVALS)

log(f"所有搜索相关记录：{trials.trials[0]}",LogLevel.INFO)

bayes_opt_parser = parsing_bayes_params_for_LightGBM(params_best)
log(f"解析参数得到结果:{bayes_opt_parser}",LogLevel.PASS)


params = bayes_opt_parser

gbm = lgb.LGBMClassifier(**params)
gbm.fit(x_train, y_train, 
        # verbose_eval=True ,
            eval_metric='auc',
    eval_set=[(x_train, y_train), (x_val, y_val)]
        # ,early_stopping_rounds=30
       )
gbm_pred = gbm.predict(x_val)
gbm_proba = gbm.predict_proba(x_val)
print('模型的评估报告：\n',classification_report(y_val, gbm_pred))
plot_roc(y_val, gbm_proba[:,1])
