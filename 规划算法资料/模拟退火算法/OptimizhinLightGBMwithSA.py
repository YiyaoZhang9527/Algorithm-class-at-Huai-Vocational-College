import pandas as pd
import numpy as np
import lightgbm as lgb
from sko.GA import GA
from sko.tools import set_run_mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from log_color import log,LogLevel
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
from sko.SA import SAFast
import time
import datetime
import os
from sklearn import datasets


cancer=datasets.load_breast_cancer()
x=cancer.data
y=cancer.target


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

 
# from matplotlib import pyplot as plt
# train_df = pd.read_csv('./train_v2.csv', index_col=0)
# test_df = pd.read_csv('./test_v2.csv', index_col=0)
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



# w1,w2,w3,w4,w5,w6,w,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19 = p

def params_logic(x):
    """
    处理一下 当： 'bagging_fraction': 1.0, 'feature_fraction': 1.0,时候
    'boosting': 'rf'会导致lightGBM模型 参数冲突的问题
    """
    if x[10] ==10 and x [11]== 10 and int(x[4]) == 2:
        index = np.random.randint(0,2)
        if index == 0:
            x[10] = 0.999999999
        elif index == 1:
            x[11] = 0.999999999
        else:
            x[2] = 0
    return x

def LightGBM_Func(x):
    boostings = ["gbdt","rf","dart"]
    tree_learners = ["serial","feature","data","voting"]

    x = params_logic(x)

    func_start = time.time()
    params = {
        'verbose':-1,
        'min_data_in_leaf': int(x[0]),#一片叶子中的数据数量最少。可以用来处理过拟​​合注意：这是基于 Hessian 的近似值，因此有时您可能会观察到分裂产生的叶节点的观测值少于这么多
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': int(x[1]), # 50,一棵树的最大叶子数
        "boosting": boostings[int(x[2])],#"gbdt",#gbdt 传统的梯度提升决策树,rf随机森林,dartDropout 遇到多个可加回归树,
        'n_estimators':int(x[3]),#2000,#基学习器
        "tree_learner":tree_learners[int(x[4])],#"feature",#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
        'max_bin': int(x[5]), #直方图分箱特征值的将被存储的最大箱数，少量的 bin 可能会降低训练准确性，但可能会增加处理过度拟合的能力
        "min_data_in_bin":int(x[6]), #一个 bin 内的数据最少数量.使用它可以避免一数据一箱（潜在的过度拟合)
        'max_depth':int(x[7]), #限制树模型的最大深度
        "learning_rate": x[8],#学习率
        #"colsample_bytree": 0.8,  
        "bagging_fraction": x[9],  # 每次迭代时用的数据比例，但这将随机选择部分数据而不重新采样，可用于加速训练可以用来处理过拟​​合
        "feature_fraction":x[10], # 每次迭代中随机选择特征的比例，lightGBM 将在每次迭代（树）上随机选择特征子集1.0。例如，如果将其设置为0.8，LightGBM 将在训练每棵树之前选择 80% 的特征
        "lambda_l1":x[11], #L1正则化 0-正无穷
        "lambda_l2":x[12],
        'n_jobs': -1,
        #'silent': 1,  # 信息输出设置成1则没有信息输出
        'seed': int(x[13]),
        'bagging_freq':int(x[14]),#装袋频率,0表示禁用装袋；k意味着在每次迭代时执行装袋k。每次k迭代，LightGBM 都会随机选择用于下一次迭代的数据bagging_fraction * 100 %k
        'is_unbalance':bool(int(x[15])), #是否为不平衡数据
        "early_stopping_rounds":int(x[16])<1 and 1 or int(x[16]),#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练
        "device_type":"cpu"#"cuda"
        #'scale_pos_weight': wt
    }  #设置出参数
    log(f"本次参数为:[{params}]",LogLevel.INFO)
    try:
        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(x_train, y_train, 
                # verbose_eval=True ,
                    eval_metric='auc',
            eval_set=[(x_train, y_train), (x_val, y_val)]
                # ,early_stopping_rounds=30
            )
        gbm_pred = gbm.predict(x_val)
        gbm_proba = gbm.predict_proba(x_val)[:,1]
        fpr,tpr,threshold = metrics.roc_curve(y_val, gbm_proba)
        roc_auc = metrics.auc(fpr,tpr)
        dtrain=None
        clf = None
        dtest = None
        lr_proba = None
        fpr,tpr,threshold = None,None,None
    except Exception as e:
        
        roc_auc = 0

    func_end = time.time()
    ### 写入优化参数日志
    params_log_path = "./lightGBM_opt_params_log_binary.csv"
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
        用时总计:[{datetime.timedelta(seconds=(func_end-SA_start_time))}]秒,
        """,LogLevel.PASS)
    
    return -roc_auc
    

SA_start_time = time.time()
NOW_FUC_RUN_ITER = 0

SA_params_table = pd.read_csv("SA_params_for_LightGMB.csv")
lbs = SA_params_table["lb"].to_list()
print(f"lbs_len:{len(lbs)}")
ubs = SA_params_table["ub"].to_list()
X0s = SA_params_table["x0"].to_list()

print(f"""lbs:{lbs},ubs:{ubs},x0s:{X0s}""")


sa = SAFast(func=LightGBM_Func
        , x0=X0s # 初始x解，初始解越大则越难到达最小值,越小则越容易错过
        , T_max=1 #系统的温度，系统初始应该要处于一个高温的状态 初始温度越高，且马尔科夫链越长，算法搜索越充分，得到全局最优解的可能性越大，但这也意味着需要耗费更多的计算时间
        , T_min=0.9  #温度的下限，若温度T达到T_min，则停止搜索
        , L=300 #最大迭代次数,每个温度下的迭代次数（又称链长）
        , max_stay_counter=10 # 最大冷却停留计数器，保证快速退出,如果 best_y 在最大停留计数器次数（也称冷却时间）内保持不变，则停止运行
        ,lb = lbs #x的下限
        ,ub = ubs #x的上限
        #,hop = [3,2,1] # x 的上下限最大差值 hop=ub-lb 
        ,m = 1 # 0-正无穷，越大，越容易冷却退出
        ,n = 1 # # 0-正无穷，越大，越不容易冷却退出
        ,quench = 1 # 淬火指数，0-正无穷，越小则越慢，但是越能求出最小，越大则越快，但是容易陷入局部最优
       )

best_x, best_y = sa.run()
print('best_x:', best_x, 'best_y:', best_y,"y_history:",len(sa.best_y_history),sa.iter_cycle)

boostings = ["gbdt","rf","dart"]
tree_learners = ["serial","feature","data","voting"]
x = best_x

params = {
        'verbose':-1,
        'min_data_in_leaf': int(x[0]),#一片叶子中的数据数量最少。可以用来处理过拟​​合注意：这是基于 Hessian 的近似值，因此有时您可能会观察到分裂产生的叶节点的观测值少于这么多
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': int(x[1]), #一棵树的最大叶子数
        "boosting": boostings[int(x[2])],#"gbdt",#gbdt 传统的梯度提升决策树,rf随机森林,dartDropout 遇到多个可加回归树,
        'n_estimators':int(x[3]),#2000,#基学习器
        # "num_leaves":int(x[4]),#50, #一棵树的最大叶子数
        "tree_learner":tree_learners[int(x[4])],#"feature",#serial:单机树学习器,feature:特征并行树学习器,data:数据并行树学习器，voting:投票并行树学习器
        'max_bin': int(x[5]), #直方图分箱特征值的将被存储的最大箱数，少量的 bin 可能会降低训练准确性，但可能会增加处理过度拟合的能力
        "min_data_in_bin":int(x[6]), #一个 bin 内的数据最少数量.使用它可以避免一数据一箱（潜在的过度拟合)
        'max_depth':int(x[7]), #限制树模型的最大深度
        #"min_data_in_leaf":int(x[8]),#一片叶子中的数据数量最少。可以用来处理过拟合
        "learning_rate": x[8],#学习率
        #"colsample_bytree": 0.8,  
        "bagging_fraction": x[9],  # 每次迭代时用的数据比例，但这将随机选择部分数据而不重新采样，可用于加速训练可以用来处理过拟​​合
        "feature_fraction":x[10], # 每次迭代中随机选择特征的比例，lightGBM 将在每次迭代（树）上随机选择特征子集1.0。例如，如果将其设置为0.8，LightGBM 将在训练每棵树之前选择 80% 的特征
        "lambda_l1":x[11], #L1正则化 0-正无穷
        "lambda_l2":x[12],
        'n_jobs': -1,
        #'silent': 1,  # 信息输出设置成1则没有信息输出
        'seed': int(x[13]),
        'bagging_freq':int(x[14]),#装袋频率,0表示禁用装袋；k意味着在每次迭代时执行装袋k。每次k迭代，LightGBM 都会随机选择用于下一次迭代的数据bagging_fraction * 100 %k
        'is_unbalance':bool(int(x[15])), #是否为不平衡数据
        "early_stopping_rounds":int(x[16])<1 and 1 or int(x[16]),#早停法 如果一个验证数据的一个指标在最后几轮中没有改善，将停止训练
        "device_type":"cpu"#"cuda"
        #'scale_pos_weight': wt
    }  #设置出参数
best_lightGBM_params_binary_path = "best_lightGBM_params_binary.csv"
params.update({"best_auc":best_y})
best_params_table = pd.DataFrame({k:[v] for k,v in params.items()})
if os.path.exists(best_lightGBM_params_binary_path ):
    best_params_table.to_csv(best_lightGBM_params_binary_path ,mode='a', header=False, index=None)
else:
    best_params_table.to_csv(best_lightGBM_params_binary_path ,index=False)