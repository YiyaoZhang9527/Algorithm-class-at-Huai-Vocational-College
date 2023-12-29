from sklearn.metrics import precision_recall_curve,auc,f1_score,roc_curve,auc
from numpy import matrix


class MyMetric():
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def ROC_AUC(test_y, proba):
        fpr,tpr,threshold = roc_curve(test_y, proba)
        roc_auc_ = auc(fpr,tpr)
        return roc_auc_
        
    @staticmethod
    def PR_AUC(test_y,proba,pred):
        precision,recall,_ = precision_recall_curve(test_y,proba)
        f1 ,pr_auc = f1_score(test_y,pred),auc(recall,precision)
        return pr_auc
    
    @staticmethod
    def TPRN_Score(test_y,pred):
        TP = ((test_y==1) * (pred ==1)).sum()
        FN = ((test_y == 1) * (pred == 0)).sum()
        FP = ((test_y==0)*(pred ==1)).sum()
        TN = ((test_y==0)*(pred ==0)).sum()
        
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
    
        return {"混淆矩阵":matrix([[TP,FN],[FP,TN]])
            ,"ACC正确率|accuracy":ACC
            ,"精确率|precision":precision
            ,"召回率|recall｜真阳率｜命中率":TPR
            ,"误报率|false alarm｜假阳率｜虚警率｜误检率":FPR
            ,"漏报率|miss rate|也称为漏警率|漏检率":FNR
            ,"特异度|specificity":1-FPR
            ,"F1-score:":F1Score
            ,"真实正样本数":P,"真实负样本数":N}
        
        