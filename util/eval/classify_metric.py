#coding=utf-8
'''
本库以numpy和pandas为基本数据结构，通过计算混淆矩阵，并以此为基础，计算分类相关评价标准
混淆矩阵的参考：https://blog.csdn.net/joeland209/article/details/71078935
https://blog.csdn.net/dgyuanshaofeng/article/details/78686117
'''
import numpy as np
import pandas as pd
import logging
import sklearn.metrics as metrics
from collections import defaultdict

MIN_VAL = 10e-13
np.set_printoptions(2)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_confusion_matrix_df(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，返回混淆矩阵的datafame形式，行列索引为label
    然后计算总体accuracy、每个类别以及所有类别平均的precision、recall、F1
    :param true_labels: 真实标记列表，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记列表，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    # logging.info('Get confusion matrix dataframe')

    get_confusion_matrix_df.__name__ = 'confusion_matrix'
    if unique_labels is None:   #如果unique为空
        unique_labels = set(true_labels) | set(pred_labels)
    # 排序的标记集合，索引号即为混淆矩阵中对应的索引号
    sort_unique_labels=sorted(unique_labels)
    cm_df = pd.DataFrame(data=np.zeros(shape=(len(sort_unique_labels),len(sort_unique_labels))), 
                         index=sort_unique_labels,
                         columns=sort_unique_labels)    #建立一个全0的混淆矩阵
    for true_label,pred_label in zip(true_labels,pred_labels):
        if true_label in sort_unique_labels and pred_label in sort_unique_labels:
        #遍历每一对标记
            cm_df.loc[true_label,pred_label]+=1
    
    # # 将true_labels,pred_labels中的标记转化为索引
    # true_classes = [sort_unique_labels.index(true_label) for true_label in true_labels]
    # pred_classes = [sort_unique_labels.index(pred_label) for pred_label in pred_labels]
    # cn = len(sort_unique_labels)  # 类别数量 class number
    # cm = np.zeros((cn, cn))  # 定义混淆矩阵 confusion matrix
    # for true_class, pred_class in zip(true_classes, pred_classes):
    #     cm[true_class, pred_class] += 1
    # cm_df=pd.DataFrame(data=cm,index=sort_unique_labels,columns=sort_unique_labels)

    return cm_df

def get_overall_accuracy(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，计算分类的总体accuracy
    :param true_labels:  真实标记，列表，例如['A','B','C','A']
    :param pred_labels:  预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    # logging.info('Get overall accuracy')

    get_overall_accuracy.__name__ = 'overall_accuracy'
    # get_overall_accuracy.metric = 'overall_accuracy'
    # cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
    # accuracy = np.sum(np.diag(cm_df)) / np.sum(np.array(cm_df)) #不转换成ndarray不行
    # accuracy=metrics.accuracy_score(true_labels,pred_labels)
    correct_num=sum([true_label==pred_label for true_label,pred_label in zip(true_labels,pred_labels)])
    accuracy=correct_num/len(true_labels)
    return accuracy

def get_precision_series(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的precision并构建成series结构
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    # logging.info('Get precision series')
    
    get_precision_series.__name__ = 'precision'
    # get_precision_series.metric = 'precision'
    cm_df=get_confusion_matrix_df(true_labels,pred_labels,unique_labels)    #构建混淆矩阵
    pred_labels_series = np.sum(cm_df, axis=0)    #series
    precision_series = pd.Series([cm_df.iloc[i, i] / (pred_labels_series.values[i] + MIN_VAL) for i in range(0, len(pred_labels_series))],
                                 index=cm_df.index.tolist())    #计算每个类的prescision组成series数据结构
    return precision_series

def get_macro_precision(true_labels, pred_labels,unique_labels=None,is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的precision
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return: 
    '''
    # logging.info('Get average precision')

    get_macro_precision.__name__ = 'macro_precision'
    # get_average_precision.metric = 'average_precision'
    precision_series=get_precision_series(true_labels,pred_labels,unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
        class_ratio_series=np.sum(cm_df, axis=1)/np.sum(np.array(cm_df)) #不转换成ndarray不行
        average_precision=float(np.sum(precision_series*class_ratio_series))
    else:
        average_precision= float(np.mean(precision_series))
    return average_precision

# get_macro_precision=get_average_precision

def get_recall_series(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的recall并构建成series结构
    recall貌似就是每个类预测正确实例的正确率，recall就是sensitivity！！！
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    # logging.info('Get recall series')

    get_recall_series.__name__ = 'recall'
    # get_recall_series.metric = 'recall'
    cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels) #构建混淆矩阵
    true_labels_series = np.sum(cm_df, axis=1)  # series
    recall_series = pd.Series([cm_df.iloc[i, i] / (true_labels_series.values[i] + MIN_VAL) for i in range(0, len(true_labels_series))],
                              index=cm_df.index.tolist())  # 计算每个类的recall组成series数据结构
    return recall_series

def get_macro_recall(true_labels, pred_labels,unique_labels=None,is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的recall
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return: 
    '''
    # logging.info('Get average recall')

    get_macro_recall.__name__ = 'macro_recall'
    # get_average_recall.metric = 'average_recall'
    recall_series=get_recall_series(true_labels,pred_labels,unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
        class_ratio_series=np.sum(cm_df, axis=1)/np.sum(np.array(cm_df)) #不转换成ndarray不行
        average_recall=float(np.sum(recall_series*class_ratio_series))
    else:
        average_recall= float(np.mean(recall_series))
    return average_recall

# get_macro_recall=get_average_recall

def get_F_score_series(true_labels, pred_labels,unique_labels=None,alpha=1.):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的precision和recalld的series数据结构
    然后根据precision和recall，以及给出的调和值alpha计算F-score
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param alpha: 综合precision和recall的调和值，默认1，即求F1-score
    :return:
    '''
    # logging.info('Get F score series')

    get_F_score_series.__name__ = 'F_score'
    # get_F_score_series.metric = 'F_score'
    precision_series=get_precision_series(true_labels, pred_labels,unique_labels)   #precision series
    recall_series=get_recall_series(true_labels, pred_labels) #recall series
    F_score_series = (1+pow(alpha,2)) * precision_series * recall_series / \
                     (pow(alpha,2)*precision_series + recall_series + MIN_VAL)
    return F_score_series

def get_macro_F_score(true_labels, pred_labels,unique_labels=None,is_weight=False,alpha=1.):
    '''
    根据真实标记和预测标记，以及给出的调和值alpha,计算所有类别平均的F_score
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :param alpha: 综合precision和recall的调和值，默认1，即求F1-score
    :return:
    '''
    # logging.info('Get average F score')

    get_macro_F_score.__name__ = 'macro_F_score'
    # get_average_F_score.metric = 'average_F_score'
    F_score_series=get_F_score_series(true_labels,pred_labels,unique_labels,alpha)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
        class_ratio_series=np.sum(cm_df, axis=1)/np.sum(np.array(cm_df)) #不转换成ndarray不行
        average_F_score=float(np.sum(F_score_series*class_ratio_series))
    else:
        average_F_score= float(np.mean(F_score_series))
    return average_F_score

def get_F1_score_series(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算每个类别的precision和recall的series数据结构
    然后根据precision和recall计算F1 score
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    # logging.info('Get F1 score series')

    get_F1_score_series.__name__ = 'F1_score'
    # get_F1_score_series.metric = 'F1_score'
    # precision_series=get_precision_series(true_labels, pred_labels)   #precision series
    # recall_series=get_recall_series(true_labels, pred_labels) #recall series
    # F1_score_series = 2.0 * precision_series * recall_series / (precision_series + recall_series + MIN_VAL)
    # return F1_score_series
    F1_score_series=get_F_score_series(true_labels, pred_labels,unique_labels,alpha=1)
    return F1_score_series

def get_average_F1_score(true_labels, pred_labels,unique_labels=None,is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的F1_score
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return: 
    '''
    # logging.info('Get average F1 score')

    get_average_F1_score.__name__ = 'average_F1_score'
    # get_average_F1_score.metric = 'average_F1_score'
    F1_score_series=get_F1_score_series(true_labels,pred_labels,unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
        class_ratio_series=np.sum(cm_df, axis=1)/np.sum(np.array(cm_df)) #不转换成ndarray不行
        average_F1_score=float(np.sum(F1_score_series*class_ratio_series))
    else:
        average_F1_score= float(np.mean(F1_score_series))
    return average_F1_score

def get_macro_F1_score(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记,计算所有macro F1 score,即类别平均的F1_score,同于get_average_F1_score
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    return metrics.f1_score(true_labels,pred_labels,labels=unique_labels,average='macro')

def get_micro_F1_score(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记,计算所有micro F1 score
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    return metrics.f1_score(true_labels,pred_labels,labels=unique_labels,average='micro')

def get_sensitivity_series(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，算每个类别的敏感性或灵敏性sensitivity, 即召回率或查全率recall
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    logging.info('Get sensitivity(=recall) series')

    get_sensitivity_series.__name__ = 'sensitivity'
    # get_sensitivity_series.metric = 'sensitivity'
    sensitivity_series=get_recall_series(true_labels, pred_labels,unique_labels)
    return sensitivity_series

def get_average_sensitivity(true_labels, pred_labels,unique_labels=None,is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的sensitivity,即平均recall
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    # logging.info('Get average sensitivity(=recall)')

    get_average_sensitivity.__name__ = 'average_sensitivity'
    # get_average_sensitivity.metric = 'average_sensitivity'
    average_sensitivity=get_average_recall(true_labels, pred_labels,unique_labels,is_weight)
    return average_sensitivity

def get_specificity_series(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算所有类别的特异性specificity
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return:
    '''
    # logging.info('Get specificity series')

    get_specificity_series.__name__ = 'specificity'
    # get_specificity_series.metric = 'specificity'
    cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
    # specificity_series=pd.Series(index=cm_df.index.tolist())
    diag_arr1 = np.diag(cm_df)  #取出对角线
    diag_sum=np.sum(diag_arr1)  #对角线求和
    pred_labels_arr1 = np.sum(cm_df, axis=0).values #每一类预测到的数量，并转成ndarray
    # cn = cm_df.shape[0] #不同类别的数量
    # specificities=[]
    # for i in range(0,cn):
    #     #根据公式计算每类的特异性，公式里是加，这里反着用减
    #     specificities.append((diag_sum-diag_arr1[i])/(pred_labels_arr1[i]-diag_arr1[i]+diag_sum-diag_arr1[i]))
    specificities=[(diag_sum-diag_arr1[i])/(pred_labels_arr1[i]+diag_sum-2*diag_arr1[i])
                   for i in range(0,cm_df.shape[0])]    #根据公式计算每类的特异性，公式里是加，这里反着用减
    #将list转换成series
    specificity_series=pd.Series(data=specificities,index=cm_df.index.tolist())
    return specificity_series

def get_average_specificity(true_labels, pred_labels,unique_labels=None,is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的specificity
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return: 
    '''
    # logging.info('Get average specificity')

    get_average_specificity.__name__ = 'average_specificity'
    # get_average_specificity.metric = 'average_specificity'
    specificity_series=get_specificity_series(true_labels,pred_labels,unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
        class_ratio_series=np.sum(cm_df, axis=1)/np.sum(np.array(cm_df)) #不转换成ndarray不行
        average_specificity=float(np.sum(specificity_series*class_ratio_series))
    else:
        average_specificity= float(np.mean(specificity_series))
    return average_specificity

def get_balanced_accuracy_series(true_labels, pred_labels,unique_labels=None):
    '''
    根据真实标记和预测标记，计算每个类别的sensitivity和specificity
    继而计算每个类别的balanced_accuracy，并构建成series结构
    balanced_accuracy=（sensitivity+specificity）/2
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    # logging.info('Get balanced accuracy series')

    get_balanced_accuracy_series.__name__ = 'balanced_accuracy'
    # get_balanced_accuracy_series.metric = 'balanced_accuracy'
    sensitivity_series=get_sensitivity_series(true_labels, pred_labels,unique_labels)
    specificity_series=get_specificity_series(true_labels, pred_labels,unique_labels)
    balanced_accuracy=(sensitivity_series+specificity_series)/2
    return balanced_accuracy

def get_average_balanced_accuracy(true_labels, pred_labels,unique_labels=None,is_weight=False):
    '''
    根据真实标记和预测标记，计算所有类别平均的balanced_accuracy
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :param is_weight: 是否加权平均，默认不加权
    :return: 
    '''
    # logging.info('Get average balanced accuracy')

    get_average_balanced_accuracy.__name__ = 'average_balanced_accuracy'
    # get_average_balanced_accuracy.metric = 'average_balanced_accuracy'
    balanced_accuracy_series = get_balanced_accuracy_series(true_labels, pred_labels,unique_labels)
    if is_weight == True:
        cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)  # 构建混淆矩阵
        class_ratio_series = np.sum(cm_df, axis=1) / np.sum(np.array(cm_df))  # 不转换成ndarray不行
        average_balanced_accuracy = float(np.sum(balanced_accuracy_series * class_ratio_series))
    else:
        average_balanced_accuracy = float(np.mean(balanced_accuracy_series))
        # if isinstance(average_balanced_accuracy,float):
        #     return average_balanced_accuracy
    return average_balanced_accuracy

def get_corr_value(true_values,pred_values,method='pearson'):
    '''
    计算真实值列表与预测值列表之间的相关系数
    :param true_values: 真实值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.2]
    :param pred_values: 预测值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.0]
    :param method : {'pearson', 'kendall', 'spearman'}
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
    :return:    相关系数
    '''
    assert len(true_values)==len(pred_values)
    df=pd.DataFrame([true_values,pred_values]).T
    df.columns=['true_value','pred_value']
    corr_df=df.corr(method=method)
    # print(corr_df)
    return corr_df.loc['true_value','pred_value']

def get_pearson_corr_val(true_values,pred_values):
    '''
    计算真实值列表与预测值列表之间的pearson相关系数
    :param true_values: 真实值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.2]
    :param pred_values: 预测值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.0]
    :return:    相关系数
    '''
    return get_corr_value(true_values,pred_values,method='pearson')

def get_spearman_corr_val(true_values,pred_values):
    '''
    计算真实值列表与预测值列表之间的spearman相关系数
    :param true_values: 真实值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.2]
    :param pred_values: 预测值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.0]
    :return:    相关系数
    '''
    return get_corr_value(true_values,pred_values,method='spearman')

def get_kendall_corr_val(true_values,pred_values):
    '''
    计算真实值列表与预测值列表之间的kendall相关系数
    :param true_values: 真实值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.2]
    :param pred_values: 预测值列表，例如[1.3, 2.3, 0.3, 3.4, 2.1, 2.0]
    :return:    相关系数
    '''
    return get_corr_value(true_values,pred_values,method='kendall')

#函数的__name__属性在函数运行前不会改变，要改变只有两种方式，要么函数运行后，要么定义在函数后面
get_confusion_matrix_df.__name__ = 'confusion_matrix'
get_overall_accuracy.__name__ = 'overall_accuracy'
get_precision_series.__name__ = 'precision'
# get_average_precision.__name__ = 'average_precision'
get_macro_precision.__name__='macro_precision'
get_recall_series.__name__ = 'recall'
# get_average_recall.__name__ = 'average_recall'
get_macro_recall.__name__='macro_recall'
get_F_score_series.__name__ = 'F_score'
get_macro_F_score.__name__ = 'macro_F_score'
get_F1_score_series.__name__ = 'F1_score'
get_average_F1_score.__name__ = 'average_F1_score'
get_macro_F1_score.__name__='macro_F1_score'
get_micro_F1_score.__name__='micro_F1_score'
get_sensitivity_series.__name__ = 'sensitivity'
get_average_sensitivity.__name__ = 'average_sensitivity'
get_specificity_series.__name__ = 'specificity'
get_average_specificity.__name__ = 'average_specificity'
get_balanced_accuracy_series.__name__ = 'balanced_accuracy'
get_average_balanced_accuracy.__name__ = 'average_balanced_accuracy'

get_corr_value.__name__='correlation_value'
get_pearson_corr_val.__name__='pearson_correlation_value'
get_spearman_corr_val.__name__='spearman_correlation_value'
get_kendall_corr_val.__name__='kendall_correlation_value'

def get_eval_ex(true_labels, pred_labels,unique_labels=None, is_weight=False):
    '''
    根据真实标记和预测标记，构建混淆矩阵，然后计算总体accuracy、每个类别以及所有类别平均的precision、recall、F1
    :param true_labels: 真实标记，列表，例如['A','B','C','A']
    :param pred_labels: 预测标记，列表，例如['A','C','B','B']
    :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
    :return:
    '''
    logging.info('Get an integrated example of eval results')

    if unique_labels is None:   #如果unique为空
        unique_labels = set(true_labels) | set(pred_labels)
    # 排序的标记集合，索引号即为混淆矩阵中对应的索引号
    sort_unique_labels=sorted(unique_labels)
    # 将true_labels,pred_labels中的标记转化为索引
    true_classes = [sort_unique_labels.index(true_label) for true_label in true_labels]
    pred_classes = [sort_unique_labels.index(pred_label) for pred_label in pred_labels]
    class_counts_arr1 = np.array([true_classes.count(sort_unique_labels.index(label)) for label in sort_unique_labels])  # 每种类别的数量
    class_ratio_arr1 = class_counts_arr1 / len(true_classes)
    # MIN_VAL = 10e-8
    cn = len(sort_unique_labels)  # 类别数量 class number
    # np.set_printoptions(2)
    cm = np.zeros((cn, cn))  # 定义混淆矩阵 confusion matrix
    for true_class, pred_class in zip(true_classes, pred_classes):
        cm[true_class, pred_class] += 1
    true_label_arr1 = np.sum(cm, axis=1)
    pred_label_arr1 = np.sum(cm, axis=0)
    precision_arr1 = np.array([cm[i, i] / (pred_label_arr1[i] + MIN_VAL) for i in range(0, cn)])
    recall_arr1 = np.array([cm[i, i] / (true_label_arr1[i] + MIN_VAL) for i in range(0, cn)])
    F1_score_arr1 = 2.0 * precision_arr1 * recall_arr1 / (precision_arr1 + recall_arr1 + MIN_VAL)
    # 行为各类别，列为precision，recall，F1
    p_r_F1_arr2 = np.transpose(np.array([precision_arr1, recall_arr1, F1_score_arr1]))
    if is_weight == True:
        average_precision = float(np.sum(precision_arr1 * class_ratio_arr1))
        average_recall = float(np.sum(recall_arr1 * class_ratio_arr1))
        average_F1_score = float(np.sum(F1_score_arr1 * class_ratio_arr1))
    else:
        average_precision = float(np.mean(precision_arr1))
        average_recall = float(np.mean(recall_arr1))
        average_F1_score = float(np.mean(F1_score_arr1))
    average_p_r_F1_arr1 = np.array([average_precision, average_recall, average_F1_score])  # 所有类别平均的precision，recall，F1

    # 行为各类别，最后一行为平均值，列为precision，recall，F1
    res_p_r_F1_arr2 = np.zeros((cn + 1, 3))
    res_p_r_F1_arr2[:-1, :] = p_r_F1_arr2
    res_p_r_F1_arr2[-1, :] = average_p_r_F1_arr1

    # 总体accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    # 构建dataframe
    if is_weight == True:
        index = sort_unique_labels + ['Weighted_Average']
    else:
        index = sort_unique_labels + ['Average']
    columns = ['Precision(%)', 'Recall(%)', 'F1(%)']
    res_p_r_F1_a_df = pd.DataFrame(res_p_r_F1_arr2 * 100, index=index, columns=columns)

    res_p_r_F1_a_df.loc['Accuracy(%)', :] = accuracy * 100
    res_p_r_F1_a_df.loc['Accuracy(%)', 1:] = None

    res_p_r_F1_a_df.loc[:, 'Class_Count'] = None
    res_p_r_F1_a_df.loc[:-2, 'Class_Count'] = class_counts_arr1
    res_p_r_F1_a_df.loc[:, 'Class_Ratio'] = None
    res_p_r_F1_a_df.loc[:-2, 'Class_Ratio'] = class_ratio_arr1

    return res_p_r_F1_a_df

def get_span_micro_F1(true_labels,pred_labels,seq_lens,out2tag,tag2span_func):
    '''
    序列标注问题中，评价获取micro F1值
    :param true_labels:
    :param pred_labels:
    :param seq_lens:
    :param out2tag:
    :param tag2span_func:
    :return:
    '''
    span_metric_obj=SeqSpanMetric(out2tag,tag2span_func)
    return span_metric_obj.get_micro_F1_score(true_labels,pred_labels,seq_lens)

def get_span_micro_precision(true_labels,pred_labels,seq_lens,out2tag,tag2span_func):
    '''
    序列标注问题中，评价获取micro precision
    :param true_labels:
    :param pred_labels:
    :param seq_lens:
    :param out2tag:
    :param tag2span_func:
    :return:
    '''
    span_metric_obj=SeqSpanMetric(out2tag,tag2span_func)
    return span_metric_obj.get_micro_precision(true_labels,pred_labels,seq_lens)

def get_span_micro_recall(true_labels,pred_labels,seq_lens,out2tag,tag2span_func):
    '''
    序列标注问题中，评价获取micro recall值
    :param true_labels:
    :param pred_labels:
    :param seq_lens:
    :param out2tag:
    :param tag2span_func:
    :return:
    '''
    span_metric_obj=SeqSpanMetric(out2tag,tag2span_func)
    return span_metric_obj.get_micro_recall(true_labels,pred_labels,seq_lens)


# def get_seq_overall_accuracy(true_labels,pred_labels,seq_lens,out2tag,tag2span_func):
#     '''
#     序列标注问题中，评价获取micro F1值
#     :param true_labels:
#     :param pred_labels:
#     :param seq_lens:
#     :param out2tag:
#     :param tag2span_func:
#     :return:
#     '''
#     span_metric_obj=SeqSpanMetric(out2tag,tag2span_func)
#     return span_metric_obj.get_overall_accuracy(true_labels,pred_labels,seq_lens)


get_span_micro_F1.__name__='micro_F1_score'
get_span_micro_precision.__name__='micro_precision'
get_span_micro_recall.__name__='micro_recall'
# get_seq_overall_accuracy.__name__='overall_accuracy'

class SeqSpanMetric(object):
    def __init__(self,out2tag,tag2span_func):
        '''
        在序列标注问题中以span的方式进行评价.
        比如POS中，会以character的方式进行标注，某个句子的POS为
        ['B-NN', 'E-NN', 'S-DET', 'B-NN', 'E-NN']。'B-NN','E-NN'会合并成'NN'。
        # :param true_labels: 真实label列表，二维列表或numpy，
        #             例如[[1，2，3，2，1，0，0],[2,1,3,1,0,0,0]]或[[1,2,3,2,1].[2,1,3,1]]
        # :param pred_labels: 预测标记列表，二维列表或numpy，
        #             例如[[2，1，3，2，1，0，0],[2,3,3,1,0,0,0]]或[[2,1,3,2,1].[2,3,3,1]]
        :param out2tag: label到真实标记tag的映射
        # :param seq_lens: 每个序列的长度
        :param tag2span_func: tag到span_tag的映射函数
        :param alpha: 计算F1时的alpha值
        # :param unique_labels: 不同的标记列表，为None时统计true_labels和pred_labels中不同的标记
        '''
        # self.true_labels=true_labels
        # self.pred_labels=pred_labels
        self.out2tag=out2tag
        # self.seq_lens=seq_lens
        self.tag2span_func=tag2span_func
        # self.alpha=alpha
        # self.unique_labels=unique_labels
        # self.span2tp=defaultdict(int)    #true positive count of each span tag
        # self.span2fp=defaultdict(int)    #false positive count of each span tag
        # self.span2fn=defaultdict(int)    #false negative count of each span tag

    def _count(self,true_labels,pred_labels,seq_lens):
        '''
        :param true_labels: 真实label列表，二维列表或numpy，
                    例如[[1，2，3，2，1，0，0],[2,1,3,1,0,0,0]]或[[1,2,3,2,1].[2,1,3,1]]
        :param pred_labels: 预测标记列表，二维列表或numpy，
                    例如[[2，1，3，2，1，0，0],[2,3,3,1,0,0,0]]或[[2,1,3,2,1].[2,3,3,1]]
        :param seq_lens: 每个序列的长度
        :return: 每个类别的true positive count，false positive count和false negative count
        '''
        span2tp = defaultdict(int)  # true positive count of each span tag
        span2fp = defaultdict(int)  # false positive count of each span tag
        span2fn = defaultdict(int)  # false negative count of each span tag
        for true_label_piece,pred_label_piece,seq_len in zip(true_labels,pred_labels,seq_lens):
            # true_label_piece=true_label_piece[:seq_len]
            # pred_label_piece=pred_label_piece[:seq_len]
            #获取每个序列标注的真实tag字符串
            true_tag_piece=[self.out2tag[label] for label in true_label_piece[:seq_len]]
            pred_tag_piece=[self.out2tag[label] for label in pred_label_piece[:seq_len]]
            #获取每个序列标注的span元组(span,(start_pos,end_pos))，然后组成的列表
            true_span_tuples=self.tag2span_func(true_tag_piece)
            pred_span_tuples=self.tag2span_func(pred_tag_piece)
            for pred_span_tuple in pred_span_tuples:
                if pred_span_tuple in true_span_tuples:
                    span2tp[pred_span_tuple[0]] += 1
                    true_span_tuples.remove(pred_span_tuple)
                else:
                    span2fp[pred_span_tuple[0]] += 1
            for true_span_tuple in true_span_tuples:
                span2fn[true_span_tuple[0]] += 1
        return span2tp,span2fp,span2fn


    def get_micro_F1_score(self,true_labels,pred_labels,seq_lens):
        '''
        计算获取micro_F1 value
        :param true_labels: 真实label列表，二维列表或numpy，
                    例如[[1，2，3，2，1，0，0],[2,1,3,1,0,0,0]]或[[1,2,3,2,1].[2,1,3,1]]
        :param pred_labels: 预测标记列表，二维列表或numpy，
                    例如[[2，1，3，2，1，0，0],[2,3,3,1,0,0,0]]或[[2,1,3,2,1].[2,3,3,1]]
        :param seq_lens: 每个序列的长度
        :return:
        '''
        # get_micro_F1_score.__name__='micro_F1_score'
        span2tp,span2fp,span2fn=self._count(true_labels,pred_labels,seq_lens)
        # print('tp:',sum(span2tp.values()),'fp:',sum(span2fp.values()),'fn:',sum(span2fn.values()))
        f,_,_=self.get_F_P_R(sum(span2tp.values()),sum(span2fp.values()),sum(span2fn.values()),alpha=1.)
        return f

    def get_micro_precision(self,true_labels,pred_labels,seq_lens):
        '''
        计算获取micro_F1 value
        :param true_labels: 真实label列表，二维列表或numpy，
                    例如[[1，2，3，2，1，0，0],[2,1,3,1,0,0,0]]或[[1,2,3,2,1].[2,1,3,1]]
        :param pred_labels: 预测标记列表，二维列表或numpy，
                    例如[[2，1，3，2，1，0，0],[2,3,3,1,0,0,0]]或[[2,1,3,2,1].[2,3,3,1]]
        :param seq_lens: 每个序列的长度
        :return:
        '''
        # get_micro_F1_score.__name__='micro_F1_score'
        span2tp,span2fp,span2fn=self._count(true_labels,pred_labels,seq_lens)
        _,p,_=self.get_F_P_R(sum(span2tp.values()),sum(span2fp.values()),sum(span2fn.values()),alpha=1.)
        return p

    def get_micro_recall(self,true_labels,pred_labels,seq_lens):
        '''
        计算获取micro_F1 value
        :param true_labels: 真实label列表，二维列表或numpy，
                    例如[[1，2，3，2，1，0，0],[2,1,3,1,0,0,0]]或[[1,2,3,2,1].[2,1,3,1]]
        :param pred_labels: 预测标记列表，二维列表或numpy，
                    例如[[2，1，3，2，1，0，0],[2,3,3,1,0,0,0]]或[[2,1,3,2,1].[2,3,3,1]]
        :param seq_lens: 每个序列的长度
        :return:
        '''
        # get_micro_F1_score.__name__='micro_F1_score'
        span2tp,span2fp,span2fn=self._count(true_labels,pred_labels,seq_lens)
        _,_,r=self.get_F_P_R(sum(span2tp.values()),sum(span2fp.values()),sum(span2fn.values()),alpha=1.)
        return r

    # def get_micro_F_P_R(self,true_labels,pred_labels,seq_lens,alpha=1.):
    #     '''
    #     计算获取micro_F P R value
    #     :param true_labels: 真实label列表，二维列表或numpy，
    #                 例如[[1，2，3，2，1，0，0],[2,1,3,1,0,0,0]]或[[1,2,3,2,1].[2,1,3,1]]
    #     :param pred_labels: 预测标记列表，二维列表或numpy，
    #                 例如[[2，1，3，2，1，0，0],[2,3,3,1,0,0,0]]或[[2,1,3,2,1].[2,3,3,1]]
    #     :param seq_lens: 每个序列的长度
    #     :return:
    #     '''
    #     # get_micro_F1_score.__name__='micro_F1_score'
    #     span2tp,span2fp,span2fn=self._count(true_labels,pred_labels,seq_lens)
    #     f,p,r=self.get_F_P_R(sum(span2tp.values()),sum(span2fp.values()),sum(span2fn.values()),alpha=alpha)
    #     return f,p,r

    # def get_overall_accuracy(self,true_labels,pred_labels,seq_lens):
    #     '''
    #     计算获取overall_accuracy
    #     :param true_labels: 真实label列表，二维列表或numpy，
    #                 例如[[1，2，3，2，1，0，0],[2,1,3,1,0,0,0]]或[[1,2,3,2,1].[2,1,3,1]]
    #     :param pred_labels: 预测标记列表，二维列表或numpy，
    #                 例如[[2，1，3，2，1，0，0],[2,3,3,1,0,0,0]]或[[2,1,3,2,1].[2,3,3,1]]
    #     :param seq_lens: 每个序列的长度
    #     :return:
    #     '''
    #     # get_micro_F1_score.__name__ = 'micro_F1_score'
    #     span2tp, span2fp, span2fn = self._count(true_labels, pred_labels, seq_lens)
    #     acc=(sum(span2tp.values()))/(sum(span2tp.values())+sum(span2fp.values()))
    #     # f, _, _ = self.get_F_P_R(sum(span2tp.values()), sum(span2fp.values()), sum(span2fn.values()), alpha=1.)
    #     return acc

    def get_F_P_R(self, tp, fp, fn,alpha=1.):
        """
        根据变量计算F、presion、recall值
        :param tp: int, true positive
        :param fn: int, false negative
        :param fp: int, false positive
        :return: (f, precison, recall)
        """
        p = tp / (fp + tp + MIN_VAL)
        r = tp / (fn + tp + MIN_VAL)
        f = (1 + pow(alpha,2)) * p * r / (pow(alpha,2)* p + r + MIN_VAL)

        return f, p, r


def tag2span_bmes(tags, ignore_labels=None):
    """
    给定一个tags的lis，比如['S-song', 'B-singer', 'M-singer', 'E-singer', 'S-moive', 'S-actor']。
    返回[('song', (0, 1)), ('singer', (1, 4)), ('moive', (4, 5)), ('actor', (5, 6))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def tag2span_bmeso(tags, ignore_labels=None):
    """
    给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]

def tag2span_bieso(tags, ignore_labels=None):
    """
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bies_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bies_tag, label = tag[:1], tag[2:]
        if bies_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bies_tag in ('i', 'e') and prev_bies_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bies_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bies_tag = bies_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]

def tag2span_bio(tags, ignore_labels=None):
    """
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]



if __name__=='__main__':
    tags=['O','B-singer', 'B-singer','B-LOC','S-LOC', 'I-singer','I-songer','I-songer']
    print(tag2span_bieso(tags))
    # # print(get_confusion_matrix_df.metric)
    true_labels = ['A', 'B', 'C', 'C', 'A', 'A', 'B', 'B','A','C','D','C','D','D']
    pred_labels = ['A', 'B', 'C', 'C', 'B', 'A', 'C', 'B','A','B','D','D','D','A']
    unique_labels=set(true_labels)
    #
    # cm_df = get_confusion_matrix_df(true_labels, pred_labels,unique_labels)
    # print('cm_df:\n',cm_df)
    #
    accuracy=get_overall_accuracy(true_labels,pred_labels)
    print('accuracy:\n',accuracy)
    #
    # precision_series=get_precision_series(true_labels,pred_labels)
    # print('precision_series:\n',precision_series)
    # average_precision = get_average_precision(true_labels, pred_labels)
    # print('average_precision:\n', average_precision)
    #
    # recall_series=get_recall_series(true_labels,pred_labels)
    # print('recall_series:\n',recall_series)
    # average_recall = get_average_recall(true_labels, pred_labels)
    # print('average_recall:\n', average_recall)
    #
    # F_score_series = get_F_score_series(true_labels, pred_labels,alpha=0.5)
    # print('F_score_series:\n', F_score_series)
    # average_F_score = get_average_F_score(true_labels, pred_labels,is_weight=True,alpha=0.5)
    # print('average_F_score:\n', average_F_score)
    #
    # F1_score_series = get_F1_score_series(true_labels, pred_labels)
    # print('F1_score_series:\n',F1_score_series)
    # average_F1_score=get_average_F1_score(true_labels, pred_labels)
    # print('average_F1_score:\n', average_F1_score)
    #
    # # res_p_r_F1_a_df=get_eval(true_labels, pred_labels, is_weight=True)
    # # save_to_excel(res_p_r_F1_a_df,'result_test')
    #
    # sensitivity_series=get_sensitivity_series(true_labels, pred_labels)
    # print('sensitivity_series:\n',sensitivity_series)
    # average_sensitivity = get_average_sensitivity(true_labels, pred_labels)
    # print('average_sensitivity:\n', average_sensitivity)
    #
    # specificity_series=get_specificity_series(true_labels, pred_labels)
    # print('specificity_series:\n',specificity_series)
    # average_specificity = get_average_specificity(true_labels, pred_labels)
    # print('average_specificity:\n', average_specificity)
    #
    # balanced_accuracy_series=get_balanced_accuracy_series(true_labels, pred_labels)
    # print('balanced_accuracy_series:\n',balanced_accuracy_series)
    # average_balanced_accuracy = get_average_balanced_accuracy(true_labels, pred_labels)
    # print('average_balanced_accuracy:\n', average_balanced_accuracy)
    #
    # # get_sensitivity_series.__name__='1'
    # # print(get_sensitivity_series.__name__)

    average_F1_score=get_average_F1_score(true_labels, pred_labels)
    print(get_average_F1_score.__name__, ':', average_F1_score)
    macro_F1_score = get_macro_F1_score(true_labels, pred_labels)
    print(get_macro_F1_score.__name__,':',macro_F1_score)
    micro_F1_score = get_micro_F1_score(true_labels, pred_labels)
    print(get_micro_F1_score.__name__, ':', micro_F1_score)


    true_values = [1.3, 2.3, 0.3, 3.4, 2.1, 2.2]
    pred_values = [1.3, 2.3, 0.3, 3.4, 2.1, 2.0]
    corr_value=get_corr_value(true_values,pred_values,method='pearson')
    print('pearson correlation value:\n',corr_value)
    corr_value = get_corr_value(true_values,pred_values, method='spearman')
    print('pearson correlation value:\n', corr_value)
    corr_value = get_corr_value(true_values,pred_values, method='kendall')
    print('pearson correlation value:\n', corr_value)