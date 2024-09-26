# coding=utf-8
'''
翻译任务的评价
'''
import numpy as np
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .google_bleu import corpus_bleu
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .nltk_bleu_score import sentence_bleu, SmoothingFunction

# from torchtext.data.metrics import bleu_score

MIN_VAL = 10e-13
np.set_printoptions(2)


def get_nltk33_sent_bleu1(preds, refs, auto_reweigh=True):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-1'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    cc = SmoothingFunction()
    scores = [sentence_bleu(ref, pred + [' '] * max(0, 2 - len(pred)), weights=(1., 0., 0., 0.),
                            smoothing_function=cc.method4, auto_reweigh=auto_reweigh)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_nltk33_sent_bleu2(preds, refs, auto_reweigh=True):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-2'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    cc = SmoothingFunction()
    scores = [sentence_bleu(ref, pred + [' '] * max(0, 2 - len(pred)), weights=(0., 1., 0., 0.),
                            smoothing_function=cc.method4, auto_reweigh=auto_reweigh)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_nltk33_sent_bleu3(preds, refs, auto_reweigh=True):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-3'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    cc = SmoothingFunction()
    scores = [sentence_bleu(ref, pred + [' '] * max(0, 2 - len(pred)), weights=(0., 0., 1., 0.),
                            smoothing_function=cc.method4, auto_reweigh=auto_reweigh)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_nltk33_sent_bleu4(preds, refs, auto_reweigh=True):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-4'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    cc = SmoothingFunction()
    scores = [sentence_bleu(ref, pred + [' '] * max(0, 2 - len(pred)), weights=(0., 0., 0., 1.),
                            smoothing_function=cc.method4, auto_reweigh=auto_reweigh)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_nltk33_sent_bleu(preds, refs, auto_reweigh=True):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    cc = SmoothingFunction()
    scores = [sentence_bleu(ref, pred + [' '] * max(0, 2 - len(pred)), weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=cc.method4, auto_reweigh=auto_reweigh)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_google_sent_bleu1(preds, refs):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-1'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    scores = [corpus_bleu([pred], [ref], max_n=1, weights=[1.], smooth=True)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_google_sent_bleu2(preds, refs):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-2'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    scores = [corpus_bleu([pred], [ref], max_n=2, weights=[0., 1.], smooth=True)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_google_sent_bleu3(preds, refs):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-3'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    scores = [corpus_bleu([pred], [ref], max_n=3, weights=[0., 0., 1.], smooth=True)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_google_sent_bleu4(preds, refs):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu-4'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    scores = [corpus_bleu([pred], [ref], max_n=4, weights=[0., 0., 0., 1.], smooth=True)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_google_sent_bleu(preds, refs):
    '''计算平均bleu，每条数据计算bleu后平均,sentence-level-bleu'''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    scores = [corpus_bleu([pred], [ref], max_n=4, weights=[0.25, 0.25, 0.25, 0.25], smooth=True)
              for pred, ref in zip(preds, refs)]
    return sum(scores) / len(scores) * 100


def get_corp_bleu1(preds, refs):
    '''
    计算corp层次的bleu,corpus-level-bleu-1
    '''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    preds = [pred + [' '] * max(0, 1 - len(pred)) for pred in preds]
    return corpus_bleu(preds, refs, max_n=1, weights=[1.], smooth=True) * 100


def get_corp_bleu2(preds, refs):
    '''
        计算corp层次的bleu,corpus-level-bleu-2
        '''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    preds = [pred + [' '] * max(0, 1 - len(pred)) for pred in preds]
    return corpus_bleu(preds, refs, max_n=2, weights=[0., 1.], smooth=True) * 100


def get_corp_bleu3(preds, refs):
    '''
        计算corp层次的bleu,corpus-level-bleu-3
        '''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    preds = [pred + [' '] * max(0, 1 - len(pred)) for pred in preds]
    return corpus_bleu(preds, refs, max_n=3, weights=[0., 0., 1.], smooth=True) * 100


def get_corp_bleu4(preds, refs):
    '''
        计算corp层次的bleu,corpus-level-bleu-4
        '''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    preds = [pred + [' '] * max(0, 1 - len(pred)) for pred in preds]
    return corpus_bleu(preds, refs, max_n=4, weights=[0., 0., 0., 1.], smooth=True) * 100


def get_corp_bleu(preds, refs):
    '''
        计算corp层次的bleu,corpus-level-bleu
        '''
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    preds = [pred + [' '] * max(0, 1 - len(pred)) for pred in preds]
    return corpus_bleu(preds, refs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25], smooth=True) * 100


def get_meteor(preds, refs):
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    return Meteor().compute_score(preds, refs)[0] * 100


def get_rouge(preds, refs):
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    return Rouge().compute_score(preds, refs)[0] * 100


def get_cider(preds, refs):
    if not isinstance(refs[0][0], list):  # 必不可少
        refs = [[seq] for seq in refs]
    return Cider().compute_score(preds, refs)[0]


#
# candidate_corp = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
# references_corp = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
# print(get_corp_bleu(candidate_corp,references_corp))
# print(get_sent_bleu(candidate_corp,references_corp))

# 函数的__name__属性在函数运行前不会改变，要改变只有两种方式，要么函数运行后，要么定义在函数后面
get_corp_bleu1.__name__ = 'C-BLEU-1(%)'
get_corp_bleu2.__name__ = 'C-BLEU-2(%)'
get_corp_bleu3.__name__ = 'C-BLEU-3(%)'
get_corp_bleu4.__name__ = 'C-BLEU-4(%)'
get_corp_bleu.__name__ = 'C-BLEU(%)'
get_nltk33_sent_bleu1.__name__ = 'S-BLEU-1(%)'
get_nltk33_sent_bleu2.__name__ = 'S-BLEU-2(%)'
get_nltk33_sent_bleu3.__name__ = 'S-BLEU-3(%)'
get_nltk33_sent_bleu4.__name__ = 'S-BLEU-4(%)'
get_nltk33_sent_bleu.__name__ = 'S-BLEU(%)'
get_google_sent_bleu1.__name__ = 'S-BLEU-1(%)'
get_google_sent_bleu2.__name__ = 'S-BLEU-2(%)'
get_google_sent_bleu3.__name__ = 'S-BLEU-3(%)'
get_google_sent_bleu4.__name__ = 'S-BLEU-4(%)'
get_google_sent_bleu.__name__ = 'S-BLEU(%)'
get_meteor.__name__ = 'METEOR(%)'
get_rouge.__name__ = 'ROUGE(%)'
get_cider.__name__ = 'CIDER'

if __name__ == '__main__':
    candidate_corpus = [['yes']]
    references_corpus = [[['yes']]]
    print(get_google_sent_bleu(candidate_corpus, references_corpus))
