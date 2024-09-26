# coding=utf-8
from .base_model import BaseModel
# sys.path.append(os.path.abspath('lib/util'))
from util.eval.translate_metric import get_corp_bleu1, get_corp_bleu2, get_corp_bleu3, get_corp_bleu4, \
    get_corp_bleu, get_meteor, get_rouge, get_cider
import torch
import os
import logging
# from nltk.translate import meteor_score
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TransSeq2Seq(BaseModel):
    def _get_log_fit_eval(self, loss, pred_tgt, gold_tgt, tgt_i2w):

        pred_tgt = torch.argmax(pred_tgt, dim=1)
        pred_tgt = pred_tgt.to('cpu').data.numpy()
        gold_tgt = gold_tgt.to('cpu').data.numpy()

        pred_tgts = self._tgt_ids2tokens(pred_tgt, tgt_i2w, self.tgt_end_idx)
        gold_tgts = self._tgt_ids2tokens(gold_tgt, tgt_i2w, self.tgt_end_idx)
        if not isinstance(gold_tgts[0][0], list):  # 必不可少
            gold_tgts = [[seq] for seq in gold_tgts]

        eval_np = np.array([metric(pred_tgts, gold_tgts) for metric in self.train_metrics])
        log_info = 'train loss:{0:.5f}'.format(loss.item())
        for eval_val, metric in zip(eval_np, self.train_metrics):
            log_info += ',{0}:{1:.5f}'.format(metric.__name__, eval_val)
        return log_info

    def _tgt_ids2tokens(self, tgts, tgt_i2w, end_idx, **kwargs):
        raise NotImplementedError


    def _do_validation(self,
                       valid_srcs=None,
                       valid_tgts=None,
                       tgt_i2w=None,
                       last=False,
                       best_valid_eval=-1,
                       increase_better=True):
        best_net_path = os.path.join(self.model_dir, '{}_best_net.net'.format(self.model_name))
        if not last and valid_srcs is not None and valid_tgts is not None:  # 如果有验证集
            if 'best_net' not in self.__dict__:
                self.best_net = None
                self.worse_epochs = 0
                if increase_better:
                    self.valid_eval = -np.inf
                else:
                    self.valid_eval = np.inf
            if best_valid_eval != -1:
                self.valid_eval = best_valid_eval
            pred_tgts = self.predict(valid_srcs, tgt_i2w)
            for i, pred_tgt in enumerate(pred_tgts):
                if len(pred_tgt) == 0:
                    pred_tgts[i] = ['.']
                assert len(pred_tgts[i]) > 0
            gold_tgts = self._tgt_ids2tokens(valid_tgts, tgt_i2w, self.pad_idx)
            if not isinstance(gold_tgts[0][0], list):
                gold_tgts = [[seq] for seq in gold_tgts]
            log_info = 'Comparison of previous and current '

            valid_eval = self.valid_metric(pred_tgts, gold_tgts)

            log_info += '{}: ({},{}) # '.format(self.valid_metric.__name__, self.valid_eval, valid_eval)

            logging.info(log_info)
            is_better = False
            if increase_better and valid_eval >= self.valid_eval:
                is_better = True
            elif not increase_better and valid_eval <= self.valid_eval:
                is_better = True

            if is_better:
                self.worse_epochs = 0
                self.valid_eval = valid_eval
                torch.save(self.net.state_dict(), best_net_path)
            else:
                self.worse_epochs += 1
            return self.worse_epochs, self.valid_eval
        elif last:
            self.net.load_state_dict(torch.load(best_net_path))
            self.net.train()

    def eval(self,
             test_srcs,
             test_tgts,
             tgt_i2w,
             ):
        pred_tgts = self.predict(test_srcs, tgt_i2w)
        for i, pred_tgt in enumerate(pred_tgts):
            if len(pred_tgt) == 0:
                pred_tgts[i] = ['.']
            assert len(pred_tgts[i]) > 0
        gold_tgts = self._tgt_ids2tokens(test_tgts, tgt_i2w, self.pad_idx)
        if not isinstance(gold_tgts[0][0], list):
            gold_tgts = [[seq] for seq in gold_tgts]

        eval_dic = dict()
        logging.info('Evaluate %s' % self.model_name)
        if not self.test_metrics:
            for metric in [get_meteor, get_rouge, get_corp_bleu, get_corp_bleu1, get_corp_bleu2, get_corp_bleu3,
                           get_corp_bleu4, get_cider]:
                eval_dic[metric.__name__] = {'OVERALL': metric(pred_tgts, gold_tgts)}
        else:
            for metric in self.test_metrics:
                eval_res = metric(pred_tgts, gold_tgts)
                eval_dic[metric.__name__] = dict()
                if isinstance(eval_res, float) or isinstance(eval_res, int):
                    eval_dic[metric.__name__]['OVERALL'] = eval_res
                elif isinstance(eval_res, pd.Series):
                    eval_dic[metric.__name__] = dict(eval_res)
        eval_df = pd.DataFrame(eval_dic)
        return eval_df