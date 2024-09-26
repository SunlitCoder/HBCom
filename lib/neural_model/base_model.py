from util.eval.classify_metric import *
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import copy
import pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Datasetx(Dataset):
    pass

class BaseNet(nn.Module):
    def _get_init_params(self):
        if 'self' in self.init_params:
            del self.init_params['self']
        if '__class__' in self.init_params:
            del self.init_params['__class__']
        return self.init_params

class BaseModel(object):
    def __init__(self,
                 model_dir,
                 model_name='model',
                 model_id=None):
        self.model_dir=model_dir
        if self.model_dir and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name=model_name
        if model_id is not None:
            self.model_name='{}_{}'.format(model_name,model_id)
        self.model_path = os.path.join(model_dir, '%s' % self.model_name)  # 模型路径

        # 配置日志信息
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def _get_init_params(self):
        # self.init_params=locals()
        if 'self' in self.init_params:
            del self.init_params['self']
        if '__class__' in self.init_params:
            del self.init_params['__class__']
        return self.init_params


    def save_params(self,param_path=None):
        logging.info('Save extern parameters of %s' % self.model_name)
        if param_path is None:
            param_path = os.path.join(self.model_dir, self.model_name + '.param.pkl')
        param_dir=os.path.dirname(param_path)
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        param_dic = self._get_init_params()
        if 'tgt_begin_idx' in self.__dict__:  # 如果是seq2seq问题:
            param_dic.update({'tgt_begin_idx': self.tgt_begin_idx,
                              'tgt_end_idx': self.tgt_end_idx,
                               'src_max_len': self.src_max_len,
                               'tgt_max_len': self.tgt_max_len,
                              'src_voc_size':self.src_voc_size,
                              'tgt_voc_size':self.tgt_voc_size
                              })
        else:
            param_dic.update({'in_max_len': self.in_max_len,
                               'out_dim': self.out_dim})
            if 'sort_unique_outs' in self.__dict__:
                param_dic.update({'sort_unique_outs':self.sort_unique_outs})
            if 'unique_outs' in self.__dict__:
                param_dic.update({'unique_outs': self.unique_outs})

        with open(param_path, 'wb') as f:
            pickle.dump(param_dic, f)

    def load_params(self,param_path=None):
        logging.info('Load extern parameters of %s' % self.model_name)
        if param_path is None:
            param_path = os.path.join(self.model_dir, self.model_name + '.param.pkl')
        with open(param_path, 'rb') as f:
            param_dic=pickle.load(f)
        self.__dict__.update(param_dic)

    def save_net(self,net_path=None):
        if net_path is None:
            net_path = os.path.join(self.model_dir, self.model_name + '.net')
        net_dir=os.path.dirname(net_path)
        if net_dir and not os.path.exists(net_dir):
            os.makedirs(net_dir)
        net_state = {'net': self.net.state_dict(),
                     'net_params':self.net.module._get_init_params()}
        torch.save(net_state,net_path)


    def load_net(self,net_path=None):
        if net_path is None:
            net_path = os.path.join(self.model_dir, self.model_name + '.net')
        checkpoint= torch.load(net_path)
        net_params=checkpoint['net_params']
        self.net=self.Net(**net_params)
        self.net = nn.DataParallel(self.net)
        self.net.load_state_dict(checkpoint['net'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        self.net.train()


    def fit(self,**kwargs):
        raise NotImplementedError

    def _tgt_ids2tokens(self,tgts, tgt_i2w, end_idx,**kwargs):
        raise NotImplementedError

    def _get_log_fit_eval(self,
                      loss,
                      big_step, batch_step, batch_epochs,
                      pred_outs, true_outs,
                      seq_mode=None):
        if len(pred_outs.size()) == 2 and self.out_dims > 1:
            pred_outs = torch.argmax(pred_outs, dim=1)
        elif len(pred_outs.size()) == 3 and self.out_dims > 1:
            pred_outs = torch.argmax(pred_outs[:, :, :], dim=1)
        if len(pred_outs.size()) == 2:
            out_lens = true_outs.sign().sum(1)
        true_out_np = true_outs.to('cpu').data.numpy()
        del true_outs
        pred_out_np = pred_outs.to('cpu').data.numpy()
        del pred_outs
        if len(true_out_np.shape) == 1 and self.out_dims > 1:
            eval_np=np.array([metric(true_out_np, pred_out_np, unique_outs=self.sort_unique_outs)
                              for metric in self.train_metrics])
        elif len(true_out_np.shape) == 2 and self.out_dims > 1:
            if seq_mode is None or seq_mode == 'POS':
                true_out_np = np.concatenate(
                    [true_out_np[i, :out_lens[i]] for i in range(out_lens.size(0))])  # (BL-,)
                pred_out_np = np.concatenate(
                    [pred_out_np[i, :out_lens[i]] for i in range(out_lens.size(0))])  # (BL-,)
                eval_np=np.array([metric(true_out_np, pred_out_np, unique_outs=self.sort_unique_outs)
                                  for metric in self.train_metrics])
            elif seq_mode == 'NER':
                eval_np=np.array([metric(true_out_np, pred_out_np,
                                    seq_lens=out_lens,
                                    out2tag=self.out2tag,
                                    tag2span_func=self.tag2span_func)
                                  for metric in self.train_metrics])
            elif seq_mode == 'WHOLE':
                true_out_list = [' '.join([str(idx) for idx in true_out_np[i, :out_lens[i]]]) for i in range(out_lens.size(0))]  # (BL-,)
                pred_out_list = [' '.join([str(idx) for idx in pred_out_np[i, :out_lens[i]]]) for i in range(out_lens.size(0))]  # (BL-,)
                eval_np = np.array([metric(true_out_list, pred_out_list) for metric in self.train_metrics])
            elif seq_mode == 'BLEU':
                raise NotImplementedError
        elif self.out_dims == 1:  # 如果输出为值
            eval_np = np.array([metric(true_out_np, pred_out_np) for metric in self.train_metrics])

        log_info = 'train loss:{0:.5f}'.format(loss.item())
        for eval_val, metric in zip(eval_np, self.train_metrics):
            log_info += ',{0}:{1:.5f}'.format(metric.__name__, eval_val)
        return log_info

    def _do_validation(self, 
                       valid_ins=None, 
                       valid_outs=None, 
                       last=False,
                       increase_better=True, 
                       seq_mode=None):

        if not last and valid_ins is not None and valid_outs is not None:
            if 'best_net' not in self.__dict__:
                self.best_net='Sure thing'
                self.valid_loss_val = 1000
                if increase_better:
                    self.valid_evals = [-1000] * (len(self.train_metrics) + 1)
                else:
                    self.valid_evals = [1000] * (len(self.train_metrics) + 1)
            #首先计算loss
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pred_out_np,pred_out_prob_np=self.predict(valid_ins)
            pred_out_probs=torch.tensor(pred_out_prob_np)
            if len(pred_out_probs.size())==3:
                true_outs=[np.lib.pad(seq, (0, pred_out_probs.size(2) - len(seq)),
                                        'constant', constant_values=(0, 0)) for seq in valid_outs]
                out_lens=[np.sign(seq).sum() for seq in valid_outs]
            true_outs=torch.tensor(true_outs)
            valid_loss_val=0.
            for batch_num,i in enumerate(range(0,true_outs.size(0),self.batch_size)):
                batch_pred_out_probs=pred_out_probs[i:i+self.batch_size].to(device)
                batch_true_outs=true_outs[i:i+self.batch_size].to(device)
                valid_loss_val+=self.criterion(batch_pred_out_probs, batch_true_outs).item()
            valid_loss_val/=(batch_num+1e-20)
            true_outs=true_outs.data.numpy()

            log_info = 'Comparison of previous and current valid loss: ({},{})'.format(self.valid_loss_val,
                                                                                       valid_loss_val)
            del pred_out_probs


            metrics = copy.deepcopy(self.train_metrics)
            if self.valid_metric is not None:
                metrics.append(self.valid_metric)
            if len(pred_out_np.shape) == 1 and self.out_dims > 1:
                valid_evals=[metric(true_outs, pred_out_np, unique_outs=self.sort_unique_outs) for metric in metrics]
            elif len(pred_out_np.shape) == 2 and self.out_dims > 1:
                if seq_mode is None or seq_mode == 'POS':
                    pred_outs = np.concatenate([pred_out_np[i,:out_len] for i,out_len in enumerate(out_lens)])  # (BL-,)
                    true_outs=np.concatenate([true_outs[i][:out_len] for i,out_len in enumerate(out_lens)])
                    valid_evals=[metric(pred_outs, true_outs, unique_outs=self.sort_unique_outs) for metric in metrics]
                elif seq_mode == 'NER':
                    valid_evals=[metric(true_outs, pred_out_np,
                                              seq_lens=out_lens,
                                              out2tag=self.out2tag,
                                              tag2span_func=self.tag2span_func)
                                 for metric in metrics]
                elif seq_mode == 'WHOLE':
                    true_out_list = [' '.join([str(idx) for idx in true_out]) for true_out in true_outs]  # (BL-,)
                    pred_out_list = [' '.join([str(idx) for idx in pred_out]) for pred_out in pred_out_np]  # (BL-,)
                    valid_evals=[metric(true_out_list, pred_out_list) for metric in metrics]
                elif seq_mode == 'BLEU':
                    raise NotImplementedError
            elif self.out_dims == 1:
                valid_evals=[metric(true_outs, pred_out_np) for metric in metrics]

            for i, metric in enumerate(metrics):
                log_info += ', average {}: ({},{})'.format(metric.__name__, self.valid_evals[i], valid_evals[i])

            logging.info(log_info)
            is_better = False
            if self.valid_metric is not None:  # 如果有valid metric
                if increase_better and valid_evals[-1] >= self.valid_evals[-1]:\
                    is_better = True
                elif not increase_better and valid_evals[-1] <= self.valid_evals[-1]:\
                    is_better = True
            elif valid_loss_val <= self.valid_loss_val:
                is_better = True
            if is_better:
                self.valid_loss_val = valid_loss_val
                self.valid_evals = valid_evals
                torch.save(self.net.state_dict(),os.path.join(self.model_dir,self.model_name+'_best_net.net'))
        elif last:
            self.net.load_state_dict(torch.load(os.path.join(self.model_dir,self.model_name+'_best_net.net')))
            self.net.train()

    def predict(self,ins):
        logging.info('Predict outputs of %s' % self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net=self.net.to(device)
        self.net.eval()
        if 'out_begin_idx' in self.__dict__:
            raise NotImplementedError

        else:
            dataset = self.Dataset(ins, in_max_len=self.in_max_len)
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                  num_workers=8)
            pred_out_prob_batches=[]
            pred_out_batches = []
            with torch.no_grad():
                for batch_features in data_loader:
                    batch_features=batch_features.to(device)
                    pred_out_probs = self.net(batch_features)
                    pred_out_prob_batches.append(pred_out_probs.to('cpu').data.numpy())
                    if len(pred_out_probs.size())==2 and self.out_dims>1:
                        pred_outs = torch.argmax(pred_out_probs, dim=1)
                    elif len(pred_out_probs.size())==3 and self.out_dims>1:
                        pred_outs=torch.argmax(pred_out_probs[:,1:,:],dim=1)+1
                        zero_mask = batch_features.eq(0)  # (B,L)
                        if len(batch_features.size()) == 3:
                            zero_mask=zero_mask[:,0,:]
                        pred_outs = pred_outs.masked_fill(zero_mask, 0)  # (B,L)
                    pred_out_batches.append(pred_outs.to('cpu').data.numpy())
            pred_out_prob_np =np.concatenate(pred_out_prob_batches, axis=0)
            pred_out_np=np.concatenate(pred_out_batches,axis=0)
        self.net.train()
        return pred_out_np,pred_out_prob_np


    def eval_class(self,
                   test_ins,
                   test_outs,
                   unique_outs=None,
                   focus_labels=[],
                   test_metrics=[get_sensitivity_series,
                                 get_specificity_series,
                                 get_balanced_accuracy_series,
                                 ],
                   percentage=False
                   ):
        logging.info('Evaluate %s' % self.model_name)
        pred_outs,_ = self.predict(test_ins)
        assert self.out_dims > 1 and len(pred_outs.shape) == 1

        if unique_outs is None and 'sort_unique_outs' in self.__dict__:
            unique_outs = self.sort_unique_outs
        elif unique_outs is None and 'unique_outs' in self.__dict__:
            unique_outs = sorted(self.unique_outs)
        if focus_labels == []:
            focus_labels = sorted(np.unique(test_outs))
        index = focus_labels
        columns = [metric.__name__ for metric in test_metrics]
        eval_df = pd.DataFrame(data=np.empty(shape=(len(index), len(columns))),
                               index=index,
                               columns=columns,
                               )
        for metric in test_metrics:
            eval_result = metric(test_outs, pred_outs, unique_outs=unique_outs)
            if isinstance(eval_result, float) or isinstance(eval_result, int):
                eval_df.loc[:, metric.__name__] = None
                eval_df.loc[:, metric.__name__].iloc[0] = eval_result
                tmp_series = eval_df.loc[:, metric.__name__]
                eval_df.drop(labels=[metric.__name__], axis=1, inplace=True)
                eval_df = pd.concat((eval_df, tmp_series), axis=1)
            elif isinstance(eval_result, pd.Series):
                eval_df.loc[focus_labels, metric.__name__] = eval_result[focus_labels].values

        return eval_df

    def eval_seq(self,
                 test_ins,
                 test_outs,
                 test_metrics=[get_span_micro_F1],
                 seq_mode=None
                 ):
        logging.info('Evaluate %s' % self.model_name)
        pred_outs,_ = self.predict(test_ins)
        out_lens = [np.sign(seq).sum() for seq in test_outs]
        assert self.out_dims > 1 and len(pred_outs.shape) == 2
        test_outs = [np.lib.pad(seq, (0, pred_outs.shape[1] - len(seq)),
                                'constant', constant_values=(0, 0)) for seq in test_outs]
        eval_dic = dict()
        for metric in test_metrics:
            if seq_mode is None or seq_mode == 'POS':
                # print(metric.__name__)
                pred_out_list = np.concatenate([pred_outs[i, :out_len] for i, out_len in enumerate(out_lens)])  # (BL-,)
                true_out_list = np.concatenate([test_outs[i][:out_len] for i, out_len in enumerate(out_lens)])
                eval_result = metric(pred_out_list, true_out_list, unique_outs=self.sort_unique_outs)
            elif seq_mode == 'NER':
                eval_result = metric(true_labels=test_outs,
                                     pred_labels=pred_outs,
                                     seq_lens=out_lens,
                                     out2tag=self.out2tag,
                                     tag2span_func=self.tag2span_func)
            elif seq_mode == 'WHOLE':
                true_out_list = [list(test_out) for test_out in test_outs]
                pred_out_list = [list(pred_out) for pred_out in pred_outs]
                eval_result = metric(true_out_list, pred_out_list)
            elif seq_mode == 'BLEU':
                true_out_list = [[[self.out_i2w[idx] for idx in (test_out[:test_out.tolist().index(0)]
                                    if 0 in test_out else test_out)]] for test_out in test_outs]  # (BL-,)
                pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
                                if 0 in pred_out else pred_out)] for pred_out in pred_outs]
                eval_result = metric(pred_out_list,true_out_list)
            eval_dic[metric.__name__] = dict()
            if isinstance(eval_result, float) or isinstance(eval_result, int):
                eval_dic[metric.__name__]['OVERALL'] = eval_result
            elif isinstance(eval_result, pd.Series):
                eval_dic[metric.__name__] = dict(eval_result)
        eval_df = pd.DataFrame(eval_dic)
        return eval_df

    def eval_reg(self,
                   test_ins,
                   test_outs,
                   test_metrics=[get_pearson_corr_val,
                                 get_spearman_corr_val,
                                 get_kendall_corr_val,
                                 ]
                   ):
        logging.info('Evaluate %s' % self.model_name)
        pred_outs,_  = self.predict(test_ins)  # 预测出的标记一维数组
        assert len(pred_outs.shape) == 1 and self.out_dims == 1  # 如果输出为值
        columns = [metric.__name__ for metric in test_metrics]  # 列名
        eval_df = pd.DataFrame(data=np.empty(shape=(1, len(columns))),
                               columns=columns,
                               )  # 评价结果
        for metric in test_metrics:
            eval_result = metric(list(test_outs), list(pred_outs))  # 计算评价结果
            eval_df.loc[0, metric.__name__] = eval_result
        return eval_df