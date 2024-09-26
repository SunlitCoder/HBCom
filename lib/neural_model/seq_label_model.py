from .base_model import BaseModel,BaseNet
from ..neural_module.learn_strategy import LrWarmUp
from ..neural_module.transformer import TranEnc
from ..neural_module.embedding import PosEnc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import logging
import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Datasetx(Dataset):
    def __init__(self,ins,outs=None,in_max_len=None):
        self.len=len(ins)
        self.in_max_len=in_max_len
        if in_max_len is None:
            self.in_max_len = max([len(seq) for seq in ins])
        self.ins=ins
        self.outs=outs
    def __getitem__(self, index):
        tru_feature=self.ins[index][:self.in_max_len]
        pad_feature = np.lib.pad(tru_feature, (0, self.in_max_len - len(tru_feature)),
                                        'constant', constant_values=(0, 0))  # padding
        if self.outs is None:
            return torch.tensor(pad_feature)
        else:
            tru_out=self.outs[index][:self.in_max_len]
            pad_out=np.lib.pad(tru_out, (0, self.in_max_len - len(tru_out)),
                                        'constant', constant_values=(0, 0))  # padding
            return torch.tensor(pad_feature),\
                   torch.tensor(pad_out).long()

    def __len__(self):
        return self.len

class TransNet(BaseNet):
    def __init__(self,
                 in_max_len,
                 vocab_size,
                 out_dims,
                 embed_dims=300,
                 token_init_embed=None,
                 token_embed_freeze=False,
                 att_layer_num=6,
                 head_num=10,
                 head_dims=None,
                 drop_rate=0.
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        #获取Net的init参数
        self.init_params=locals()
        del self.init_params['self']
        self.position_encoding = PosEnc(max_len=in_max_len,embed_dims=embed_dims,train=True)
        if token_init_embed is None:
            self.token_embedding=nn.Embedding(vocab_size,embed_dims,padding_idx=0)
            nn.init.xavier_uniform_(self.token_embedding.weight[1:,:])  #nn.init.xavier_uniform_
        else:
            token_init_embed=torch.tensor(token_init_embed,dtype=torch.float32)
            self.token_embedding=nn.Embedding.from_pretrained(token_init_embed,freeze=token_embed_freeze,padding_idx=0)

        self.encoder = TranEnc(query_dims=embed_dims,
                               head_num=head_num,
                               head_dims=head_dims,
                               layer_num=att_layer_num,
                               drop_rate=drop_rate)

        self.layer_norm = nn.LayerNorm(embed_dims)
        self.out_fc = nn.Sequential(
            nn.Linear(embed_dims, 128),  # 4
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, out_dims),
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        token_embed=self.token_embedding(x)*np.sqrt(self.embed_dims)
        pos_embed=self.position_encoding(x)
        token_mask = x.abs().sign()
        token_coder = self.layer_norm(token_embed.add(pos_embed))
        token_coder=self.dropout(token_coder)
        token_coder=self.encoder(token_coder,token_mask)
        outputs=self.out_fc(token_coder)

        return outputs.transpose(1, 2)


class TransSeqLabel(BaseModel):
    def __init__(self,
                 model_dir,
                 model_name='Transformer_based_model',
                 model_id=None,
                 embed_dims=512,
                 token_embed_path=None,
                 token_embed_freeze=True,
                 head_num=8,
                 head_dims=None,
                 att_layer_num=6,
                 drop_rate=0.3,
                 batch_size=32,
                 big_epochs=20,
                 regular_rate=1e-5,
                 lr_base=0.001,
                 lr_decay=0.9,
                 min_lr_rate=0.01,
                 warm_big_epochs=2,
                 Net=TransNet,
                 Dataset=Datasetx,
                 ):
        logging.info('Construct %s'%model_name)
        self.init_params = locals()
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         model_id=model_id)

        self.embed_dims = embed_dims
        self.token_embed_path = token_embed_path
        self.token_embed_freeze = token_embed_freeze
        self.head_num = head_num
        self.head_dims=head_dims
        self.att_layer_num = att_layer_num
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.big_epochs = big_epochs
        self.regular_rate=regular_rate
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.min_lr_rate=min_lr_rate
        self.warm_big_epochs=warm_big_epochs
        self.Net = Net
        self.Dataset = Dataset

    def fit(self,
            train_features,
            train_outs,
            out2tag=None,
            tag2span_func=None,
            valid_features=None,
            valid_outs=None,
            train_metrics=[get_overall_accuracy],
            valid_metric=get_overall_accuracy,
            verbose=0
            ):
        logging.info('Train %s' % self.model_name)
        self.out2tag=out2tag
        self.tag2span_func=tag2span_func
        self.train_metrics = train_metrics
        self.valid_metric = valid_metric
        self.in_max_len = max(len(seq) for seq in train_features)
        self.vocab_size = max(np.max(seq) for seq in train_features) + 1
        self.sort_unique_outs = sorted(list(np.unique(np.concatenate(train_outs))))
        self.out_dims=len(self.sort_unique_outs)+1

        token_embed_weight = None
        if self.token_embed_path is not None:  # 如果加载预训练词向量
            token_embed_weight = np.load(self.token_embed_path)
            self.vocab_size = token_embed_weight.shape[0]
        net = self.Net(in_max_len=self.in_max_len,
                       vocab_size=self.vocab_size,
                       embed_dims=self.embed_dims,
                       token_init_embed=token_embed_weight,
                       token_embed_freeze=self.token_embed_freeze,
                       att_layer_num=self.att_layer_num,
                       head_num=self.head_num,
                       head_dims=self.head_dims,
                       out_dims=self.out_dims,
                       drop_rate=self.drop_rate,
                       )

        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #选择GPU优先
        self.net = nn.DataParallel(net.to(device))  # 并行使用多GPU
        self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(),
                               lr=self.lr_base,
                               weight_decay=self.regular_rate)
        if self.token_embed_path is not None:  # 如果加载预训练词向量
            token_embed_param = [x for x in self.net.parameters() if x.requires_grad and x.size(0) == self.vocab_size]
            ex_param = [x for x in self.net.parameters() if x.requires_grad and x.size(0) != self.vocab_size]
            optim_cfg = [{'params': token_embed_param, 'lr': self.lr_base*0.1},
                         {'params': ex_param, 'lr': self.lr_base, 'weight_decay': self.regular_rate}, ]
            self.optimizer = optim.Adam(optim_cfg)
        if self.warm_big_epochs is None:
            self.warm_big_epochs= max(self.big_epochs // 10, 2)
        self.scheduler = LrWarmUp(self.optimizer,
                             min_rate=self.min_lr_rate,
                             lr_decay=self.lr_decay,
                             warm_steps=self.warm_big_epochs * len(train_loader),
                             reduce_steps=len(train_loader))

        if self.out2tag is None:
            self.seq_mode='POS'
        else:
            self.seq_mode='NER'
        for i in range(self.big_epochs):
            for j, (batch_features, batch_outs) in enumerate(train_loader):
                batch_features=batch_features.to(device)
                batch_outs=batch_outs.to(device)
                pred_outs=self.net(batch_features)
                loss=self.criterion(pred_outs,batch_outs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.scheduler.step()
                self._log_fit_eval(loss=loss,
                                   big_step=i+1,
                                   batch_step=j+1,
                                   big_epochs=self.big_epochs,
                                   batch_epochs=len(train_loader),
                                   pred_outs=pred_outs,
                                   true_outs=batch_outs,
                                   seq_mode=self.seq_mode)

            self._do_validation(valid_features=valid_features,
                                valid_outs=valid_outs,
                                increase_better=True,
                                seq_mode=self.seq_mode,
                                last=False)

        self._do_validation(valid_features=valid_features,
                            valid_outs=valid_outs,
                            increase_better=True,
                            seq_mode=self.seq_mode,
                            last=True)

    def pred_out_tags(self,
                        ins,
                        tag_i2w,
                        pred_out_tag_path=None,
                        true_outs=None,
                        ):
        logging.info('---Predict the real tags of the sequences')
        if pred_out_tag_path is not None:
            pred_out_tag_dir=os.path.dirname(pred_out_tag_path)
            if not os.path.exists(pred_out_tag_dir):
                os.makedirs(pred_out_tag_dir)
        pred_outs,_=self.predict(ins)
        pred_out_tags=[[tag_i2w[out_idx] for out_idx in pred_out_seq[:list(pred_out_seq).index(0)]]
                         for pred_out_seq in pred_outs]
        if pred_out_tag_path is not None:
            feature_content=[' '.join(feature_seq) for feature_seq in ins]
            pred_out_tag_content = [' '.join(out_tag_seq) for out_tag_seq in pred_out_tags]
            if true_outs is not None:
                true_out_tag_content = [' '.join([tag_i2w[out_idx] for out_idx in true_out_seq])
                                   for true_out_seq in true_outs]
                content='\n\n'.join(['\n'.join(content_tuple) for content_tuple in zip(feature_content,true_out_tag_content,pred_out_tag_content)])
            else:
                content='\n\n'.join(['\n'.join(content_tuple) for content_tuple in zip(feature_content,pred_out_tag_content)])
            with codecs.open(pred_out_tag_path,'w') as f:
                f.write(content)
        return pred_out_tags