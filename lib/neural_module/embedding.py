import os
import logging
import codecs
import numpy as np
import pickle
import json
import torch
import torch.nn as nn

def parse_glove(glove_path,emb_dims=300):
    logging.info('########### Start parsing glove ##########')
    words=['<PAD>','<UNK>']
    embs=[[0]*emb_dims,[1]*emb_dims]
    with codecs.open(glove_path,'r') as f:
        for line in f:
            elements=line.rstrip().split(' ')
            words.append(elements[0])
            embs.append(elements[1:])
    emb_arr2=np.asarray(embs,dtype=np.float32)
    emb_arr2[1,:]=np.random.normal(emb_arr2[2,:].mean(axis=0),emb_arr2.std(axis=0),
                                     size=(emb_dims,))
    glove_dir=os.path.dirname(glove_path)
    np.save(os.path.join(glove_dir,'embed_weight.npy'),emb_arr2)
    word2idx, idx2word = {}, {}
    logging.info('Make the dictionary')
    for idx, word in enumerate(words):
        word2idx[word] = idx
        idx2word[idx] = word
    w2i2w = {'word2idx': word2idx, 'idx2word': idx2word}

    w2i2w_path=os.path.join(glove_dir,'w2i2w.pkl')
    logging.info('Save the dictionary into %s' % w2i2w_path)
    with codecs.open(w2i2w_path, 'wb') as f:
        pickle.dump(w2i2w, f)
    w2i2w_json_path = os.path.splitext(w2i2w_path)[0] + '.json'
    logging.info('Save the dictionary into %s' % w2i2w_json_path)
    with codecs.open(w2i2w_json_path, 'w', encoding='utf-8') as f:
        json.dump(w2i2w, f, indent=4)


class PosEnc(nn.Module):
    def __init__(self,max_len, emb_dims,train=True,pad=True,pad_idx=0):
        super().__init__()
        self.pad=pad
        self.pad_idx=pad_idx
        if not train:
            position_code = np.array([
                [pos / np.power(10000, 2.0 * (j // 2) / emb_dims) for j in range(emb_dims)]
                for pos in range(max_len)])
            position_code[:, 0::2] = np.sin(position_code[:, 0::2])
            position_code[:, 1::2] = np.cos(position_code[:, 1::2])
            position_code=torch.tensor(position_code).float()

            if pad:
                pad_row = torch.zeros(1, emb_dims)
                position_code = torch.cat((pad_row, position_code),dim=0)
                self.position_encoder = nn.Embedding(max_len + 1, emb_dims,padding_idx=self.pad_idx)
                self.position_encoder.weight = nn.Parameter(position_code,requires_grad=False)
            else:
                self.position_encoder = nn.Embedding(max_len, emb_dims, padding_idx=None)
                self.position_encoder.weight = nn.Parameter(position_code,requires_grad=False)
        else:
            if pad:
                self.position_encoder=nn.Embedding(max_len+1,emb_dims,padding_idx=self.pad_idx)
                nn.init.xavier_uniform_(self.position_encoder.weight[1:,:])
            else:
                self.position_encoder = nn.Embedding(max_len, emb_dims, padding_idx=None)
                nn.init.xavier_uniform_(self.position_encoder.weight)

    def forward(self, x):
        tensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        if self.pad:
            if len(x.size())==3:
                x_lens = x.abs().sum(2).sign().sum(1).to('cpu').int().data.numpy()
            elif len(x.size())==2: 
                x_lens=x.abs().sign().sum(1).int().cpu().data.numpy()
            x_pos = tensor([list(range(1, x_len + 1)) + [0] * (x.size(1) - x_len) for x_len in x_lens])
        else:
            x_pos=tensor(range(x.size(1))).unsqueeze(0).expand(x.size(0),-1)
        return self.position_encoder(x_pos)    #(B,L,D)

class LayerEnc(nn.Module):

    def __init__(self,layer_num, emb_dims,train=False):
        super().__init__()
        self.emb_dims=emb_dims
        if not train:
            layer_code = np.array([
                [pos / np.power(10000, 2.0 * (j // 2) / emb_dims) for j in range(emb_dims)]
                for pos in range(layer_num)])
            layer_code[:, 0::2] = np.sin(layer_code[:, 0::2])
            layer_code[:, 1::2] = np.cos(layer_code[:, 1::2])
            layer_code=torch.tensor(layer_code).float()

            self.layer_encoder = nn.Embedding(layer_num, emb_dims,padding_idx=None)
            self.layer_encoder.weight = nn.Parameter(layer_code,requires_grad=False)
        else:
            self.layer_encoder = nn.Embedding(layer_num, emb_dims, padding_idx=None)
            nn.init.xavier_uniform_(self.layer_encoder.weight)
        
    def forward(self,x,i):
        layer_code=torch.zeros(x.size(0),x.size(1),self.emb_dims,device=x.device)
        i_code=self.layer_encoder(torch.tensor([i],device=x.device))
        if len(x.size())==3:
            x_lens = x.sum(2).abs().sign().sum(1).to('cpu').data.numpy().astype(np.int)
        elif len(x.size())==2:
            x_lens=x.abs().sign().sum(1).to('cpu').data.numpy().astype(np.int)
        for j,x_len in enumerate(x_lens):
            layer_code[j,:x_len,:]=i_code.expand(x_len,-1)
        return layer_code
