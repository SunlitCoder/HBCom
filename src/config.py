import logging
import os

from lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
    get_nltk33_sent_bleu2 as get_sent_bleu2, \
    get_nltk33_sent_bleu3 as get_sent_bleu3, \
    get_nltk33_sent_bleu4 as get_sent_bleu4, \
    get_nltk33_sent_bleu as get_sent_bleu
from lib.util.eval.translate_metric import get_corp_bleu1, get_corp_bleu2, get_corp_bleu3, get_corp_bleu4, \
    get_corp_bleu
from lib.util.eval.translate_metric import get_meteor, get_rouge, get_cider
import math


train_data_name = 'train_data'
valid_data_name = 'valid_data'
test_data_name = 'test_data'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 顶级数据目录
top_data_dir = '../data'

raw_data_dir = os.path.join(top_data_dir, 'raw_data/')
train_raw_data_path = os.path.join(raw_data_dir, '{}.json'.format(train_data_name))
valid_raw_data_path = os.path.join(raw_data_dir, '{}.json'.format(valid_data_name))
test_raw_data_path = os.path.join(raw_data_dir, '{}.json'.format(test_data_name))
tech_term_path = os.path.join(raw_data_dir, 'tech_term.txt')
keep_test_data_id_path = os.path.join(raw_data_dir, 'keep_test_data_ids.txt')

max_code_len = 64
max_graph_size = 128
max_text_len = 38

token_data_dir = os.path.join(top_data_dir, 'token_data/')
train_token_data_path = os.path.join(token_data_dir, '{}.json'.format(train_data_name))
valid_token_data_path = os.path.join(token_data_dir, '{}.json'.format(valid_data_name))
test_token_data_path = os.path.join(token_data_dir, '{}.json'.format(test_data_name))

basic_info_dir = os.path.join(top_data_dir, 'basic_info/')
size_info_path = os.path.join(basic_info_dir, 'size_info.pkl')
rev_dic_path = os.path.join(basic_info_dir, 'rev_dic.json')
noise_token_path = os.path.join(basic_info_dir, 'noise_token.json')

w2i2w_dir = os.path.join(top_data_dir, 'w2i2w/')
io_token_w2i_path = os.path.join(w2i2w_dir, 'io_token_w2i.pkl')
io_token_i2w_path = os.path.join(w2i2w_dir, 'io_token_i2w.pkl')

io_min_token_count = 3
unk_aliased = True

avail_data_dir = os.path.join(top_data_dir, 'avail_data/')
train_avail_data_path = os.path.join(avail_data_dir, '{}.pkl'.format(train_data_name))
valid_avail_data_path = os.path.join(avail_data_dir, '{}.pkl'.format(valid_data_name))
test_avail_data_path = os.path.join(avail_data_dir, '{}.pkl'.format(test_data_name))

OUT_BEGIN_TOKEN = '</s>'
OUT_END_TOKEN = '</e>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
USER_WORDS = [('\\', 'n'), ('e', '.', 'g', '.'), ('i', '.', 'e', '.'), ('-', '>')]

model_dir = os.path.join(top_data_dir, 'model/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, TransformerConv


emb_dims = 256
graph_gnn_layers = 4
text_att_layers = 8
train_batch_size = 32
version = '6-3'
model_name = 'HBCom'
params = dict(model_dir=model_dir,
              model_name=model_name,
              model_id=None,
              emb_dims=emb_dims,
              graph_gnn_layers=graph_gnn_layers,
              graph_GNN=SAGEConv,
              graph_gnn_aggr='mean',
              text_att_layers=text_att_layers,
              text_att_heads=8,
              text_att_head_dims=None,
              text_ff_hid_dims=4 * emb_dims,
              drop_rate=0.2,
              copy=True,
              pad_idx=0,
              train_batch_size=train_batch_size,
              pred_batch_size=math.ceil(train_batch_size * 1.25),
              max_train_size=-1,
              max_valid_size=-1,
              max_big_epochs=100,
              early_stop=10,
              regular_rate=1e-5,
              lr_base=5e-4,
              lr_decay=0.95,
              min_lr_rate=0.01,
              warm_big_epochs=3,
              beam_width=5,
              start_valid_epoch=60,
              gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
              train_mode=True)
train_metrics = [get_sent_bleu]
valid_metric = get_sent_bleu
test_metrics = [get_rouge, get_cider, get_meteor,
                get_sent_bleu1, get_sent_bleu2, get_sent_bleu3, get_sent_bleu4, get_sent_bleu,
                get_corp_bleu1, get_corp_bleu2, get_corp_bleu3, get_corp_bleu4, get_corp_bleu]  # [get_corp_bleu]

res_dir = os.path.join(top_data_dir, 'result/')
res_path = os.path.join(res_dir, model_name + '.json')

import random
import torch
import numpy as np


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(1024)
