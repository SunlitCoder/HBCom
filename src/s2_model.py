import re
import sys

from lib.neural_module.learn_strategy import LrWarmUp
from lib.neural_module.transformer import TranEnc, TranDec, DualTranDec, ResFF, ResMHA
from lib.neural_module.embedding import PosEnc
from lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
from lib.neural_module.copy_attention import DualMultiCopyGenerator, MultiCopyGenerator, DualCopyGenerator
from lib.neural_module.beam_search import trans_beam_search
from lib.neural_model.seq_to_seq_model import TransSeq2Seq
from lib.neural_model.base_model import BaseNet
from typing import Any, Optional, Union

from config import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader.data_list_loader import DataListLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.data.storage import (BaseStorage, NodeStorage, EdgeStorage)
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.nn import HeteroConv, GraphNorm
import random
import numpy as np
import os
import logging
import pickle
import json
import codecs
from tqdm import tqdm
import pickle
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]


class Datax(HeteroData):
    def __cat_dim__(self, key: str, value: Any,
                    store: Optional[NodeOrEdgeStorage] = None, *args,
                    **kwargs) -> Any:
        if bool(re.search('(token)', key)):  # |map
            return None  # generate a new 0 dimension
        if bool(re.search('(pos)', key)):
            return -1
        return super().__cat_dim__(key, value, store)  # return不能漏了！！！

class Datasetx(Dataset):
    def __init__(self,
                 code_graphs,
                 texts=None,
                 ids=None,
                 text_max_len=None,
                 text_begin_idx=1,
                 text_end_idx=2,
                 pad_idx=0):
        self.len = len(code_graphs)
        self.text_max_len = text_max_len
        self.text_begin_idx = text_begin_idx
        self.text_end_idx = text_end_idx

        if text_max_len is None and texts is not None:
            self.text_max_len = max([len(text) for text in texts])  # 每个输出只是一个序列
        self.code_graphs = code_graphs
        self.texts = texts
        self.ids = ids
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        if self.texts is None:
            pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)  # decoder端的输入
            pad_text_in[0] = self.text_begin_idx
            pad_text_out = None
        else:
            tru_text = self.texts[index][:self.text_max_len]  # 先做截断
            pad_text_in = np.lib.pad(tru_text,
                                     (1, self.text_max_len - len(tru_text)),
                                     'constant',
                                     constant_values=(self.text_begin_idx, self.pad_idx))
            tru_text_out = np.lib.pad(tru_text,
                                      (0, 1),
                                      'constant',
                                      constant_values=(0, self.text_end_idx))  # padding
            pad_text_out = np.lib.pad(tru_text_out,
                                      (0, self.text_max_len + 1 - len(tru_text_out)),
                                      'constant',
                                      constant_values=(self.pad_idx, self.pad_idx))  # padding
        data = Datax()
        data['node'].x = torch.tensor(self.code_graphs[index]['nodes'])
        data['node'].src_map = torch.tensor(self.code_graphs[index]['node2text_map_ids']).long()
        data['node'].code_mask = torch.tensor(self.code_graphs[index]['code_node_mask']).bool()
        data['node', 'base_child', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['base_father2child_edges']).long()
        data['node', 'base_father', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['base_child2father_edges']).long()
        data['node', 'sibling_next', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['sibling_prev2next_edges']).long()
        data['node', 'sibling_prev', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['sibling_next2prev_edges']).long()
        data['node', 'dfg_next', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['dfg_prev2next_edges']).long()
        data['node', 'dfg_prev', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['dfg_next2prev_edges']).long()
        data['node', 'code_next', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['code_prev2next_edges']).long()
        data['node', 'code_prev', 'node'].edge_index = torch.tensor(
            self.code_graphs[index]['code_next2prev_edges']).long()
        data['text'].text_token_input = torch.tensor(pad_text_in).long()
        if self.texts is not None:
            data['text'].text_token_output = torch.tensor(pad_text_out).long()
        data['text'].num_nodes = pad_text_in.shape[0]
        if self.ids is not None:
            data['idx'].idx = torch.tensor(self.ids[index])
            data['idx'].num_nodes = 1
        return data

    def __len__(self):
        return self.len


class CodeGraphEnc(nn.Module):
    def __init__(self,
                 emb_dims,
                 graph_max_size,
                 code_max_len,
                 graph_node_emb_op,
                 graph_gnn_layers=6,
                 graph_GNN=SAGEConv,
                 graph_gnn_aggr='mean',
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.graph_max_size = graph_max_size
        self.code_max_len = code_max_len
        self.emb_dims = emb_dims

        self.graph_node_emb_op = graph_node_emb_op
        self.emb_drop_op = nn.Dropout(p=drop_rate)

        self.gnn_layers = graph_gnn_layers
        self.gnn_ops = nn.ModuleList()
        self.gnorm_ops = nn.ModuleList()
        self.grelu_ops = nn.ModuleList()
        for _ in range(graph_gnn_layers):
            if graph_GNN == TransformerConv:
                print('#' * 50 + 'Transformer')
                gnn = HeteroConv({
                    ('node', 'base_child', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                    heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                    root_weight=True),
                    ('node', 'base_father', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                     heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                     root_weight=False),
                    ('node', 'sibling_next', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                      heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                      root_weight=False),
                    ('node', 'sibling_prev', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                      heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                      root_weight=False),
                    ('node', 'dfg_next', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                  heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                  root_weight=False),
                    ('node', 'dfg_prev', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                  heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                  root_weight=False),
                    ('node', 'code_next', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                   heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                   root_weight=False),
                    ('node', 'code_prev', 'node'): TransformerConv((emb_dims, emb_dims), out_channels=emb_dims // 8,
                                                                   heads=8, aggr=graph_gnn_aggr, dropout=drop_rate,
                                                                   root_weight=False),
                }, aggr='sum')
                assert emb_dims / 8 == emb_dims // 8
            elif graph_GNN == GCNConv:
                print('#' * 50 + 'GCN')
                gnn = HeteroConv({
                    ('node', 'base_child', 'node'): GCNConv(emb_dims, emb_dims // 8),
                    ('node', 'base_father', 'node'): GCNConv(emb_dims, emb_dims // 8),
                    ('node', 'sibling_next', 'node'): GCNConv(emb_dims, emb_dims // 8),
                    ('node', 'sibling_prev', 'node'): GCNConv(emb_dims, emb_dims // 8),
                    ('node', 'dfg_next', 'node'): GCNConv(emb_dims, emb_dims // 8),
                    ('node', 'dfg_prev', 'node'): GCNConv(emb_dims, emb_dims // 8),
                    ('node', 'code_next', 'node'): GCNConv(emb_dims, emb_dims // 8),
                    ('node', 'code_prev', 'node'): GCNConv(emb_dims, emb_dims // 8),
                }, aggr='sum')

            elif graph_GNN == SAGEConv:
                print('#' * 50 + 'SAGE')
                gnn = HeteroConv({
                    ('node', 'base_child', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                                                              root_weight=True),
                    ('node', 'base_father', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                                                               root_weight=False),
                    # ('node', 'sibling_next', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                    #                                             root_weight=False),
                    # ('node', 'sibling_prev', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                    #                                             root_weight=False),
                    ('node', 'dfg_next', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                                                            root_weight=False),
                    ('node', 'dfg_prev', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                                                            root_weight=False),
                    ('node', 'code_next', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                                                             root_weight=False),
                    ('node', 'code_prev', 'node'): graph_GNN((emb_dims, emb_dims), emb_dims, aggr=graph_gnn_aggr,
                                                             root_weight=False),
                }, aggr='sum')
            elif graph_GNN == GATConv:
                print('#' * 50 + 'GAT')
                gnn = HeteroConv({
                    ('node', 'base_child', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                            dropout=drop_rate),
                    ('node', 'base_father', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                             dropout=drop_rate),
                    ('node', 'sibling_next', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                              dropout=drop_rate),
                    ('node', 'sibling_prev', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                              dropout=drop_rate),
                    ('node', 'dfg_next', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                          dropout=drop_rate),
                    ('node', 'dfg_prev', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                          dropout=drop_rate),
                    ('node', 'code_next', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                           dropout=drop_rate),
                    ('node', 'code_prev', 'node'): GATConv((emb_dims, emb_dims), heads=8, concat=True,
                                                           dropout=drop_rate),
                }, aggr='sum')
            else:
                gnn = HeteroConv({
                    ('node', 'base_child', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    ('node', 'base_father', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    ('node', 'sibling_next', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    ('node', 'sibling_prev', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    ('node', 'dfg_next', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    ('node', 'dfg_prev', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    ('node', 'code_next', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                    ('node', 'code_prev', 'node'): graph_GNN(emb_dims, emb_dims, aggr=graph_gnn_aggr),
                }, aggr='sum')
            self.gnn_ops.append(gnn)
            self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
            self.gnorm_ops.append(GraphNorm(emb_dims))

    def forward(self, data):
        assert len(data['node'].x.size()) == 1 
        assert len(data['node'].src_map.size()) == 1  # [batch_graph_node_num,]
        assert len(data['node'].code_mask.size()) == 1  # [batch_graph_node_num,]
        assert len(data.edge_index_dict[('node', 'base_child', 'node')].size()) == 2
        assert len(data.edge_index_dict[('node', 'base_father', 'node')].size()) == 2
        assert len(data.edge_index_dict[('node', 'sibling_prev', 'node')].size()) == 2
        assert len(data.edge_index_dict[('node', 'sibling_next', 'node')].size()) == 2
        assert len(data.edge_index_dict[('node', 'dfg_prev', 'node')].size()) == 2
        assert len(data.edge_index_dict[('node', 'dfg_next', 'node')].size()) == 2
        assert len(data.edge_index_dict[('node', 'code_prev', 'node')].size()) == 2
        assert len(data.edge_index_dict[('node', 'code_next', 'node')].size()) == 2

        graph_node_emb = self.graph_node_emb_op(data.x_dict['node'])  ##[batch_graph_node_num,emb_dims]
        data['node'].x = self.emb_drop_op(graph_node_emb)  ##[batch_graph_node_num,emb_dims]

        code_x_batch = data.x_batch_dict['node'][data['node'].code_mask == True]  # [batch_leaf_node_num,]

        for gnn, relu, norm in zip(self.gnn_ops, self.grelu_ops, self.gnorm_ops):
            x_dict = gnn(x_dict=data.x_dict,
                         edge_index_dict=data.edge_index_dict)  # dict(xx_node:[batch_xx_node_num,hid_dims])
            data['node'].x = norm(data['node'].x.add(
                relu(x_dict['node'])))  # data[key].x residual connection ,batch=data.x_batch_dict['node']

        graph_enc, _ = to_dense_batch(data.x_dict['node'],
                                      batch=data.x_batch_dict['node'],  # data['leaf'].x_batch也可以
                                      fill_value=self.pad_idx,
                                      max_num_nodes=self.graph_max_size)  # [batch_size,graph_max_size,emb_dims],[batch_size,graph_max_size]

        code_src_map, _ = to_dense_batch(data.src_map_dict['node'][data['node'].code_mask == True],
                                         batch=code_x_batch,  # data['leaf'].x_batch也可以
                                         fill_value=self.pad_idx,
                                         max_num_nodes=self.code_max_len)  # [batch_data_num,code_max_len]
        graph_code_enc, _ = to_dense_batch(data.x_dict['node'][data['node'].code_mask == True],
                                           batch=code_x_batch,  # data['leaf'].x_batch也可以
                                           fill_value=self.pad_idx,
                                           max_num_nodes=self.code_max_len)  # [batch_data_num,code_max_len]

        return graph_enc, graph_code_enc, code_src_map


class Dec(nn.Module):
    def __init__(self,
                 emb_dims,
                 text_voc_size,
                 text_emb_op,
                 text_max_len,
                 enc_out_dims,
                 att_layers,
                 att_heads,
                 att_head_dims=None,
                 ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        kwargs.setdefault('copy', True)
        self._copy = kwargs['copy']
        self.emb_dims = emb_dims
        self.text_voc_size = text_voc_size

        self.text_emb_op = text_emb_op

        self.pos_encoding = PosEnc(max_len=text_max_len + 1, emb_dims=emb_dims, train=True, pad=True,
                                   pad_idx=kwargs['pad_idx'])
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        self.text_dec_op = TranDec(query_dims=emb_dims,
                                   key_dims=enc_out_dims,
                                   head_nums=att_heads,
                                   head_dims=att_head_dims,
                                   layer_num=att_layers,
                                   ff_hid_dims=ff_hid_dims,
                                   drop_rate=drop_rate,
                                   pad_idx=kwargs['pad_idx'],
                                   self_causality=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.out_fc = nn.Linear(emb_dims, text_voc_size)
        self.copy_generator = MultiCopyGenerator(tgt_dims=emb_dims,
                                                 tgt_voc_size=text_voc_size,
                                                 src_dims=enc_out_dims,
                                                 att_heads=att_heads,
                                                 att_head_dims=att_head_dims,
                                                 drop_rate=drop_rate,
                                                 pad_idx=kwargs['pad_idx'])

    def forward(self, graph_enc, graph_code_enc, code_src_map, text_input):

        text_emb = self.text_emb_op(text_input) 
        text_emb = text_emb * np.sqrt(self.emb_dims)
        pos_emb = self.pos_encoding(text_input)
        text_dec = self.dropout(text_emb.add(pos_emb)) 
        text_dec = self.emb_layer_norm(text_dec) 

        graph_mask = graph_enc.abs().sum(-1).sign()
        text_mask = text_input.abs().sign()
        text_dec = self.text_dec_op(query=text_dec,
                                    key=graph_enc,
                                    query_mask=text_mask,
                                    key_mask=graph_mask
                                    ) 

        if not self._copy:
            text_output = self.out_fc(text_dec)
        else:
            text_output = self.copy_generator(text_dec,
                                              graph_code_enc, code_src_map)
        return text_output.transpose(1, 2)


class TNet(BaseNet):
    def __init__(self,
                 emb_dims,
                 graph_max_size,
                 code_max_len,
                 text_max_len,
                 io_voc_size,
                 text_voc_size,
                 graph_gnn_layers=6,
                 graph_GNN=SAGEConv,
                 graph_gnn_aggr='add',
                 text_att_layers=3,
                 text_att_heads=8,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('copy', True)
        kwargs.setdefault('pad_idx', 0)
        self.init_params = locals()
        io_token_emb_op = nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        nn.init.xavier_uniform_(io_token_emb_op.weight[1:, ])
        self.enc_op = CodeGraphEnc(emb_dims=emb_dims,
                                   graph_max_size=graph_max_size,
                                   code_max_len=code_max_len,
                                   graph_node_emb_op=io_token_emb_op,
                                   graph_gnn_layers=graph_gnn_layers,
                                   graph_GNN=graph_GNN,
                                   graph_gnn_aggr=graph_gnn_aggr,
                                   drop_rate=drop_rate,
                                   pad_idx=kwargs['pad_idx'])
        self.dec_op = Dec(emb_dims=emb_dims,
                          text_voc_size=text_voc_size,
                          text_max_len=text_max_len,
                          text_emb_op=io_token_emb_op,
                          enc_out_dims=emb_dims,
                          att_layers=text_att_layers,
                          att_heads=text_att_heads,
                          att_head_dims=text_att_head_dims,
                          ff_hid_dims=text_ff_hid_dims,
                          drop_rate=drop_rate,
                          copy=kwargs['copy'],
                          pad_idx=kwargs['pad_idx'])

    def forward(self, code_graph):
        text_input = code_graph['text'].text_token_input.clone()
        del code_graph['text']
        graph_enc, graph_code_enc, code_src_map = self.enc_op(data=code_graph)
        text_output = self.dec_op(graph_enc=graph_enc, graph_code_enc=graph_code_enc,
                                  code_src_map=code_src_map,
                                  text_input=text_input)
        return text_output


class TModel(TransSeq2Seq):
    def __init__(self,
                 model_dir,
                 model_name='Transformer_based_model',
                 model_id=None,
                 emb_dims=512,
                 graph_gnn_layers=3,
                 graph_GNN=SAGEConv,
                 graph_gnn_aggr='add',
                 text_att_layers=3,
                 text_att_heads=8,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                 copy=True,
                 pad_idx=0,
                 train_batch_size=32,
                 pred_batch_size=32,
                 max_train_size=-1,
                 max_valid_size=32 * 10,
                 max_big_epochs=20,
                 regular_rate=1e-5,
                 lr_base=0.001,
                 lr_decay=0.9,
                 min_lr_rate=0.01,
                 warm_big_epochs=2,
                 start_valid_epoch=20,
                 early_stop=20,
                 Net=TNet,
                 Dataset=Datasetx,
                 beam_width=1,
                 train_metrics=[get_sent_bleu],
                 valid_metric=get_sent_bleu,
                 test_metrics=[get_sent_bleu],
                 train_mode=True,
                 **kwargs
                 ):
        logging.info('Construct %s' % model_name)
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         model_id=model_id)
        self.init_params = locals()
        self.emb_dims = emb_dims
        self.graph_gnn_layers = graph_gnn_layers
        self.graph_GNN = graph_GNN
        self.graph_gnn_aggr = graph_gnn_aggr
        self.text_att_layers = text_att_layers
        self.text_att_heads = text_att_heads
        self.text_att_head_dims = text_att_head_dims
        self.text_ff_hid_dims = text_ff_hid_dims
        self.drop_rate = drop_rate
        self.pad_idx = pad_idx
        self.copy = copy
        self.train_batch_size = train_batch_size
        self.pred_batch_size = pred_batch_size
        self.max_train_size = max_train_size
        self.max_valid_size = max_valid_size
        self.max_big_epochs = max_big_epochs
        self.regular_rate = regular_rate
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.min_lr_rate = min_lr_rate
        self.warm_big_epochs = warm_big_epochs
        self.start_valid_epoch = start_valid_epoch
        self.early_stop = early_stop
        self.Net = Net
        self.Dataset = Dataset
        self.beam_width = beam_width
        self.train_metrics = train_metrics
        self.valid_metric = valid_metric
        self.test_metrics = test_metrics
        self.train_mode = train_mode

    def _logging_paramerter_num(self):
        logging.info("{} have {} paramerters in total".format(self.model_name, sum(
            x.numel() for x in self.net.parameters() if x.requires_grad)))
        code_graph_enc_param_num = sum(
            x.numel() for x in self.net.module.enc_op.gnn_ops.parameters() if x.requires_grad) + \
                                   sum(x.numel() for x in self.net.module.enc_op.gnorm_ops.parameters() if
                                       x.requires_grad) + \
                                   sum(x.numel() for x in self.net.module.enc_op.grelu_ops.parameters() if
                                       x.requires_grad)

        text_dec_param_num = sum(x.numel() for x in self.net.module.dec_op.text_dec_op.parameters() if x.requires_grad)
        enc_dec_param_num = code_graph_enc_param_num + text_dec_param_num
        logging.info("{} have {} paramerters in encoder".format(self.model_name, code_graph_enc_param_num))
        logging.info("{} have {} paramerters in decoder".format(self.model_name, text_dec_param_num))
        logging.info("{} have {} paramerters in encoder and decoder".format(self.model_name, enc_dec_param_num))

    def fit(self,
            train_data,
            valid_data,
            **kwargs
            ):
        logging.info("开始进行fit...")
        self.graph_max_size = 0
        self.code_max_len = 0
        self.io_voc_size = 0
        self.text_max_len = 0
        for code_graph, text in zip(train_data['code_graphs'], train_data['texts']):
            self.graph_max_size = max(self.graph_max_size, len(code_graph['nodes']))
            self.code_max_len = max(self.code_max_len, code_graph['code_node_mask'].sum())
            self.io_voc_size = max(self.io_voc_size, max(code_graph['nodes']))
            self.text_max_len = max(self.text_max_len, len(text))
        self.io_voc_size += 1

        self.text_voc_size = len(train_data['text_dic']['text_i2w'])  # 包含了begin_idx和end_idx
        self.io_voc_size = max(self.io_voc_size, self.text_voc_size + 2 * self.code_max_len)

        net = self.Net(
            emb_dims=self.emb_dims,
            graph_max_size=self.graph_max_size,
            code_max_len=self.code_max_len,
            text_max_len=self.text_max_len,
            io_voc_size=self.io_voc_size,
            text_voc_size=self.text_voc_size,
            graph_gnn_layers=self.graph_gnn_layers,
            graph_GNN=self.graph_GNN,
            graph_gnn_aggr=self.graph_gnn_aggr,
            text_att_layers=self.text_att_layers,
            text_att_heads=self.text_att_heads,
            text_att_head_dims=self.text_att_head_dims,
            text_ff_hid_dims=self.text_ff_hid_dims,
            drop_rate=self.drop_rate,
            pad_idx=self.pad_idx,
            copy=self.copy
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = DataParallel(net.to(device), follow_batch=['x'])
        self._logging_paramerter_num()
        self.net.train()

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr_base,
                                    weight_decay=self.regular_rate)

        self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)

        self.text_begin_idx = self.text_voc_size - 1
        self.text_end_idx = self.text_voc_size - 2
        self.tgt_begin_idx, self.tgt_end_idx = self.text_begin_idx, self.text_end_idx
        assert train_data['text_dic']['text_i2w'][self.text_end_idx] == OUT_END_TOKEN
        assert train_data['text_dic']['text_i2w'][self.text_begin_idx] == OUT_BEGIN_TOKEN

        self.max_train_size = len(train_data['code_graphs']) if self.max_train_size == -1 else self.max_train_size
        train_code_graphs, train_texts, train_ids = zip(
            *random.sample(list(zip(train_data['code_graphs'], train_data['texts'], train_data['ids'])),
                           min(self.max_train_size,
                               len(train_data['code_graphs']))
                           )
        )

        train_set = self.Dataset(code_graphs=train_code_graphs,
                                 texts=train_texts,
                                 ids=train_ids,
                                 text_max_len=self.text_max_len,
                                 text_begin_idx=self.text_begin_idx,
                                 text_end_idx=self.text_end_idx,
                                 pad_idx=self.pad_idx)
        train_loader = DataListLoader(dataset=train_set,
                                      batch_size=self.train_batch_size,
                                      shuffle=True,
                                      drop_last=True)

        if self.warm_big_epochs is None:
            self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
        self.scheduler = LrWarmUp(self.optimizer,
                                  min_rate=self.min_lr_rate,
                                  lr_decay=self.lr_decay,
                                  warm_steps=self.warm_big_epochs * len(train_loader),
                                  reduce_steps=len(train_loader))

        # 在程序开始前检查是否存在检查点文件
        checkpoint_dir = 'checkpoints' + version
        checkpoint_filename = os.path.join(checkpoint_dir, 'checkpoint.pth')

        worse_epochs = -1
        best_valid_eval = -1

        if os.path.exists(checkpoint_filename):
            checkpoint = torch.load(checkpoint_filename)
            # 加载模型状态
            self.net.load_state_dict(checkpoint['model_state_dict'])
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 加载学习率调度器状态
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 获取上一次的 epoch
            start_epoch = checkpoint['epoch']

            worse_epochs = checkpoint['worse_epochs']
            best_valid_eval = checkpoint['best_valid_eval']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0
            print("Starting training from scratch")

        print("self.start_valid_epoch : " + str(self.start_valid_epoch))

        if self.train_mode:
            for current_epoch in range(start_epoch, self.max_big_epochs):
                pbar = tqdm(train_loader)
                for j, batch_data in enumerate(pbar):
                    batch_text_output = []
                    ids = []
                    for data in batch_data:
                        batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
                        del data['text'].text_token_output
                        ids.append(data['idx'].idx.item())
                        del data['idx']

                    batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    pred_text_output = self.net(batch_data)

                    loss = self.criterion(pred_text_output, batch_text_output)  # 计算loss
                    self.optimizer.zero_grad()  # 梯度置0
                    loss.backward()  # 反向传播
                    self.optimizer.step()
                    self.scheduler.step()

                    text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
                                'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
                    log_info = self._get_log_fit_eval(loss=loss,
                                                      pred_tgt=pred_text_output,
                                                      gold_tgt=batch_text_output,
                                                      tgt_i2w=text_dic
                                                      )
                    log_info = '[Big epoch:{}/{},{}]'.format(current_epoch + 1, self.max_big_epochs, log_info)
                    pbar.set_description(log_info)
                    del pred_text_output, batch_text_output, batch_data

                del pbar

                if current_epoch + 1 > self.start_valid_epoch:
                    print("当前epoch：" + str(current_epoch + 1) + ", 进入评估训练阶段")
                    self.max_valid_size = len(
                        valid_data['code_graphs']) if self.max_valid_size == -1 else self.max_valid_size
                    valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_graphs'],
                                                                                       valid_data['texts'],
                                                                                       valid_data['text_dic'][
                                                                                           'ex_text_i2ws'])),
                                                                              min(self.max_valid_size,
                                                                                  len(valid_data['code_graphs']))
                                                                              )
                                                               )
                    text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
                                'ex_text_i2ws': ex_text_i2ws}
                    worse_epochs, valid_eval = self._do_validation(valid_srcs=valid_srcs,
                                                                   # valid_data['code_graphs']
                                                                   valid_tgts=valid_tgts,  # valid_data['texts']
                                                                   tgt_i2w=text_dic,  # valid_data['text_dic']
                                                                   increase_better=True,
                                                                   last=False,
                                                                   best_valid_eval=best_valid_eval)  # 根据验证集loss选择best_net
                    print("当前的worse_epochs为：" + str(worse_epochs))
                    if (worse_epochs + 1) >= self.early_stop:
                        break
                else:
                    valid_eval = -1
                    worse_epochs = -1

                # 保存检查点
                # 检查目录是否存在，如果不存在则创建
                checkpoint_dir = 'checkpoints' + version
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_filename = os.path.join(checkpoint_dir, 'checkpoint.pth')

                print("保存检查点:epoch" + str(current_epoch + 1))
                torch.save({
                    'epoch': current_epoch + 1,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss,
                    'worse_epochs': worse_epochs,  # 保存验证过程的信息
                    'best_valid_eval': valid_eval,  # 保存验证过程中的最优得分
                    # 可以添加其他需要保存的内容
                }, checkpoint_filename)

        # torch.cuda.empty_cache()
        self._do_validation(valid_srcs=valid_data['code_graphs'],
                            valid_tgts=valid_data['texts'],
                            tgt_i2w=valid_data['text_dic'],
                            increase_better=True,
                            last=True)  # 根据验证集loss选择best_net
        self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name

    def predict(self,
                code_graphs,
                text_dic):
        logging.info('Predict outputs of %s' % self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        self.net.eval()
        enc_op = DataParallel(self.net.module.enc_op, follow_batch=['x'])
        dec_op = torch.nn.DataParallel(self.net.module.dec_op)
        # enc.eval()
        # dec.eval()
        data_set = self.Dataset(code_graphs=code_graphs,
                                texts=None,
                                ids=None,
                                text_max_len=self.text_max_len,
                                text_begin_idx=self.text_begin_idx,
                                text_end_idx=self.text_end_idx,
                                pad_idx=self.pad_idx)  # 数据集，没有out，不需要id

        data_loader = DataListLoader(dataset=data_set,
                                     batch_size=self.pred_batch_size,  # 1.5,2.5
                                     shuffle=False)
        pred_text_id_np_batches = []
        with torch.no_grad(): 
            pbar = tqdm(data_loader)
            for batch_data in pbar:
                batch_text_input = []
                for data in batch_data:
                    batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
                    del data['text']
                batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

                batch_graph_enc, batch_graph_code_enc, batch_code_src_map = enc_op(batch_data)
                batch_text_output: list = [] 
                if self.beam_width == 1:
                    for i in range(self.text_max_len + 1): 
                        pred_out = dec_op(graph_enc=batch_graph_enc, graph_code_enc=batch_graph_code_enc,
                                          code_src_map=batch_code_src_map,
                                          text_input=batch_text_input) 
                        batch_text_output.append(
                            pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy()) 
                        if i < self.text_max_len:
                            batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
                    batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1] 
                    batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf 
                    batch_pred_text[:, self.pad_idx, :] = -np.inf 
                    batch_pred_text_np = np.argmax(batch_pred_text, axis=1)
                    pred_text_id_np_batches.append(batch_pred_text_np)
                else:
                    batch_pred_text = trans_beam_search(net=dec_op,
                                                        beam_width=self.beam_width,
                                                        dec_input_arg_name='text_input',
                                                        length_penalty=1,
                                                        begin_idx=self.tgt_begin_idx,
                                                        pad_idx=self.pad_idx,
                                                        end_idx=self.tgt_end_idx,
                                                        graph_enc=batch_graph_enc,
                                                        graph_code_enc=batch_graph_code_enc,
                                                        code_src_map=batch_code_src_map,
                                                        text_input=batch_text_input
                                                        )  # (B,L_tgt)

                    pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:, :-1])  # [(B,L_tgt)]

        pred_text_id_np = np.concatenate(pred_text_id_np_batches, axis=0)  # (AB,tgt_voc_size,L_tgy)
        self.net.train()  # 切换回训练模式
        # 利用字典将msg转为token
        pred_texts = self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)

        return pred_texts  # 序列概率输出形状为（A,D)

    def generate_texts(self, code_graphs, text_dic, res_path, gold_texts, raw_data, token_data, **kwargs):
        logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
        kwargs.setdefault('beam_width', 1)
        res_dir = os.path.dirname(res_path)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        pred_texts = self.predict(code_graphs=code_graphs,
                                  text_dic=text_dic
                                  )
        gold_texts = self._tgt_ids2tokens(gold_texts, text_dic, self.pad_idx)
        res_data = []
        for i, (pred_text, gold_text, raw_item, token_item) in \
                enumerate(zip(pred_texts, gold_texts, raw_data, token_data)):
            sent_bleu = self.valid_metric([pred_text], [gold_text])
            res_data.append(dict(pred_text=' '.join(pred_text),
                                 gold_text=' '.join(gold_text),
                                 sent_bleu=sent_bleu,
                                 raw_code=raw_item['code'],
                                 raw_text=raw_item['text'],
                                 id=raw_item['id'],
                                 token_text=token_item['text'],
                                 ))
        with codecs.open(res_path, 'w', encoding='utf-8') as f:
            json.dump(res_data, f, indent=4, ensure_ascii=False)
        self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name
        logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

    def _code_ids2tokens(self, code_idss, code_i2w, end_idx):
        return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)]
                                           if end_idx in code_ids else code_ids)]
                for code_ids in code_idss]

    def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
        if self.copy:
            text_tokens: list = []
            for j, text_ids in enumerate(text_id_np):
                text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
                end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
                text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
        else:
            text_i2w = text_dic['text_i2w']
            text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)]
                                                      if end_idx in text_ids else text_ids)]
                           for text_ids in text_id_np]

        return text_tokens


if __name__ == '__main__':

    logging.info(
        'Parameters are listed below: \n' + '\n'.join(['{}: {}'.format(key, value) for key, value in params.items()]))

    model = TModel(
        model_dir=params['model_dir'],
        model_name=params['model_name'],
        model_id=params['model_id'],
        emb_dims=params['emb_dims'],
        graph_gnn_layers=params['graph_gnn_layers'],
        graph_GNN=params['graph_GNN'],
        graph_gnn_aggr=params['graph_gnn_aggr'],
        text_att_layers=params['text_att_layers'],
        text_att_heads=params['text_att_heads'],
        text_att_head_dims=params['text_att_head_dims'],
        text_ff_hid_dims=params['text_ff_hid_dims'],
        drop_rate=params['drop_rate'],
        copy=params['copy'],
        pad_idx=params['pad_idx'],
        train_batch_size=params['train_batch_size'],
        pred_batch_size=params['pred_batch_size'],
        max_train_size=params['max_train_size'],  # -1 means all
        max_valid_size=params['max_valid_size'],  ####################10
        max_big_epochs=params['max_big_epochs'],
        regular_rate=params['regular_rate'],
        lr_base=params['lr_base'],
        lr_decay=params['lr_decay'],
        min_lr_rate=params['min_lr_rate'],
        warm_big_epochs=params['warm_big_epochs'],
        early_stop=params['early_stop'],
        start_valid_epoch=params['start_valid_epoch'],
        Net=TNet,
        Dataset=Datasetx,
        beam_width=params['beam_width'],
        train_metrics=train_metrics,
        valid_metric=valid_metric,
        test_metrics=test_metrics,
        train_mode=params['train_mode'])

    logging.info('Load data ...')
    with codecs.open(train_avail_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with codecs.open(valid_avail_data_path, 'rb') as f:
        valid_data = pickle.load(f)
    with codecs.open(test_avail_data_path, 'rb') as f:
        test_data = pickle.load(f)


    with codecs.open(test_token_data_path, 'r') as f:
        test_token_data = json.load(f)

    with codecs.open(test_raw_data_path, 'r') as f:
        test_raw_data = json.load(f)

    model.fit(train_data=train_data,
              valid_data=valid_data)

    for key, value in params.items():
        logging.info('{}: {}'.format(key, value))
    logging.info(
        'Parameters are listed below: \n' + '\n'.join(['{}: {}'.format(key, value) for key, value in params.items()]))

    test_eval_df = model.eval(test_srcs=test_data['code_graphs'],
                              test_tgts=test_data['texts'],
                              tgt_i2w=test_data['text_dic'])
    logging.info('Model performance on test dataset:\n')
    for i in range(0, len(test_eval_df.columns), 4):
        print(test_eval_df.iloc[:, i:i + 4])

    model.generate_texts(code_graphs=test_data['code_graphs'],
                         text_dic=test_data['text_dic'],
                         res_path=res_path,
                         gold_texts=test_data['texts'],
                         raw_data=test_raw_data,
                         token_data=test_token_data)
