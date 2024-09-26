import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DualCopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.tgt_softmax = nn.Softmax(dim=-1)
        self.copy_attention1 = CrossAttention(query_dims=tgt_dims,
                                              key_dims=src_dims,
                                              drop_rate=drop_rate,
                                              pad_idx=kwargs['pad_idx']
                                              )
        self.copy_attention2 = CrossAttention(query_dims=tgt_dims,
                                              key_dims=src_dims,
                                              drop_rate=drop_rate,
                                              pad_idx=kwargs['pad_idx']
                                              )
        self.linear = nn.Linear(2 * src_dims + tgt_dims, 3)
        self.p_softmax = nn.Softmax(dim=-1)

    def forward(self, tgt_dec_out,
                src1_key, src1_map_idx,
                src2_key, src2_map_idx):
        tgt_output = self.out_fc(tgt_dec_out)
        tgt_output = F.layer_norm(tgt_output, (tgt_output.size(-1),))
        src1_len, src2_len = src1_key.size(1), src2_key.size(1)
        tgt_output = F.pad(tgt_output, (0, src1_len + src2_len),
                           value=self.pad_idx)

        tgt_mask = tgt_dec_out.abs().sum(-1).sign()

        src1_mask = src1_key.abs().sum(-1).sign()
        att1, c1 = self.copy_attention1(query=tgt_dec_out,
                                        key=src1_key,
                                        query_mask=tgt_mask,
                                        key_mask=src1_mask)
        att1 = F.layer_norm(att1, (att1.size(-1),))
        copy_output1 = torch.zeros_like(tgt_output)
        src1_map = src1_map_idx.unsqueeze(dim=1).expand(-1, att1.size(1), -1)
        indices0, indices1, _ = torch.meshgrid(torch.arange(att1.size(0)), torch.arange(att1.size(1)),
                                               torch.arange(att1.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src1_map.flatten())
        copy_output1.index_put_(indices=indices, values=att1.flatten(),
                                accumulate=True)

        src2_mask = src2_key.abs().sum(-1).sign()
        att2, c2 = self.copy_attention2(query=tgt_dec_out,
                                        key=src2_key,
                                        query_mask=tgt_mask,
                                        key_mask=src2_mask)
        att2 = F.layer_norm(att2, (att2.size(-1),))
        copy_output2 = torch.zeros_like(tgt_output)
        src2_map = src2_map_idx.unsqueeze(dim=1).expand(-1, att2.size(1), -1)
        indices0, indices1, _ = torch.meshgrid(torch.arange(att2.size(0)), torch.arange(att2.size(1)),
                                               torch.arange(att2.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src2_map.flatten())
        copy_output2.index_put_(indices=indices, values=att2.flatten(),
                                accumulate=True)

        p = F.softmax(self.linear(torch.cat([tgt_dec_out, c1, c2], dim=-1)), dim=-1)
        p = p.unsqueeze(2).expand(-1, -1, tgt_output.size(2), -1)

        output = (tgt_output.mul(p[:, :, :, 0])).add(copy_output1.mul(p[:, :, :, 1])).add(
            copy_output2.mul(p[:, :, :, 2]))
        return output


class CopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.copy_attention = CrossAttention(query_dims=tgt_dims,
                                             key_dims=src_dims,
                                             drop_rate=drop_rate,
                                             pad_idx=kwargs['pad_idx']
                                             )
        self.tgt_softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(src_dims, 1)
        self.p_softmax = nn.Softmax(dim=-1)

    def forward(self, tgt_dec_out, src_key, src_map_idx):
        tgt_output = self.out_fc(tgt_dec_out)
        tgt_output = F.layer_norm(tgt_output, (tgt_output.size(-1),))
        tgt_output = F.pad(tgt_output, (0, src_key.size(1)),
                           value=self.pad_idx)

        tgt_mask = tgt_dec_out.abs().sum(-1).sign()
        src_mask = src_key.abs().sum(-1).sign()
        att, c = self.copy_attention(query=tgt_dec_out,
                                     key=src_key,
                                     query_mask=tgt_mask,
                                     key_mask=src_mask)
        att = F.layer_norm(att, (att.size(-1),))
        copy_output = torch.zeros_like(tgt_output)
        src_map = src_map_idx.unsqueeze(dim=1).expand(-1, att.size(1), -1)
        indices0, indices1, _ = torch.meshgrid(torch.arange(att.size(0)), torch.arange(att.size(1)),
                                               torch.arange(att.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src_map.flatten())
        copy_output.index_put_(indices=indices, values=att.flatten(), accumulate=True)

        p = torch.sigmoid(self.linear(c))
        p = p.expand(-1, -1, copy_output.size(2))
        output = (tgt_output.mul(p)).add(copy_output.mul(1 - p))
        return output


class CrossAttention(nn.Module):
    def __init__(self, query_dims, key_dims, drop_rate=0.0, **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.attn = nn.Linear(key_dims + query_dims, query_dims)
        self.v = nn.Linear(query_dims, 1, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.key_dims = key_dims

    def forward(self, query, key, query_mask, key_mask):
        key_len = key.shape[1]
        query_len = query.shape[1]
        query = query.unsqueeze(2).expand(-1, -1, key_len, -1)
        key_ = key.unsqueeze(1).expand(-1, query_len, -1, -1)

        energy = torch.tanh(self.attn(torch.cat((query, key_), dim=-1)))
        del key_
        attention = self.v(energy).squeeze(-1)

        del energy
        out_att = attention.clone()
        if key_mask is not None:
            key_mask_ = key_mask.eq(self.pad_idx)
            key_mask_ = key_mask_.unsqueeze(dim=1).expand(-1, query_len, -1)
            attention = attention.masked_fill(key_mask_, -np.inf)

            out_att = out_att.mul(key_mask[:, None, :].expand(-1, query.size(1), -1).float())
        attention = F.softmax(attention, dim=-1)
        weight = torch.bmm(attention, key)
        weight = self.dropout(weight)
        del attention
        if query_mask is not None:
            query_mask_ = query_mask[:, :, None].repeat(1, 1, self.key_dims)
            weight = weight.mul(query_mask_.float()) 
            out_att = out_att.mul(query_mask[:, :, None].expand(-1, -1, key.size(1)).float())
        return out_att, weight 


class DualMultiCopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 att_heads=8,
                 att_head_dims=None,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.tgt_softmax = nn.Softmax(dim=-1)
        self.copy_attention1 = MultiHeadCopyAttention(query_dims=tgt_dims,
                                                      key_dims=src_dims,
                                                      head_num=att_heads,
                                                      head_dims=att_head_dims,
                                                      drop_rate=drop_rate,
                                                      pad_idx=kwargs['pad_idx']
                                                      )
        self.copy_attention2 = MultiHeadCopyAttention(query_dims=tgt_dims,
                                                      key_dims=src_dims,
                                                      head_num=att_heads,
                                                      head_dims=att_head_dims,
                                                      drop_rate=drop_rate,
                                                      pad_idx=kwargs['pad_idx']
                                                      )
        self.linear = nn.Linear(3 * tgt_dims, 3)
        self.p_softmax = nn.Softmax(dim=-1)

    def forward(self, tgt_dec_out,
                src1_key, src1_map_idx,
                src2_key, src2_map_idx):
        tgt_output = self.out_fc(tgt_dec_out)
        tgt_output = F.layer_norm(tgt_output, (tgt_output.size(-1),))
        src1_len, src2_len = src1_key.size(1), src2_key.size(1)
        tgt_output = F.pad(tgt_output, (0, src1_len + src2_len),
                           value=self.pad_idx)

        tgt_mask = tgt_dec_out.abs().sum(-1).sign()

        src1_mask = src1_key.abs().sum(-1).sign()
        att1, c1 = self.copy_attention1(query=tgt_dec_out,
                                        key=src1_key,
                                        query_mask=tgt_mask,
                                        key_mask=src1_mask)
        att1 = F.layer_norm(att1, (att1.size(-1),))
        copy_output1 = torch.zeros_like(tgt_output)
        src1_map = src1_map_idx.unsqueeze(dim=1).expand(-1, att1.size(1), -1)
        indices0, indices1, _ = torch.meshgrid(torch.arange(att1.size(0)), torch.arange(att1.size(1)),
                                               torch.arange(att1.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src1_map.flatten())
        copy_output1.index_put_(indices=indices, values=att1.flatten(),
                                accumulate=True)

        src2_mask = src2_key.abs().sum(-1).sign()
        att2, c2 = self.copy_attention2(query=tgt_dec_out,
                                        key=src2_key,
                                        query_mask=tgt_mask,
                                        key_mask=src2_mask)
        att2 = F.layer_norm(att2, (att2.size(-1),))
        copy_output2 = torch.zeros_like(tgt_output)
        src2_map = src2_map_idx.unsqueeze(dim=1).expand(-1, att2.size(1), -1)
        indices0, indices1, _ = torch.meshgrid(torch.arange(att2.size(0)), torch.arange(att2.size(1)),
                                               torch.arange(att2.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src2_map.flatten())
        copy_output2.index_put_(indices=indices, values=att2.flatten(),
                                accumulate=True)

        p = F.softmax(self.linear(torch.cat([tgt_dec_out, c1, c2], dim=-1)), dim=-1)
        p = p.unsqueeze(2).expand(-1, -1, tgt_output.size(2), -1)

        output = (tgt_output.mul(p[:, :, :, 0])).add(copy_output1.mul(p[:, :, :, 1])).add(
            copy_output2.mul(p[:, :, :, 2]))
        return output


class MultiCopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 att_heads=8,
                 att_head_dims=None,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.copy_attention = MultiHeadCopyAttention(query_dims=tgt_dims,
                                                     key_dims=src_dims,
                                                     head_num=att_heads,
                                                     head_dims=att_head_dims,
                                                     drop_rate=drop_rate,
                                                     pad_idx=kwargs['pad_idx']
                                                     )
        self.tgt_softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(tgt_dims, 1)
        self.p_softmax = nn.Softmax(dim=-1)

    def forward(self, tgt_dec_out, src_key, src_map_idx):
        tgt_output = self.out_fc(tgt_dec_out)
        tgt_output = F.layer_norm(tgt_output, (tgt_output.size(-1),))
        tgt_output = F.pad(tgt_output, (0, src_key.size(1)),
                           value=self.pad_idx)

        tgt_mask = tgt_dec_out.abs().sum(-1).sign()
        src_mask = src_key.abs().sum(-1).sign()
        att, c = self.copy_attention(query=tgt_dec_out,
                                     key=src_key,
                                     query_mask=tgt_mask,
                                     key_mask=src_mask)
        att = F.layer_norm(att, (att.size(-1),))
        copy_output = torch.zeros_like(tgt_output)
        src_map = src_map_idx.unsqueeze(dim=1).expand(-1, att.size(1), -1)
        indices0, indices1, _ = torch.meshgrid(torch.arange(att.size(0)), torch.arange(att.size(1)),
                                               torch.arange(att.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src_map.flatten())
        copy_output.index_put_(indices=indices, values=att.flatten(), accumulate=True)

        p = torch.sigmoid(self.linear(c))
        p = p.expand(-1, -1, copy_output.size(2))
        output = (tgt_output.mul(p)).add(copy_output.mul(1 - p))
        return output


class MultiHeadCopyAttention(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 head_num=8,
                 head_dims=None,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.query_dims = query_dims
        self.key_dims = query_dims if key_dims is None else key_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.hid_dims = self.head_num * self.head_dims
        self.conv1d_ins = nn.ModuleList([nn.Conv1d(io_dims, self.hid_dims, kernel_size=1, padding=0)
                                         for io_dims in [self.query_dims, self.key_dims, self.key_dims]])

        self.conv1d_out = nn.Conv1d(self.hid_dims, self.query_dims, kernel_size=1, padding=0)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, query, key, query_mask, key_mask, value=None):
        if value is None:
            value = key.clone()  # 深度拷贝
        batch_size = query.size(0)  # B

        query_, key_, value_ = [conv1d_in(x.transpose(1, 2)) for conv1d_in, x in
                                zip(self.conv1d_ins, (query, key, value))]
        query_, key_, value_ = [x.view(batch_size, self.head_num, self.head_dims, -1).transpose(2, 3)
                                for x in (query_, key_, value_)]

        query_ = query_.mul(float(self.head_dims) ** -0.5)
        attention = torch.einsum('abcd,abed->abce', query_, key_)

        out_att = attention.clone()
        if key_mask is not None:
            key_mask_ = key_mask.eq(self.pad_idx)
            key_mask_ = key_mask_.unsqueeze(dim=1).repeat(1, self.head_num, 1)
            key_mask_ = key_mask_.unsqueeze(dim=2).expand(-1, -1, query.size(1), -1)
            attention = attention.masked_fill(key_mask_, -np.inf)

            out_att = out_att.mul(key_mask[:, None, None, :].expand(-1, self.head_num, query.size(1), -1).float())

        attention = self.softmax(attention)

        attention = self.dropout(attention)

        output = torch.matmul(attention, value_)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hid_dims) 
        output = self.conv1d_out(output.transpose(1, 2)).transpose(1, 2) 

        if query_mask is not None:
            query_mask_ = query_mask[:, :, None].repeat(1, 1, self.query_dims) 
            output = output.mul(query_mask_.float()) 
            out_att = out_att.mul(query_mask[:, None, :, None].expand(-1, self.head_num, -1, key.size(1)).float())

        return out_att.mean(dim=1), output 
