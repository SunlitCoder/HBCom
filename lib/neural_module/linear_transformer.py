#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LiTranEnc(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=8,
                 head_dim=None,
                 ff_hid_dim=2048,
                 layer_num=6,
                 drop_rate=0.,
                 ):
        super().__init__()
        self.query_dim = query_dim
        self.head_num = head_num
        self.layer_num=layer_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.enc_blocks = nn.ModuleList([EncBlock(query_dim=self.query_dim,
                                                  head_num=self.head_num,
                                                  head_dim=self.head_dim,
                                                  ff_hid_dim=ff_hid_dim,
                                                  drop_rate=drop_rate,) for _ in range(layer_num)])

    def forward(self, query,query_mask=None):
        flag=0
        if query_mask is None:
            flag=1
            query_len = query.size(1)
            batch_max_query_len = query.abs().sum(-1).sign().sum(-1).max().int()
            query = query[:, :batch_max_query_len, :]
            query_mask = query.abs().sum(-1).sign()
        for i in range(self.layer_num):
            query=self.enc_blocks[i](x=query,x_mask=query_mask)
        if flag==1:
            query=query.mul(query_mask.unsqueeze(-1).expand(-1, -1, self.query_dim).float())
            query=F.pad(query,[0,0,0,query_len-batch_max_query_len,0,0])
        return query

class EncBlock(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=8,
                 head_dim=None,
                 ff_hid_dim=2048,
                 drop_rate=0.,
                 ):
        super().__init__()
        self.query_dim = query_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.res_att=ResEncAtt(query_dim=query_dim,
                            head_num=head_num,
                            head_dim=head_dim,
                            drop_rate=drop_rate
                               )
        self.res_ff = ResFF(in_dim=query_dim,
                            hid_dim=ff_hid_dim,
                            # out_dim=query_dim,
                            drop_rate=drop_rate)
    def forward(self, x,x_mask):
        x=self.res_att(x=x, x_mask=x_mask)
        x=self.res_ff(x,mask=x_mask)
        return x  
  
class ResFF(nn.Module):
    def __init__(self,
                 in_dim=512,
                 hid_dim=2048,
                 drop_rate=0.):
        super().__init__()
        self.feedforward = FeedForward(in_dim=in_dim,
                                    hid_dim=hid_dim,
                                    out_dim=in_dim,
                                    drop_rate=drop_rate)
        self.layer_norm = nn.LayerNorm(in_dim, elementwise_affine=True)
    def forward(self, x,mask=None):
        x_ = self.feedforward(x, mask=mask)  # (B,L-,D)
        x = self.layer_norm(x.add(x_))
        return x

class ResEncAtt(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=8,
                 head_dim=None,
                 drop_rate=0.
                 ):
        super().__init__()
        self.query_dim = query_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.attention = EncAtt(query_dim=self.query_dim,
                                            head_num=self.head_num,
                                            head_dim=self.head_dim,
                                            drop_rate=drop_rate,)
        self.layer_norm = nn.LayerNorm(self.query_dim, elementwise_affine=True)

    def forward(self, x, x_mask):
        x_ = self.attention(x=x, x_mask=x_mask)
        x = self.layer_norm(x_.add(x))
        return x  

class EncAtt(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=5,
                 head_dim=None,
                 drop_rate=0.,
                 ):
        super().__init__()
        self.query_dim=query_dim
        self.head_num=head_num
        self.head_dim=head_dim
        self.head_dim = query_dim // head_num
        self.hid_dim = self.head_num * self.head_dim
        self.conv1d_ins=nn.ModuleList([nn.Conv1d(query_dim, self.hid_dim, kernel_size=3,padding=1) for _ in range(3)])
        
        self.conv1d_gate=nn.Conv1d(query_dim,query_dim, kernel_size=1,padding=0)


        self.conv1d_out=nn.Conv1d(self.hid_dim,query_dim, kernel_size=3,padding=1)

        self.softmax=nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout(drop_rate)

    def forward(self,x,x_mask):
        batch_size=x.size(0)    #B
        query_len=x.size(1)     #L_q
        gate=self.sigmoid(self.conv1d_gate(x.transpose(1, 2)).transpose(1,2))
        query,key,value= [conv1d(x.transpose(1, 2)) for conv1d, x in
                                zip(self.conv1d_ins, (x, x, x))]  ,(B,D,L),(B,D,L)
        query_=query.view(batch_size,self.head_num,self.head_dim,-1,1).permute(0,1,3,4,2)
        query_mask=x_mask.unsqueeze(-1).expand(-1,-1,self.query_dim).float()

        key_t,value_t=[x.view(batch_size,self.head_num,self.head_dim,-1,1).permute(0,1,3,4,2)
                                     for x in (key,value)]

        key_g, value_g = [x.max(dim=-1)[0][:, :, None, None].expand(-1, -1, query_len, -1).
                                view(batch_size, self.head_num, self.head_dim, query_len, -1).permute(0, 1, 3, 4, 2)
                            for x in (key, value)]
        key_,value_=[torch.cat([x_t,x_g],dim=-2) for x_t,x_g
                     in ((key_t,key_g),(value_t,value_g))]

        attention=torch.einsum('abcde,abcfe->abcdf',query_,key_)

        attention=attention / (self.hid_dim**0.5)

        attention = self.softmax(attention)

        attention=self.dropout(attention)

        output=torch.einsum('abcde,abcef->abcdf',attention,value_)
        output=output.squeeze(dim=-2)

        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.head_num*self.head_dim)
        output=self.conv1d_out(output.transpose(1,2)).transpose(1,2)
        output=output.mul(gate).mul(query_mask)

        return output

class FeedForward(nn.Module):
    def __init__(self,
                 in_dim=512,
                 hid_dim=2048,
                 out_dim=None,
                 drop_rate=0.
                 ):
        super().__init__()
        out_dim=in_dim if out_dim is None else out_dim
        self.conv1d_in=nn.Conv1d(in_dim,hid_dim,kernel_size=1,padding=0)
        self.relu=nn.ReLU()
        self.leaky_relu=nn.LeakyReLU()
        self.conv1d_out=nn.Conv1d(hid_dim,out_dim,kernel_size=1)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x,mask=None):
        output = self.conv1d_in(x.transpose(1, 2))  
        output = self.relu(output)  
        output = self.conv1d_out(output).transpose(1, 2)  
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1)) 
            output=output.mul(mask.float())
        output=self.dropout(output)
        return output


class SAttention(nn.Module):
    def __init__(self,
                 unit_num=512,
                 head_num=8,
                 drop_rate=0.,
                 # head_mode=0,
                 residual=True,
                 norm=True,
                 causality=False,
                 ):
        super().__init__()
        self.head_num=head_num
        self.head_dim=unit_num//head_num
        self.hid_dim = self.head_num * self.head_dim
        self.residual=residual
        self.norm=norm
        self.causality=causality
        self.conv1d_ins=nn.ModuleList([nn.Conv1d(unit_num, self.head_dim*head_num, kernel_size=3,padding=1) for _ in range(3)])
        self.conv1d_h=nn.Conv1d(self.head_num,self.head_num,kernel_size=1)
        self.conv1d_out=nn.Conv1d(self.head_dim*head_num,unit_num, kernel_size=1,padding=0)

        self.relu=nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.softmax=nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout(drop_rate)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=3//2)


    def forward(self,t,g,g_mask,t_mask):
        batch_size = t.size(0)  # B
        query_len = 1  # L_q

        query, key, value = [conv1d(x.transpose(1, 2)) for conv1d, x in
                             zip(self.conv1d_ins, (t, t, t))]  ,(B,D,L),(B,D,L)
        query_=query.max(-1)[0].view(batch_size, self.head_num, self.head_dim, 1).transpose(2, 3)


        key_g=key.max(-1)[0].view(batch_size, self.head_num, self.head_dim, 1).transpose(2, 3)
        key_t=key.view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)
        key_c=self.maxpool1d(key).view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)

        key_=torch.cat([key_g,key_c],dim=-2)

        value_g = value.max(-1)[0].view(batch_size, self.head_num, self.head_dim, 1).transpose(2, 3)
        value_t = value.view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)
        value_c=self.maxpool1d(value).view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)
        value_ = torch.cat([value_g, value_c], dim=-2)

        attention=torch.einsum('abcd,abed->abce',query_,key_)
        attention=attention / (self.hid_dim**0.5)

        if t_mask is not None:
            key_mask=torch.cat([g_mask.float(),t_mask.float()],dim=-1)
            key_mask_=key_mask.eq(0)
            key_mask_=key_mask_[:,None,None,:].expand(-1,self.head_num,query_len,-1)
            attention=attention.masked_fill(key_mask_,-np.inf)

        if self.causality:
            seq_mask=torch.triu(torch.ones_like(attention[0,:,:],dtype=torch.unit8),diagonal=1)
            seq_mask=seq_mask[None,None,:,:].expand(batch_size,self.head_num,-1,-1)
            attention=attention.masked_fill(seq_mask,-np.inf)

        attention = self.softmax(attention)

        if g_mask is not None:
            query_mask=g_mask[:,None,:,None].expand(-1,self.head_num,-1,key_.size(-2))
            attention=attention.mul(query_mask.float())

        attention=self.dropout(attention)
        output=torch.einsum('abcd,abdf->abcf',attention,value_)

        output=self.conv1d_h(output.view(batch_size,self.head_num,-1)).view(batch_size,self.head_num,-1,self.head_dim)
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.head_num*self.head_dim)
        output=self.conv1d_out(output.transpose(1,2)).transpose(1,2)

        return output