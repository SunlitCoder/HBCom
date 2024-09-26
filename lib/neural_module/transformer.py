import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Tranformer(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 head_num=8,
                 head_dims=None,
                 layer_num=6,
                 drop_rate=0.,
                 causality=False,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx',0)
        self.pad_idx=kwargs['pad_idx']
        self.query_dims = query_dims
        self.key_dims=query_dims if key_dims is None else key_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.layer_num = layer_num
        self.drop_rate = drop_rate
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.attentions = nn.ModuleList(
            [MultiHeadAttention(query_dims=self.query_dims,
                                key_dims=self.key_dims,
                                head_num=self.head_num,
                                head_dims=self.head_dims,
                                drop_rate=drop_rate,
                                causality=causality,
                                pad_idx=kwargs['pad_idx']) for _ in range(layer_num)]
        )
        self.layer_norms1 = nn.ModuleList(
            [nn.LayerNorm(self.query_dims, elementwise_affine=True) for _ in range(layer_num)]
        )
        self.forwards = nn.ModuleList(
            [FeedForward(in_dims=self.query_dims,
                         hid_dims=self.query_dims * 4,
                         drop_rate=drop_rate) for _ in range(layer_num)]
        )
        self.layer_norms2 = nn.ModuleList(
            [nn.LayerNorm(self.query_dims, elementwise_affine=True) for _ in range(layer_num)]
        )

    def forward(self, query,key,query_mask=None,key_mask=None):
        if query_mask is None:
            query_len = query.size(1)  # L
            batch_max_query_len = query_mask.sum(1).max().int() 
            query = query[:, :batch_max_query_len, :]  
            query_mask = query_mask[:, :batch_max_query_len]  
        if key_mask is None:
            # key_len = query.size(1)  # L
            batch_max_key_len = key_mask.sum(1).max().int() 
            query = query[:, :batch_max_key_len, :]  
            key_mask = key_mask[:, :batch_max_key_len]  
        for i in range(self.layer_num):
            query_ = self.attentions[i](query=query, key=key,query_mask=query_mask,key_mask=key_mask)  
            query=self.layer_norms1[i](query_.add(query))
            query_=self.forwards[i](query,mask=query_mask)   
            query=self.layer_norms2[i](query_.add(query))
        if query_mask is None:
            query=query.mul(query.unsqueeze(-1).expand(-1, -1, self.unit_num).float())
            query=F.pad(query,(0,0,0,query_len-batch_max_query_len,0,0),value=self.pad_idx)
        return query  

class TranEnc(nn.Module):
    def __init__(self,
                 query_dims=512,
                 head_num=8,
                 head_dims=None,
                 ff_hid_dims=2048,
                 layer_num=6,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        # print(layer_num)
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.query_dims = query_dims
        self.head_num = head_num
        self.layer_num=layer_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.enc_blocks = nn.ModuleList([EncBlock(query_dims=self.query_dims,
                                                  head_num=self.head_num,
                                                  head_dims=self.head_dims,
                                                  ff_hid_dims=ff_hid_dims,
                                                  drop_rate=drop_rate,
                                                  pad_idx=kwargs['pad_idx']) for _ in range(layer_num)])

    def forward(self, query,query_mask=None):
        flag=0
        if query_mask is None:
            flag=1
            query_len = query.size(1)  # L
            batch_max_query_len = query.abs().sum(-1).sign().sum(-1).max().int() 
            query = query[:, :batch_max_query_len, :]  
            query_mask = query.abs().sum(-1).sign()  
        for i in range(self.layer_num):
            query=self.enc_blocks[i](query=query,query_mask=query_mask)
        if flag==1:
            query=query.mul(query_mask.unsqueeze(-1).expand(-1, -1, self.query_dims).float())
            query=F.pad(query,[0,0,0,query_len-batch_max_query_len,0,0],value=self.pad_idx)
        return query  

class DualTranDec(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 head_num=8,
                 ff_hid_dims=2048,
                 head_dims=None,
                 layer_num=6,
                 drop_rate=0.,
                 mode='sequential',
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.query_dims = query_dims
        self.key_dims=query_dims if key_dims is None else key_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.layer_num = layer_num
        self.dec_blocks = nn.ModuleList([DualDecBlock(query_dims=self.query_dims,
                                                      key_dims=self.key_dims,
                                                      head_num=self.head_num,
                                                      head_dims=self.head_dims,
                                                      ff_hid_dims=ff_hid_dims,
                                                      drop_rate=drop_rate,
                                                      mode=mode,
                                                      pad_idx=kwargs['pad_idx']) for _ in range(layer_num)])

    def forward(self, query,key1,key2,query_mask=None,key_mask1=None,key_mask2=None):
        flag=0
        if query_mask is None:
            flag=1
            query_len = query.size(1)
            batch_max_query_len = query.abs().sum(-1).sign().sum(-1).max().int() 
            query = query[:, :batch_max_query_len, :]  
            query_mask = query.abs().sum(-1).sign()  
        if key_mask1 is None:
            batch_max_key_len = key1.abs().sum(-1).sign().sum(-1).max().int()
            key1= key1[:, :batch_max_key_len, :]  
            key_mask1 = key1.abs().sum(-1).sign()  
        if key_mask2 is None:
            batch_max_key_len = key2.abs().sum(-1).sign().sum(-1).max().int()
            key2= key2[:, :batch_max_key_len, :]  
            key_mask2 = key2.abs().sum(-1).sign()  
        for i in range(self.layer_num):
            query=self.dec_blocks[i](query=query,
                                     key1=key1,
                                     key2=key2,
                                     query_mask=query_mask,
                                     key_mask1=key_mask1,
                                     key_mask2=key_mask2,
                                     )
        if flag==1:
            query=query.mul(query_mask.unsqueeze(-1).expand(-1, -1, self.query_dims).float())
            query=F.pad(query,[0,0,0,query_len-batch_max_query_len,0,0],value=self.pad_idx) 
        return query  

class TranDec(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 head_num=8,
                 ff_hid_dims=2048,
                 head_dims=None,
                 layer_num=6,
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        kwargs.setdefault('self_causality',True)
        self.query_dims = query_dims
        self.key_dims=query_dims if key_dims is None else key_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.layer_num = layer_num
        self.dec_blocks = nn.ModuleList([DecBlock(query_dims=self.query_dims,
                                                  key_dims=self.key_dims,
                                                  head_num=self.head_num,
                                                  head_dims=self.head_dims,
                                                  ff_hid_dims=ff_hid_dims,
                                                  drop_rate=drop_rate,
                                                  self_causality=kwargs['self_causality'],
                                                  pad_idx=kwargs['pad_idx']) for _ in range(layer_num)])

    def forward(self, query,key,query_mask=None,key_mask=None):
        flag=0
        if query_mask is None:
            flag=1
            query_len = query.size(1)
            batch_max_query_len = query.abs().sum(-1).sign().sum(-1).max().int()
            query = query[:, :batch_max_query_len, :]
            query_mask = query.abs().sum(-1).sign()
        if key_mask is None:
            batch_max_key_len = key.abs().sum(-1).sign().sum(-1).max().int()
            key= key[:, :batch_max_key_len, :]
            key_mask = key.abs().sum(-1).sign()
        for i in range(self.layer_num):
            query=self.dec_blocks[i](query=query,key=key,query_mask=query_mask,key_mask=key_mask)
        if flag==1:
            query=query.mul(query_mask.unsqueeze(-1).expand(-1, -1, self.query_dims).float())
            query=F.pad(query,[0,0,0,query_len-batch_max_query_len,0,0])
        return query

class EncBlock(nn.Module):
    def __init__(self,
                 query_dims=512,
                 head_num=8,
                 head_dims=None,
                 ff_hid_dims=2048,
                 drop_rate=0.,
                 # causality=False,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.query_dims = query_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.res_att=ResMHA(query_dims=query_dims,
                            key_dims=query_dims,
                            head_num=head_num,
                            head_dims=head_dims,
                            drop_rate=drop_rate,
                            causality=False,
                            pad_idx=kwargs['pad_idx'])
        self.res_ff = ResFF(in_dims=query_dims,
                            hid_dims=ff_hid_dims,
                            drop_rate=drop_rate)
    def forward(self, query,query_mask):
        query=self.res_att(query=query, key=query,query_mask=query_mask,key_mask=query_mask)
        query=self.res_ff(query,mask=query_mask)
        return query

class DualDecBlock(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 ff_hid_dims=2048,
                 head_num=8,
                 head_dims=None,
                 drop_rate=0.,
                 mode='sequential',
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.query_dims = query_dims
        self.key_dims=query_dims if key_dims is None else key_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.mode=mode
        self.res_self_att=ResMHA(query_dims=query_dims,
                                 key_dims=query_dims,
                                 head_num=head_num,
                                 head_dims=head_dims,
                                 drop_rate=drop_rate,
                                 causality=True,
                                 pad_idx=kwargs['pad_idx'])
        if mode not in 'same-sub':
            self.res_cross_att1 = ResMHA(query_dims=query_dims,
                                        key_dims=key_dims,
                                        head_num=head_num,
                                        head_dims=head_dims,
                                        drop_rate=drop_rate,
                                        causality=False,
                                     pad_idx=kwargs['pad_idx'])
            self.res_cross_att2 = ResMHA(query_dims=query_dims,
                                            key_dims=key_dims,
                                            head_num=head_num,
                                            head_dims=head_dims,
                                            drop_rate=drop_rate,
                                            causality=False,
                                         pad_idx=kwargs['pad_idx'])
        else:
            self.attention = MultiHeadAttention(query_dims=self.query_dims,
                                                key_dims=self.key_dims,
                                                head_num=self.head_num,
                                                head_dims=self.head_dims,
                                                drop_rate=drop_rate,
                                                causality=False,
                                                pad_idx=kwargs['pad_idx'])
            self.layer_norm = nn.LayerNorm(self.query_dims, elementwise_affine=True)
        self.res_ff= ResFF(in_dims=query_dims,
                                    hid_dims=ff_hid_dims,
                                    drop_rate=drop_rate)
    def forward(self, query,key1,key2,query_mask,key_mask1,key_mask2):
        query=self.res_self_att(query=query, key=query,query_mask=query_mask,key_mask=query_mask)
        if self.mode=='sequential':
            query=self.res_cross_att1(query=query,key=key1,query_mask=query_mask,key_mask=key_mask1)
            query=self.res_cross_att2(query=query,key=key2,query_mask=query_mask,key_mask=key_mask2)
        elif self.mode in 'add':
            query1 = self.res_cross_att1(query=query, key=key1, query_mask=query_mask, key_mask=key_mask1)
            query2 = self.res_cross_att2(query=query, key=key2, query_mask=query_mask, key_mask=key_mask2)
            query=query1.add(query2)
        elif self.mode=='sub':
            query1 = self.res_cross_att1(query=query, key=key1, query_mask=query_mask, key_mask=key_mask1)
            query2 = self.res_cross_att2(query=query, key=key2, query_mask=query_mask, key_mask=key_mask2)
            query=query2.sub(query1)
        elif self.mode=='same-sub':
            query1 = self.attention(query=query, key=key1, query_mask=query_mask, key_mask=key_mask1)
            query2 = self.attention(query=query, key=key2, query_mask=query_mask, key_mask=key_mask2)
            query = self.layer_norm(query.add(query2.sub(query1)))

        query=self.res_ff(query,mask=query_mask)
        return query  

class DecBlock(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 ff_hid_dims=2048,
                 head_num=8,
                 head_dims=None,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        kwargs.setdefault('self_causality', True) 
        self.query_dims = query_dims
        self.key_dims=query_dims if key_dims is None else key_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.res_self_att=ResMHA(query_dims=query_dims,
                                 key_dims=query_dims,
                                 head_num=head_num,
                                 head_dims=head_dims,
                                 drop_rate=drop_rate,
                                 causality=kwargs['self_causality'],
                                 pad_idx=kwargs['pad_idx'])
        self.res_cross_att = ResMHA(query_dims=query_dims,
                                    key_dims=key_dims,
                                    head_num=head_num,
                                    head_dims=head_dims,
                                    drop_rate=drop_rate,
                                    causality=False,
                                    pad_idx=kwargs['pad_idx'])
        self.res_ff= ResFF(in_dims=query_dims,
                           hid_dims=ff_hid_dims,
                           drop_rate=drop_rate)
    def forward(self, query,key,query_mask,key_mask):
        query=self.res_self_att(query=query, key=query,query_mask=query_mask,key_mask=query_mask)
        query=self.res_cross_att(query=query,key=key,query_mask=query_mask,key_mask=key_mask)
        query=self.res_ff(query,mask=query_mask)
        return query  

class ResMHA(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 head_num=8,
                 head_dims=None,
                 drop_rate=0.,
                 causality=False,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.query_dims = query_dims
        self.key_dims=query_dims if key_dims is None else key_dims
        self.head_num = head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.attention=MultiHeadAttention(query_dims=self.query_dims,
                                          key_dims=self.key_dims,
                                          head_num=self.head_num,
                                          head_dims=self.head_dims,
                                          drop_rate=drop_rate,
                                          causality=causality,
                                          pad_idx=kwargs['pad_idx'])
        self.layer_norm =nn.LayerNorm(self.query_dims, elementwise_affine=True)


    def forward(self, query,key,query_mask,key_mask):
        query_ = self.attention(query=query, key=key,query_mask=query_mask,key_mask=key_mask)
        query=self.layer_norm(query_.add(query))
        return query  

class ResFF(nn.Module):
    def __init__(self,
                 in_dims=512,
                 hid_dims=2048,
                 drop_rate=0.):
        super().__init__()
        self.feedforward = FeedForward(in_dims=in_dims,
                                    hid_dims=hid_dims,
                                    out_dims=in_dims,
                                    drop_rate=drop_rate)
        self.layer_norm = nn.LayerNorm(in_dims, elementwise_affine=True)
    def forward(self, x,mask=None):
        x_ = self.feedforward(x, mask=mask)
        x = self.layer_norm(x.add(x_))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 head_num=8,
                 head_dims=None,
                 drop_rate=0.,
                 causality=False,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.query_dims = query_dims
        self.key_dims = query_dims if key_dims is None else key_dims
        self.head_num=head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.hid_dims=self.head_num*self.head_dims

        self.causality=causality
        self.conv1d_ins=nn.ModuleList([nn.Conv1d(io_dims, self.hid_dims, kernel_size=1,padding=0)
                                    for io_dims in [self.query_dims,self.key_dims,self.key_dims]])

        self.conv1d_out=nn.Conv1d(self.hid_dims,self.query_dims, kernel_size=1,padding=0)

        self.softmax=nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(drop_rate)

    def forward(self, query,key,query_mask,key_mask,value=None):
        if value is None:
            value=key.clone()   
        batch_size=query.size(0)   

        query_,key_,value_=[conv1d_in(x.transpose(1,2)) for conv1d_in,x in
                            zip(self.conv1d_ins,(query,key,value))] 
        query_, key_, value_ = [x.view(batch_size, self.head_num, self.head_dims, -1).transpose(2, 3)
                                    for x in (query_, key_, value_)] 

        query_=query_.mul(float(self.head_dims)**-0.5)        
        attention=torch.einsum('abcd,abed->abce',query_,key_)  
        if key_mask is not None:
            key_mask=key_mask.eq(self.pad_idx)
            key_mask=key_mask.unsqueeze(dim=1).repeat(1,self.head_num,1)  
            key_mask=key_mask.unsqueeze(dim=2).expand(-1,-1,query.size(1),-1)    
            attention=attention.masked_fill(key_mask,-np.inf)   

        if self.causality:
            seq_mask=torch.triu(torch.ones_like(attention[0,0,:,:]),diagonal=1).float()
            seq_mask = seq_mask.masked_fill(seq_mask == 1, float('-inf'))
            seq_mask=seq_mask[None,None,:,:].expand(batch_size,self.head_num,-1,-1) 

            attention=attention.add(seq_mask)

        attention = self.softmax(attention)  

        attention=self.dropout(attention)   

        output=torch.matmul(attention,value_) 
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.hid_dims)    
        output=self.conv1d_out(output.transpose(1,2)).transpose(1,2)  

        if query_mask is not None:
            query_mask=query_mask[:,:,None].expand(-1,-1,self.query_dims)  
            output=output.mul(query_mask.float()) 
        return output

class FeedForward(nn.Module):
    def __init__(self,
                 in_dims=512,
                 hid_dims=2048,
                 out_dims=None,
                 drop_rate=0.
                 ):
        super().__init__()
        out_dims=in_dims if out_dims is None else out_dims
        self.linear_in=nn.Linear(in_dims,hid_dims)
        self.relu=nn.ReLU()
        self.leaky_relu=nn.LeakyReLU()
        self.linear_out=nn.Linear(hid_dims,out_dims)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x,mask=None):
        output = self.linear_in(x) 
        output = self.leaky_relu(output) 
        output = self.linear_out(output)  
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))  
            output=output.mul(mask.float()) 

        output=self.dropout(output) 
        return output
