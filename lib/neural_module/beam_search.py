import numpy as np
import torch
import torch.nn.functional as F

def trans_beam_search(net,beam_width,length_penalty=1,dec_input_arg_name='dec_input',begin_idx=None,end_idx=None,pad_idx=0,**net_args,):
    assert dec_input_arg_name in net_args.keys()
    batch_size=net_args[dec_input_arg_name].size(0)
    max_len=net_args[dec_input_arg_name].size(1)

    advance_out = torch.ones_like(net_args[dec_input_arg_name])*pad_idx
    advance_mean_prob = torch.zeros_like(advance_out[:,0]).float()

    pred_out = net(**net_args)[:, :, 0]
    pred_out[:,begin_idx]=-np.inf
    pred_out[:,end_idx] = -np.inf
    pred_out[:,pad_idx]=-np.inf
    if begin_idx is None and end_idx is None:
        begin_idx=pred_out.size(1)-1
        end_idx=pred_out.size(1)-2
    pred_out = F.softmax(pred_out, dim=1)
    pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)
    acc_prob = pred_out.clone()
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].unsqueeze(1).expand(-1,beam_width,-1).clone()
    net_args[dec_input_arg_name][:, :, 1] = pred_out_ids
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)

    for key in net_args.keys():
        if key!=dec_input_arg_name:
            size=list(net_args[key].size())
            size.insert(1,beam_width)
            net_args[key]=net_args[key].unsqueeze(1).expand(size)
            size.pop(0)
            size[0]=batch_size*beam_width
            net_args[key]=net_args[key].reshape(size)

    for i in range(1,max_len):
        pred_out = net(**net_args)[:, :, i]
        pred_out[:, begin_idx] = -np.inf
        pred_out[:, pad_idx] = -np.inf
        pred_out = F.softmax(pred_out, dim=1)
        pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)
        pred_out = pred_out.view(-1, beam_width * beam_width)
        pred_out_ids = pred_out_ids.view(-1, beam_width * beam_width)

        net_args[dec_input_arg_name] = net_args[dec_input_arg_name].unsqueeze(1).expand(-1, beam_width, -1). \
            reshape(-1, beam_width * beam_width, max_len)
        acc_prob=acc_prob.unsqueeze(2).expand(-1,-1,beam_width).reshape(-1,beam_width*beam_width)
        acc_prob=acc_prob.add(pred_out)
        mean_prob=acc_prob/(i+1)**length_penalty

        if i<max_len-1:
            mean_prob, topk_ids = mean_prob.topk(beam_width, largest=True, dim=1)
            pred_out_id_list,dec_input_list,acc_prob_list=[],[],[]
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j,:].index_select(dim=0,index=topk_ids[j,:])
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :])
                j_acc_prob=acc_prob[j,:].index_select(dim=0,index=topk_ids[j,:])
                for k in range(beam_width):
                    if j_pred_out_ids[k].item()==end_idx:
                        j_acc_prob[k]=-np.inf
                        if mean_prob[j,k]>advance_mean_prob[j]:
                            advance_mean_prob[j]=mean_prob[j,k]
                            advance_out[j,:-1]=j_dec_input[k,1:]
                            advance_out[j,i]=j_pred_out_ids[k]
                pred_out_id_list.append(j_pred_out_ids.unsqueeze(0))
                dec_input_list.append(j_dec_input.unsqueeze(0))
                acc_prob_list.append(j_acc_prob.unsqueeze(0))

            pred_out_ids=torch.cat(pred_out_id_list,dim=0)
            net_args[dec_input_arg_name]=torch.cat(dec_input_list,dim=0)
            acc_prob=torch.cat(acc_prob_list,dim=0)
            net_args[dec_input_arg_name][:,:,i + 1] = pred_out_ids
            net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)
        if i==max_len-1:
            mean_prob, topk_ids = mean_prob.topk(1, largest=True, dim=1)
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j, :].index_select(dim=0, index=topk_ids[j, :])
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :])
                if mean_prob[j,0]>advance_mean_prob[j] or advance_mean_prob[j].item()==0.:
                    advance_out[j, :-1] = j_dec_input[0, 1:]
                    advance_out[j, -1] = j_pred_out_ids[0]
            return advance_out #(Batch,max_Len)


def rnn_beam_search(net,beam_width,length_penalty=1,
                    dec_input_arg_name='dec_input',dec_hid_arg_name='dec_hid',
                    begin_idx=None,end_idx=None,pad_idx=0,**net_args,):
    assert dec_input_arg_name in net_args.keys()
    batch_size=net_args[dec_input_arg_name].size(0)
    max_len=net_args[dec_input_arg_name].size(1)

    advance_out = torch.ones_like(net_args[dec_input_arg_name])*pad_idx
    advance_mean_prob = torch.zeros_like(advance_out[:,0]).float()

    pred_out,net_args[dec_hid_arg_name] = net(**net_args)
    pred_out[:,begin_idx]=-np.inf
    pred_out[:,end_idx] = -np.inf
    pred_out[:,pad_idx]=-np.inf
    if begin_idx is None and end_idx is None:
        begin_idx=pred_out.size(1)-1
        end_idx=pred_out.size(1)-2
    pred_out = F.softmax(pred_out, dim=1)
    pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)  
    acc_prob = pred_out.clone() 
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].unsqueeze(1).expand(-1,beam_width,-1).clone() 
    net_args[dec_input_arg_name][:, :, 1] = pred_out_ids
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)

    for key in net_args.keys():
        if key!=dec_input_arg_name:
            size=list(net_args[key].size())
            size.insert(1,beam_width)
            net_args[key]=net_args[key].unsqueeze(1).expand(size)
            size.pop(0)
            size[0]=batch_size*beam_width
            net_args[key]=net_args[key].reshape(size)

    for i in range(1,max_len):
        pred_out,net_args[dec_hid_arg_name] = net(**net_args)
        pred_out[:, begin_idx] = -np.inf
        pred_out[:, pad_idx] = -np.inf
        pred_out = F.softmax(pred_out, dim=1)
        pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)
        pred_out = pred_out.view(-1, beam_width * beam_width)
        pred_out_ids = pred_out_ids.view(-1, beam_width * beam_width)

        net_args[dec_input_arg_name] = net_args[dec_input_arg_name].unsqueeze(1).expand(-1, beam_width, -1). \
            reshape(-1, beam_width * beam_width, max_len)
        acc_prob=acc_prob.unsqueeze(2).expand(-1,-1,beam_width).reshape(-1,beam_width*beam_width)
        acc_prob=acc_prob.add(pred_out)
        mean_prob=acc_prob/(i+1)**length_penalty

        if i<max_len-1:
            mean_prob, topk_ids = mean_prob.topk(beam_width, largest=True, dim=1)
            pred_out_id_list,dec_input_list,acc_prob_list=[],[],[]
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j,:].index_select(dim=0,index=topk_ids[j,:])
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :])
                j_acc_prob=acc_prob[j,:].index_select(dim=0,index=topk_ids[j,:]) 
                for k in range(beam_width):
                    if j_pred_out_ids[k].item()==end_idx:
                        j_acc_prob[k]=-np.inf
                        if mean_prob[j,k]>advance_mean_prob[j]:
                            advance_mean_prob[j]=mean_prob[j,k]
                            advance_out[j,:-1]=j_dec_input[k,1:]
                            advance_out[j,i]=j_pred_out_ids[k]
                pred_out_id_list.append(j_pred_out_ids.unsqueeze(0))
                dec_input_list.append(j_dec_input.unsqueeze(0))
                acc_prob_list.append(j_acc_prob.unsqueeze(0))

            pred_out_ids=torch.cat(pred_out_id_list,dim=0)
            net_args[dec_input_arg_name]=torch.cat(dec_input_list,dim=0)
            acc_prob=torch.cat(acc_prob_list,dim=0)
            net_args[dec_input_arg_name][:,:,i + 1] = pred_out_ids
            net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)
        if i==max_len-1:
            mean_prob, topk_ids = mean_prob.topk(1, largest=True, dim=1)
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j, :].index_select(dim=0, index=topk_ids[j, :])
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :])
                if mean_prob[j,0]>advance_mean_prob[j] or advance_mean_prob[j].item()==0.:
                    advance_out[j, :-1] = j_dec_input[0, 1:]
                    advance_out[j, -1] = j_pred_out_ids[0]
            return advance_out