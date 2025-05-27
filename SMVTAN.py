# -*- coding:utf-8 -*-
import copy
import torch
import torch.nn as nn
from torch import Tensor
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Conv2d


class glu(nn.Module):
    def __init__(self, num_of_features, bias=True):
        super(glu, self).__init__()
        self.bias = bias
        self.num_of_features = num_of_features
        self.weight = nn.Parameter(Tensor(num_of_features, 2 * num_of_features))
        if bias == True:
            self.bias = nn.Parameter(Tensor(1, 2 * num_of_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)

    def forward(self, data):
        data = torch.matmul(data, self.weight) + self.bias
        lhs, rhs = torch.split(data, self.num_of_features, dim=-1)
        return lhs * torch.sigmoid(rhs)

class fc(nn.Module):
    def __init__(self, num_of_input, num_of_output, bias=True):
        super(fc, self).__init__()
        self.bias = bias
        self.weight = nn.Parameter(Tensor(num_of_input, num_of_output))
        if bias == True:
            self.bias = nn.Parameter(Tensor(1, num_of_output))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight, gain=0.0003)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias)

    def forward(self, data):
        return torch.matmul(data, self.weight) + self.bias

class mf(nn.Module):
    def __init__(self, merge_step, num_of_vertices, num_of_latents, dropout_rate, adj_st, cand=True):
        super(mf, self).__init__()

        self.num_of_vertices = num_of_vertices
        self.node_emb = nn.Parameter(Tensor(merge_step * num_of_vertices, num_of_latents))
        if cand == False:
            TI = []
            for i in range(merge_step):
                TI.append(torch.eye(num_of_vertices))
            self.TI = torch.stack(TI, dim=1).reshape(num_of_vertices, -1).cuda()
        elif merge_step == 1:
            self.TI = adj_st[:, -num_of_vertices:]
        else: 
            self.TI = adj_st

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.node_emb, gain=0.02)

    def forward(self):
        adj = torch.mm(self.node_emb[-self.num_of_vertices:], self.node_emb.transpose(1, 0)) + self.TI

        adj = self.dropout(self.softmax(adj))

        return adj


class position_embedding(nn.Module):
    def __init__(self,
                 input_length,
                 num_of_vertices,
                 embedding_size,
                 temporal=True,
                 spatial=True,
                 embedding_type='add'):
        super(position_embedding, self).__init__()
        self.embedding_type = embedding_type

        if temporal:
            self.temporal_emb = nn.Parameter(Tensor(1, input_length, 1, embedding_size))

        if spatial:
            self.spatial_emb = nn.Parameter(Tensor(1, 1, num_of_vertices, embedding_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.temporal_emb, gain=0.0003)
        torch.nn.init.xavier_normal_(self.spatial_emb, gain=0.0003)

    def forward(self, data):
        if self.embedding_type == 'add':
            if self.temporal_emb is not None:
                data = data + self.temporal_emb

            if self.spatial_emb is not None:
                data = data + self.spatial_emb

        elif self.embedding_type == 'multiply':
            if self.temporal_emb is not None:
                data = data * self.temporal_emb
            if self.spatial_emb is not None:
                data = data * self.spatial_emb

        return data

class output_layer(nn.Module):
    def __init__(self,
                 num_of_vertices,
                 input_length,
                 num_of_features,
                 predict_length,
                 num_of_hidden=128,
                 num_of_output=1):
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.predict_length = predict_length

        self.weight1 = nn.Parameter(Tensor(predict_length, input_length * num_of_features, num_of_hidden))
        self.bias1 = nn.Parameter(Tensor(predict_length, 1, num_of_hidden))
        self.weight2 = nn.Parameter(Tensor(predict_length, num_of_hidden, num_of_output))
        self.bias2 = nn.Parameter(Tensor(predict_length, 1, num_of_output))

        self.pos_emb = position_embedding(predict_length, num_of_vertices, num_of_hidden)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight1, gain=0.0003)
        torch.nn.init.normal_(self.bias1)
        torch.nn.init.xavier_normal_(self.weight2, gain=0.0003)
        torch.nn.init.normal_(self.bias2)

    def forward(self, data):
        data = torch.swapaxes(data, 1, 2)

        data = torch.reshape(data, (-1, self.num_of_vertices, self.input_length * self.num_of_features))
        data = torch.relu(torch.einsum('bnc, tcd->bntd', data, self.weight1))

        data = torch.swapaxes(data, 1, 2)
        data = self.pos_emb(data)

        data = torch.einsum('btnd, tdc->btnc', data, self.weight2)

        return data

class gcn_operation(nn.Module):
    def __init__(self,
                 num_of_features,
                 num_of_hidden,
                 num_of_vertices):
        super(gcn_operation, self).__init__()

        self.glu_layer = glu(num_of_features)

    def forward(self, wing, data, adj):
        data = torch.cat((wing, data), dim=0)

        data = torch.einsum('nm, mbc->nbc', adj, data)

        data = self.glu_layer(data)

        return data


class stack_gcn(nn.Module):
    def __init__(self, num_of_features, num_of_vertices, num_of_kernel, merge_step):
        super(stack_gcn, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.merge_step = merge_step
        self.gcn_operations = nn.ModuleList([
            gcn_operation(num_of_features,
                          num_of_features,
                          num_of_vertices)
            for i in range(num_of_kernel)])

    def forward(self, data, adj):

        need_concat = []

        wing = data[0: (self.merge_step - 1) * self.num_of_vertices]

        data = data[-self.num_of_vertices:]

        res = data[-self.num_of_vertices:]

        for i, gcn in enumerate(self.gcn_operations):

            data = gcn(wing, data, adj)

            need_concat.append(data)

        need_concat = torch.stack(need_concat, dim=0)
        return torch.max(need_concat, dim=0).values + res

class fstgcn(nn.Module):
    def __init__(self,
                 i,
                 current_step,
                 num_of_vertices,
                 num_of_features,
                 merge_step,
                 num_of_kernel,
                 temporal_emb=True,
                 spatial_emb=True):
        super(fstgcn, self).__init__()
        self.current_step = current_step
        self.merge_step = merge_step
        self.conv1 = nn.Conv2d(in_channels=num_of_features,
                               out_channels=num_of_features * 2,
                               kernel_size=(1, merge_step),
                               stride=(1, 1))

        if self.current_step != 12:
            self.conv2 = nn.Conv2d(in_channels=num_of_features,
                                   out_channels=num_of_features * 2,
                                   kernel_size=(1, merge_step + (merge_step - 1) * i),
                                   stride=(1, 1))

        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features

        self.position_embedding = position_embedding(current_step,
                                                     num_of_vertices,
                                                     num_of_features,
                                                     temporal_emb,
                                                     spatial_emb)

        self.stack_gcns = nn.ModuleList([stack_gcn(num_of_features,
                                                   num_of_vertices,
                                                   num_of_kernel,
                                                   merge_step)
                                         for i in range(current_step - (merge_step - 1))])

    def forward(self, original_data, data, adj_st):
        temp = torch.permute(data, (0, 3, 2, 1))
        temp = self.conv1(temp)
        data_left, data_right = torch.split(temp, int(temp.shape[1] / 2), dim=1)
        data_time_axis = torch.sigmoid(data_left) * data_right

        if self.current_step != 12:
            temp = torch.permute(original_data, (0, 3, 2, 1))
            temp = self.conv2(temp)
            data_left, data_right = torch.split(temp, int(temp.shape[1] / 2), dim=1)
            data_time_axis += torch.sigmoid(data_left) * data_right

        data_res = torch.permute(data_time_axis, (0, 3, 2, 1))
        data = self.position_embedding(data)

        need_concat = []
        for i, layer in enumerate(self.stack_gcns):
            t = data[:, i: i + self.merge_step]

            t = torch.reshape(t, (-1, self.merge_step * self.num_of_vertices, self.num_of_features))

            t = torch.permute(t, (1, 0, 2))

            t = layer(t, adj_st)

            t = torch.swapaxes(t, 0, 1)

            need_concat.append(t)

        need_concat = torch.stack(need_concat, dim=1)

        data = need_concat + data_res

        return data

class s2t(nn.Module):
    def __init__(self, num_of_features, input_length, num_of_vertices):
        super(s2t, self).__init__()
        self.position_embedding = position_embedding(input_length,
                                                     num_of_vertices,
                                                     num_of_features)
        self.glu_layer = glu(num_of_features)

    def forward(self, data, temp, adj_s, adj_t):
        res = data
        data = self.position_embedding(data)
        data = torch.einsum('nn, btnc -> btnc', adj_s, data)
        data = torch.einsum('tt, btnc -> btnc', adj_t, data)
        data = self.glu_layer(data)

        return data + res + temp

class t2s(nn.Module):
    def __init__(self, num_of_features, input_length, num_of_vertices):
        super(t2s, self).__init__()
        self.position_embedding = position_embedding(input_length,
                                                     num_of_vertices,
                                                     num_of_features)
        self.glu_layer = glu(num_of_features)

    def forward(self, data, temp, adj_s, adj_t):
        res = data
        data = self.position_embedding(data)
        data = torch.einsum('tt, btnc -> btnc', adj_t, data)
        data = torch.einsum('ss, btnc -> btnc', adj_s, data)
        data = self.glu_layer(data)

        return data + res + temp

class PositionalEncoding(nn.Module):
    def __init__(self, out_dim, max_len=12):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, out_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_dim, 2) *
                             - math.log(10000.0) / out_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe.to(x.device), requires_grad=False)
        return x

class Gate(nn.Module):
    def __init__(self, n_out):
        super(Gate, self).__init__()
        self.n_out = n_out
        self.W_z = nn.Parameter(torch.empty(size=(2 * n_out, n_out)))
        nn.init.xavier_uniform_(self.W_z.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(1, n_out)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, x, h):
        x_h = torch.cat((x, h), dim=-1)
        Wh = torch.matmul(x_h, self.W_z)
        gate = torch.sigmoid(Wh + self.b)
        one_vec = torch.ones_like(gate)
        z = gate * x + (one_vec - gate) * h
        return z
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.padding = (kernel_size - 1)//2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)

        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)

        N = query.size(1)
        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous()
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class make_model(nn.Module):
    def __init__(self, adj_st, input_length, num_of_vertices, d_model, filter_list, use_mask,
                 mask_init_value_st, temporal_emb, spatial_emb, predict_length, num_of_features, receptive_length,
                 ST_adj_dropout_rate, num_of_latents, num_of_gcn_filters, x2=True, x3=True):

        super(make_model, self).__init__()
        self.x2 = x2
        self.x3 = x3
        self.adj_st = adj_st
        self.d_model = d_model
        self.predict_length = predict_length
        self.num_of_vertices = num_of_vertices
        self.filter_list = filter_list
        self.input_length = input_length
        self.stack_times = int((input_length - 1) / (receptive_length - 1))
        self.reduce_length = receptive_length - 1
        self.final_length = input_length - self.stack_times * self.reduce_length
        S_adj_dropout_rate = 1 - (1 - ST_adj_dropout_rate) * receptive_length
        self.I = torch.eye(input_length).cuda()

        self.fc = fc(num_of_features, d_model)
        self.mf = mf(receptive_length, num_of_vertices, num_of_latents, ST_adj_dropout_rate, adj_st)
        if x2:
            self.mf_s2 = mf(1, num_of_vertices, num_of_latents, S_adj_dropout_rate, adj_st[:, -num_of_vertices:])
        if x3:
            self.mf_s3 = mf(1, num_of_vertices, num_of_latents, S_adj_dropout_rate, adj_st[:, -num_of_vertices:])
        if x2:
            self.mf_t2 = nn.Parameter(Tensor(input_length, input_length))
        if x3:
            self.mf_t3 = nn.Parameter(Tensor(input_length, input_length))

        self.fstgcns = nn.ModuleList([fstgcn(i, input_length - (receptive_length - 1) * i, num_of_vertices,
                                             d_model, receptive_length, num_of_gcn_filters, temporal_emb=temporal_emb,
                                             spatial_emb=spatial_emb) for i in range(self.stack_times)])
        if x2:
            self.s2t = nn.ModuleList([
            s2t(d_model, input_length, num_of_vertices)
            for i in range(self.stack_times)
        ])
            self.conv2 = nn.Conv2d(in_channels=input_length,
                               out_channels=3,
                               kernel_size=(1, 1))
        if x3:
            self.t2s = nn.ModuleList([
            t2s(d_model, input_length, num_of_vertices)
            for i in range(self.stack_times)
        ])
            self.conv3 = nn.Conv2d(in_channels=input_length,
                               out_channels=3,
                               kernel_size=(1, 1))

        self.output_layer = output_layer(num_of_vertices,
                                         self.final_length,
                                         d_model,
                                         predict_length)

        self.reset_parameters()
        self.end_conv = nn.Conv2d(1, 64, kernel_size=(1, 1), bias=True)
        self.end_conv3 = nn.Conv2d(64, 1, kernel_size=(1, 1), bias=True)

        self.self_attn = MultiHeadAttentionAwareTemporalContex_q1d_k1d(8, 64, 0, 0, 1, 12, 3,
                                                                       dropout=0.0)
    def reset_parameters(self) -> None:
        if self.x2:
            torch.nn.init.xavier_normal_(self.mf_t2, gain=0.0003)
        if self.x3:
            torch.nn.init.xavier_normal_(self.mf_t3, gain=0.0003)

    def forward(self, data):
        data = self.fc(data)
        if self.x2:
            adj_s2 = self.mf_s2()
            adj_t2 = self.mf_t2 + self.I
        if self.x3:
            adj_s3 = self.mf_s3()
            adj_t3 = self.mf_t3 + self.I
        adj = self.mf()

        temp, x1, x2, x3 = data, data, data, data
        for i in range(self.stack_times):
            x1 = self.fstgcns[i](temp, x1, adj)
        if self.x2:
            x2 = self.s2t[i](x2, temp, adj_s2, adj_t2)
        if self.x3:
            x3 = self.t2s[i](x3, temp, adj_s3, adj_t3)
        if self.x2:
            x2 = self.conv2(x2)
        if self.x3:
            x3 = self.conv3(x3)

        if self.x2 == True and self.x3 == False:
            data = x1 + x2
        elif self.x3 == True and self.x2 == False:
            data = x1 + x3
        else:
            data = x1 + x2 + x3
        x = data.transpose(1, 2)
        x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
        data = x.transpose(1, 2)
        data = self.output_layer(data)
        data = data.squeeze(dim=3)

        return data




