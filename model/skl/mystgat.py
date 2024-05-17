import copy as cp
import torch
import torch.nn as nn

from model.skl.utils import Graph
from model.skl.utils import mstcn, unit_gcn, unit_tcn
from model.heads import GCNHead
from builder.builder import ModelRegistry
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

EPS = 1e-4

import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.Wlinear = nn.Linear(in_feature, out_feature)
        # self.W=nn.Parameter(torch.empty(size=(batch_size,in_feature,out_feature)))
        nn.init.xavier_uniform_(self.Wlinear.weight, gain=1.414)

        self.aiLinear = nn.Linear(out_feature, 1)
        self.ajLinear = nn.Linear(out_feature, 1)
        # self.a=nn.Parameter(torch.empty(size=(batch_size,2*out_feature,1)))
        nn.init.xavier_uniform_(self.aiLinear.weight, gain=1.414)
        nn.init.xavier_uniform_(self.ajLinear.weight, gain=1.414)

        self.leakyRelu = nn.LeakyReLU(self.alpha)

    def getAttentionE(self, Wh):
        # 重点改了这个函数
        Wh1 = self.aiLinear(Wh)
        Wh2 = self.ajLinear(Wh)
        Wh2 = Wh2.view(Wh2.shape[0], Wh2.shape[2], Wh2.shape[1])
        # Wh1=torch.bmm(Wh,self.a[:,:self.out_feature,:])    #Wh:size(node,out_feature),a[:out_eature,:]:size(out_feature,1) => Wh1:size(node,1)
        # Wh2=torch.bmm(Wh,self.a[:,self.out_feature:,:])    #Wh:size(node,out_feature),a[out_eature:,:]:size(out_feature,1) => Wh2:size(node,1)

        e = Wh1 + Wh2  # broadcast add, => e:size(node,node)
        return self.leakyRelu(e)

    def forward(self, h, adj):
        # print(h.shape)
        B, S, T, V = h.shape
        h = h.view(B, S * T, V)
        Wh = self.Wlinear(h)
        # Wh=torch.bmm(h,self.W)   #h:size(node,in_feature),W:size(in_feature,out_feature) => Wh:size(node,out_feature)
        e = self.getAttentionE(Wh)

        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_hat = torch.bmm(attention, Wh)
        # attention:size(node,node),Wh:size(node,out_feature) => h_hat:size(node,out_feature)

        if self.concat:
            return F.elu(h_hat)
        else:
            return h_hat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_feature) + '->' + str(self.out_feature) + ')'


class GAT(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature, attention_layers, dropout, alpha):
        super(GAT, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_feature = hidden_feature
        self.dropout = dropout
        self.alpha = alpha
        self.attention_layers = attention_layers

        self.attentions = [GraphAttentionLayer(in_feature, hidden_feature, dropout, alpha, True) for i in
                           range(attention_layers)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_attention = GraphAttentionLayer(attention_layers * hidden_feature, out_feature, dropout, alpha, False)

    def forward(self, h, adj):
        # print(h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = torch.cat([attention(h, adj) for attention in self.attentions], dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.elu(self.out_attention(h, adj))
        return h


class STGATBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()
        gat_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gat_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gat_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gat_type = gat_kwargs.pop('type', 'unit_gat')
        assert gat_type in ['unit_gat']

        # self.gat = unit_gat(in_channels, out_channels, A, **gat_kwargs)
        # self.gat = GATConv(in_channels, out_channels, heads=8, dropout=0.6, **gat_kwargs)
        self.gat = GraphAttentionLayer(A.size * in_channels, out_channels, dropout=0.6, alpha=0.01, concat=True)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gat(x, A)) + res
        return self.relu(x)


# @ModelRegistry.register_module()
class STGAT(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.graph_cfg = graph_cfg
        self.graph = Graph(**graph_cfg)
        # A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False))
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGATBlock(in_channels, base_channels, self.graph.A, 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGATBlock(in_channels, out_channels, self.graph.A, stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gat = nn.ModuleList(modules)
        self.pretrained = pretrained
        self.A_GAT = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])

    def init_weights(self):
        pass
        # if isinstance(self.pretrained, str):
        # self.pretrained = cache_checkpoint(self.pretrained)
        # load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):

        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gat[i](x, self.graph.A)
            # x = self.gat[i](x, self.A_GAT)

        x = x.reshape((N, M) + x.shape[1:])

        return x
