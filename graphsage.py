import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math

class NewConvolution(nn.Module):
    """
    A Graph Convolution Layer for GraphSage
    """
    def __init__(self, in_features, out_features, bias=True):
        super(NewConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_1 = nn.Linear(in_features, out_features, bias=bias)
        self.W_2 = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W_1.weight.size(1))
        self.W_1.weight.data.uniform_(-stdv, stdv)
        self.W_2.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support_1 = self.W_1(input)
        support_2 = self.W_2(input)
        output = torch.mm(adj, support_2)
        output = output + support_1
        return output

class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage, self).__init__()
        self.gc1 = NewConvolution(nfeat, nhid)
        self.gc2 = NewConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

