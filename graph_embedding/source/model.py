import torch
import torch.nn as nn
import torch.nn.functional as F



class MolGraph(nn.Module):
    def __init__(self,gat,hidden,vocab_size):
        super().__init__()
        self.gat = gat
        self.out = MaskPredict(hidden,vocab_size)
    def forward(self, x,adj):
        x = self.gat(x,adj)
        return self.out(x)

class MolGraph2(nn.Module):
    def __init__(self,gat,hidden,vocab_size):
        super().__init__()
        self.gat = gat
        self.out1 = MaskPredict(hidden,vocab_size)
        self.out2 = NextPredict(hidden)
    def forward(self, x1,x2,adj1,adj2):
        x1 = self.gat(x1,adj1)
        x2 = self.gat(x2,adj2)
        x = torch.cat([torch.sum(x1,1),torch.sum(x2,1)],dim=1)
        return self.out1(x1),self.out1(x2),self.out2(x)


class NextPredict(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.linear = nn.Linear(2*hidden, hidden)
        self.linear1 = nn.Linear(hidden,2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear1(self.linear(x)))


class MaskPredict(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden,hidden)
        self.linear1 = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(self.linear(x)))


class GAT(nn.Module):
    def __init__(self,nhid,nheads,vocab_size,layers,dropout=0.1,alpha=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, nhid,padding_idx=0)
        self.gat = nn.ModuleList(
            [GATlayer(nhid,nheads,dropout,alpha) for _ in range(layers)]
        )
    def forward(self, x,adj):
        x = self.embedding(x)
        for gnn in self.gat:
            x = gnn(x,adj)
        return x

class GATlayer(nn.Module):
    def __init__(self, nhid, nheads,  dropout=0.1, alpha=0.2):
        """Dense version of GAT."""
        super(GATlayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.attentions = nn.ModuleList([GraphAttentionLayer(nhid, int(nhid/nheads), dropout=dropout,
            alpha=alpha, concat=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        return self.dropout(x)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # bs,N,d; bs,N,N
        # print(input.get_device(),self.W.get_device())
        h = torch.matmul(input, self.W)  # bs,N,d1
        N = h.size()[1]
        batch_size = h.size()[0]

        # bs,N*N,2*out
        a_input = torch.cat([h.repeat(1, N, 1), h.repeat(1, 1, N).view(batch_size, N * N, -1)], dim=2)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)).view(batch_size, N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # bs,N,N
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # bs,N,d
        if self.concat:
            return F.elu(h_prime)  # bs,N,N
        else:
            return h_prime  # bs,N,N

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


