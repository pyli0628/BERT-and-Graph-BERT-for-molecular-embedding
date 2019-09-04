import torch
import torch.nn as nn

class Readout(nn.Module):
    def __init__(self, bert_hs, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(bert_hs,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,1)
    def forward(self,x):
        x = self.relu(self.fc1(x[:,0]))
        x = self.fc2(x)
        return x

class bert_fc(nn.Module):
    def __init__(self, bert_path, hidden_size):
        super().__init__()
        self.bert = torch.load(bert_path)
        self.bert_hs = self.bert.hidden
        self.readout = Readout(self.bert_hs, hidden_size)
    def forward(self,x,seg_label):
        return self.readout(self.bert(x,seg_label))
