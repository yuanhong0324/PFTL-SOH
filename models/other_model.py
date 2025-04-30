import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self,feature_dim=32):
        super(MLP, self).__init__()
        self.layer0 = nn.Linear(feature_dim, 64 * 4)

        self.net = nn.Sequential(
            nn.Linear(64 * 4, 64 * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        '''
        :param x: (N,4,128)
        :return:
        '''
        x = self.layer0(x)
        x = x.view(-1, 4 * 64)
        out = self.net(x)
        return out,None,None

class LSTM(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self,feature_dim=32):
        super(LSTM, self).__init__()
        self.net = nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.layer0 = nn.Linear(feature_dim, 64 * 4)
        print("已加载LSTM模型")

    def forward(self, x):
        '''
        :param x: (N,4,128)
        :return:
        '''

        x = self.layer0(x)
        x = x.view(-1, 4, 64)
        x = x.transpose(1, 2)
        embed, (_, _) = self.net(x)
        out = embed[:, -1, :]
        pred = self.predictor(out)
        return pred,None,None

class GRU(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self,feature_dim=32):
        super(GRU, self).__init__()
        self.net = nn.GRU(input_size=4, hidden_size=64, num_layers=4, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.layer0 = nn.Linear(feature_dim, 64 * 4)
        print('已加载GRU模型')

    def forward(self, x):
        '''
        :param x: (N,4,128)
        :return:
        '''
        x = self.layer0(x)
        x = x.view(-1, 4, 64)
        x = x.transpose(1, 2)
        embed, _ = self.net(x)
        out = embed[:, -1, :]
        pred = self.predictor(out)
        return pred,None,None

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_pool=False):
        super(CBR, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.is_pool = is_pool
        if is_pool:
            self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.is_pool:
            x = self.pool(x)
        x = self.relu(x)
        return x

class CNN(nn.Module):
    '''
    input shape: (N,4,128)
    '''

    def __init__(self,feature_dim=32):
        super(CNN, self).__init__()
        self.layer0 = nn.Linear(feature_dim, 64 * 4)
        self.net = nn.Sequential(
            CBR(4, 16, 3, 1, 1,is_pool=True),
            CBR(16, 32, 3, 1, 1,is_pool=True),
            CBR(32, 64, 3, 1, 1,is_pool=True),
            CBR(64, 128, 3, 1, 1,is_pool=True),
            nn.Flatten(),
            nn.Linear(128 * 4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        '''
        :param x: (N,4,128)
        :return:
        '''
        x = self.layer0(x)
        x = x.view(-1, 4, 64)
        out = self.net(x)
        return out,None,None

class RNN(nn.Module):
    def __init__(self, feature_dim=32):
        super(RNN, self).__init__()
        self.layer0 = nn.Linear(feature_dim, 64 * 4)
        self.hidden_size = 64
        self.num_layers = 3
        self.net = nn.RNN(input_size=4, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = x.view(-1, 4, 64)
        x = x.transpose(1, 2)
        embed, _ = self.net(x)
        out = embed[:, -1, :]
        pred = self.predictor(out)
        return pred,None,None

if __name__ == '__main__':
    model = CNN()
    data = torch.randn(100, 32)
    out,_,_= model(data)
    print(out.shape)
