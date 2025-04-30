import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,nz=100, feature_dim=27, ngf=8, nc=4):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.in_FC = nn.Linear(nz, ngf * 8 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh())
        self.out_FC = nn.Linear(nc * 4 * 8, feature_dim)

    def forward(self, x):
        x = self.in_FC(x)
        x = x.view(x.size(0), self.ngf * 8, 4)
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.out_FC(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, feature_dim=27, ndf=8, nc=4):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.in_FC = nn.Linear(feature_dim, nc * 32)
        self.main = nn.Sequential(
            nn.Conv1d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8))
        self.out_FC1 = nn.Linear(ndf * 8 * (32 // 8), 32)
        self.leak_relu = nn.LeakyReLU(0.2, inplace=True)
        self.out_FC2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.in_FC(x)
        x = x.view(x.size(0), self.nc, 32)
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.out_FC1(x)
        x = self.leak_relu(x)
        x = self.out_FC2(x)
        x = self.sigmoid(x)
        return x