# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1411:24  
# 文件      ：ae_demo.py
# IDE       ：PyCharm
import torch
from torch.nn import functional as F
from torch import nn, optim


class AutoEncoder(nn.Module):
    def __init__(self, x_shape):
        super(AutoEncoder, self).__init__()

        self.input = x_shape
        self.fc11 = nn.Linear(x_shape, 500)
        self.fc12 = nn.Linear(500, 450)
        self.fc13 = nn.Linear(450, 400)
        self.fc14 = nn.Linear(400, 350)
        self.fc15 = nn.Linear(350, 16)

        self.fc21 = nn.Linear(16, 350)
        self.fc22 = nn.Linear(350, 400)
        self.fc23 = nn.Linear(400, 450)
        self.fc24 = nn.Linear(450, 500)
        self.fc25 = nn.Linear(500, x_shape)

    def encoder(self, x):
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        z = torch.relu(self.fc15(x))
        return z

    def decoder(self, z):
        z = torch.relu(self.fc21(z))
        z = torch.relu(self.fc22(z))
        z = torch.relu(self.fc23(z))
        z = torch.relu(self.fc24(z))
        y = torch.sigmoid(self.fc25(z))
        return y

    def forward(self, x):
        encoded = self.encoder(x.view(-1, self.input))
        decoded = self.decoder(encoded)
        return encoded, decoded
