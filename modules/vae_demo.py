# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1316:44  
# 文件      ：vae_demo.py
# IDE       ：PyCharm
import torch
from torch.nn import functional as F
from torch import nn, optim


class VAE(nn.Module):
    def __init__(self, x_shape):
        super(VAE, self).__init__()

        self.input = x_shape
        self.fc11 = nn.Linear(x_shape, 500)
        self.fc12 = nn.Linear(500, 450)
        self.fc13 = nn.Linear(450, 400)
        self.fc14 = nn.Linear(400, 350)
        self.fc151 = nn.Linear(350, 300)
        self.fc152 = nn.Linear(350, 300)

        self.fc21 = nn.Linear(300, 350)
        self.fc22 = nn.Linear(350, 400)
        self.fc23 = nn.Linear(400, 450)
        self.fc24 = nn.Linear(450, 500)
        self.fc25 = nn.Linear(500, x_shape)

    def encode(self, x):
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))

        return torch.relu(self.fc151(x)), torch.relu(self.fc152(x))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 均匀分布随机抽与std相同size
        return mu + eps * std

    def decode(self, z):
        z = torch.relu(self.fc21(z))
        z = torch.relu(self.fc22(z))
        z = torch.relu(self.fc23(z))
        z = torch.relu(self.fc24(z))
        y = torch.sigmoid(self.fc25(z))
        return y

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z
