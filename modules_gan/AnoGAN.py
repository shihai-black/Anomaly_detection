# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/12/1810:29  
# 文件      ：AnoGAN.py
# IDE       ：PyCharm
import torch
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init
from utils.summary import print_summary


class Generator(nn.Module):
    def __init__(self, x_shape=512, hidden_list=None, lat_dim=7):
        super(Generator, self).__init__()
        self.x_shape = x_shape
        self.lat_dim = lat_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=x_shape, out_features=self.x_shape * self.lat_dim * self.lat_dim),
            nn.BatchNorm1d(self.x_shape * self.lat_dim * self.lat_dim),
            nn.ReLU(),
        )
        if hidden_list:
            self.hidden_list = hidden_list
        else:
            self.hidden_list = [256, 128, 64, 32, 1]
        modules = []
        for hidden in self.hidden_list:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.x_shape, hidden, 3, 1, 1),
                    nn.BatchNorm2d(hidden),
                    nn.LeakyReLU()
                )
            )
            self.x_shape = hidden

        self.layer2 = nn.Sequential(*modules)

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=self.lat_dim * self.lat_dim, out_features=512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.layer1(z).view(-1, 512, self.lat_dim, self.lat_dim)
        out = self.layer2(z)
        out = out.view(out.size()[0], -1)
        out = self.layer3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, hidden_list=None, start_dim=1, lat_dim=7):
        super(Discriminator, self).__init__()
        if hidden_list:
            self.hidden_list = hidden_list
        else:
            self.hidden_list = [1, 32, 64, 128, 256]
        self.start_dim = start_dim
        self.lat_dim = lat_dim

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=self.lat_dim * self.lat_dim),
            nn.BatchNorm1d(self.lat_dim * self.lat_dim),
            nn.ReLU(),
        )
        modules = []
        for hidden in self.hidden_list:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.start_dim, hidden, 3, 1, 1),
                    nn.BatchNorm2d(hidden),
                    nn.LeakyReLU()
                )
            )
            self.start_dim = hidden
        self.layer2 = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_list[-1] * self.lat_dim * self.lat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(-1, 1, self.lat_dim, self.lat_dim)
        out = self.layer2(out).view(out.size()[0],-1)
        feature = out
        pred = self.fc(out)
        return pred, feature


if __name__ == '__main__':
    learning_rate = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    X = Variable(torch.randn(128, 512)).to(device)
    # print_summary(generator, (128, 512))
    # print_summary(discriminator, (128, 512))
    ones_label = Variable(torch.ones(128, 1)).to(device)
    zeros_label = Variable(torch.zeros(128, 1)).to(device)
    gen_optim = torch.optim.Adam(generator.parameters(), lr=5 * learning_rate, betas=(0.5, 0.999))
    dis_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # generator
    loss_func = nn.MSELoss()
    gen_optim.zero_grad()
    # z = Variable(torch.randn(128, 512)).to(device)
    z = torch.randn(128, 512).to(device)
    gen_fake = generator.forward(z)
    dis_fake, _ = discriminator.forward(gen_fake)

    gen_loss = torch.sum(loss_func(dis_fake, ones_label))  # fake classified as real
    gen_loss.backward(retain_graph=True)
    gen_optim.step()

    # discriminator
    dis_optim.zero_grad()
    z = Variable(torch.randn(128, 512)).to(device)
    gen_fake = generator.forward(z)
    dis_fake, _ = discriminator.forward(gen_fake)

    dis_real, _ = discriminator.forward(X)
    dis_loss = torch.sum(loss_func(dis_fake, zeros_label)) + torch.sum(loss_func(dis_real, ones_label))
    dis_loss.backward()
    dis_optim.step()
    print("iteration gen_loss: {} dis_loss: {}".format(gen_loss.data, dis_loss.data))
