# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/12/319:33  
# 文件      ：f_AnoGAN.py
# IDE       ：PyCharm
import torch
from torch.optim.rmsprop import RMSprop
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init
from utils.summary import print_summary


class Generator(nn.Module):
    def __init__(self, z_dim=100, hidden_list=None, lat_dim=28,x_shape=512):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.lat_dim = lat_dim
        self.x_shape =x_shape
        if hidden_list:
            self.hidden_list = hidden_list
        else:
            self.hidden_list = [256,128,64,32]
        self.start_dim = self.hidden_list[0]
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.z_dim, out_features=self.hidden_list[0] * self.lat_dim * self.lat_dim),
            nn.BatchNorm1d(self.hidden_list[0] * self.lat_dim * self.lat_dim),
            nn.ReLU(),
        )
        modules = []
        for hidden in self.hidden_list:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.start_dim, hidden, 3, 1, 1),
                    nn.BatchNorm2d(hidden),
                    nn.LeakyReLU()
                )
            )
            self.start_dim = hidden

        self.layer2 = nn.Sequential(*modules)

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=self.hidden_list[-1]*self.lat_dim * self.lat_dim, out_features=self.x_shape),
            nn.BatchNorm1d(self.x_shape),
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.layer1(z)
        z =z.view(-1, self.hidden_list[0], self.lat_dim, self.lat_dim)
        out = self.layer2(z)
        out = out.view(out.size()[0], -1)
        out = self.layer3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, hidden_list=None, lat_dim=28,x_shape=512,out_feat=False):
        super(Discriminator, self).__init__()
        if hidden_list:
            self.hidden_list = hidden_list
        else:
            self.hidden_list = [16,32,64,128]
        self.start_dim = self.hidden_list[0]
        self.x_shape =x_shape
        self.lat_dim = lat_dim
        self.out_feat = out_feat

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.x_shape, out_features=self.start_dim*self.lat_dim * self.lat_dim),
            nn.BatchNorm1d(self.start_dim*self.lat_dim * self.lat_dim),
            nn.ReLU(),
        )
        modules = []
        for hidden in self.hidden_list:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.start_dim, hidden, 3, 1, 1),
                    nn.LayerNorm(torch.Size([hidden,self.lat_dim,self.lat_dim])),
                    nn.LeakyReLU()
                )
            )
            self.start_dim = hidden
        self.layer2 = nn.Sequential(*modules)

        self.fc = nn.Linear(self.hidden_list[-1] * self.lat_dim * self.lat_dim, 1)

    def forward(self, x):
        out = self.layer1(x.view(-1,self.x_shape))
        out = out.view(-1, self.hidden_list[0], self.lat_dim, self.lat_dim)
        out = self.layer2(out).view(out.size()[0],-1)
        if self.out_feat:
            return out
        out = self.fc(out)
        return out

class Encoder(nn.Module):
    def __init__(self,hidden_list=None,lat_dim=28,z_dim=100,x_shape=512):
        super(Encoder, self).__init__()
        if hidden_list:
            self.hidden_list = hidden_list
        else:
            self.hidden_list = [2,4,8,16]
        self.start_dim = self.hidden_list[0]
        self.lat_dim = lat_dim
        self.x_shape = x_shape
        self.z_dim = z_dim

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.x_shape,out_features=self.hidden_list[0]*self.lat_dim**2), # 1*28*28
            nn.BatchNorm1d(self.lat_dim**2),
            nn.LeakyReLU()
        )
        models =[]
        for hidden_list in self.hidden_list:
            models.append(
                nn.Sequential(
                    nn.Conv2d(self.start_dim,hidden_list,3,1,1),
                    nn.BatchNorm2d(hidden_list),
                    nn.LeakyReLU()
                )
            )
            self.start_dim = hidden_list
        self.layer2= nn.Sequential(*models)

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=self.hidden_list[-1]*self.lat_dim**2,out_features=self.z_dim)
        )
        

    def forward(self, x):
        x = self.layer1(x.view(-1,self.x_shape))
        x = x.view(-1,self.hidden_list[0],self.lat_dim,self.lat_dim)  # batch_size*1*28*28
        x = self.layer2(x).view(x.size()[0],-1)  # batch_size*256*28*28
        out = self.layer3(x)
        return out
if __name__ == '__main__':
    x = torch.randn(128,512)
    model_e = Encoder()
    model_g = Generator(z_dim=100,lat_dim=28,x_shape=512)
    model_d = Discriminator()
    
    optim_en = RMSprop(params=model_e.parameters(),lr=1e-3)
    optim_en = RMSprop(params=model_e.parameters(),lr=1e-3)
    optim_en = RMSprop(params=model_e.parameters(),lr=1e-3)
    
    loss_fun = nn.MSELoss()
    optim_en.zero_grad()
    optim_gn.zero_grad()

    loss = torch.sum(loss_fun(x,x1))
    loss.backward()
    optim_en.step()
    optim_gn.step()
    print(f'loss_encoder:{loss}')


