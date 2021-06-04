# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/12/110:18  
# 文件      ：cvae.py
# IDE       ：PyCharm
import torch
from modules.BaseVAE import BaseVAE
from torch import nn
from utils.losses import cvae_loss
from torch.nn import functional as F
from .types_ import *


class CondiationalVAE(BaseVAE):
    def __init__(self,
                 start_dim: int,
                 latent_dim: int,
                 batch_size: int,
                 hidden_dim: List = None,
                 **kwargs):
        super(CondiationalVAE, self).__init__()
        self.start_dim = start_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        if hidden_dim is None:
            self.hidden_dim = [32, 16, 8, 4]
        else:
            self.hidden_dim = hidden_dim
        # Build Encode
        modules = []
        for h_dim in self.hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=start_dim, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            start_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(self.hidden_dim[-1] * self.latent_dim, self.latent_dim)
        self.log_var = nn.Linear(self.hidden_dim[-1] * self.latent_dim, self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dim[-1] * self.latent_dim)

        self.hidden_dim.reverse()

        for i in range(len(self.hidden_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels=self.hidden_dim[i], out_channels=self.hidden_dim[i + 1],
                                       kernel_size=3, stride=1,
                                       padding=1),
                    nn.BatchNorm1d(self.hidden_dim[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim[-1],
                               self.hidden_dim[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.BatchNorm1d(self.hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(self.hidden_dim[-1], out_channels=self.hidden_dim[-1], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.mu(result)
        log_var = self.log_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z).view(-1, self.hidden_dim[0], self.latent_dim)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 均匀分布随机抽与std相同size
        return mu + eps * std

    def forward(self, x: Tensor) -> List[Tensor]:
        x = x.view(-1, self.start_dim, self.latent_dim)  # multi-scale input
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var, z]


if __name__ == '__main__':
    x = torch.rand(128, 784)
    cvae = CondiationalVAE(start_dim=28, latent_dim=28, batch_size=128)
    train = cvae.train()
    decoded, mu, log_var, encoded = train(x)
    loss, recons_loss, kl_loss = cvae_loss(decoded, x, mu, log_var, x_shape=512)
