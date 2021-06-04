# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1411:29  
# 文件      ：losses.py
# IDE       ：PyCharm
from torch.nn import functional as F
from torch.nn import MSELoss
import torch


def loss_function(recon_x, x, mu, log_var, x_shape):
    """
    VAE loss function
    :param recon_x:
    :param x: input_x
    :param mu:mean
    :param log_var:log(std)
    :param x_shape:x.shape
    :return: Loss,Reconstruction of probability
    """
    BCE = F.mse_loss(input=recon_x, target=x.view(-1, x_shape))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = BCE + KLD
    return loss, BCE


def ae_loss(decoded, x, x_shape):
    loss = F.mse_loss(input=decoded, target=x.view(-1, x_shape))
    return loss, loss


def kl_divergence(p, q, bsz):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :param bsz:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0) / bsz
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def cvae_loss(recon_x, x, mu, log_var, x_shape):
    kld_weight = 0.005  # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(recon_x.view(-1, x_shape), x.view(-1, x_shape))
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)  # 消除batch影响
    loss = recons_loss + kld_weight * KLD
    return loss, recons_loss, -KLD


def multi_scale_loss(recon_x, x, recon_x_quarter, x_quarter, recon_x_half, x_half, mu, log_var, x_shape):
    kld_weight = 0.0006  # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(recon_x.view(-1, x_shape), x.view(-1, x_shape)) \
                  + F.mse_loss(recon_x_quarter.view(-1, x_shape / 4), x_quarter.view(-1, x_shape / 4)) \
                  + F.mse_loss(recon_x_half.view(-1, x_shape / 2), x_half.view(-1, x_shape / 2))
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)  # 消除batch影响
    loss = recons_loss + kld_weight * KLD
    return loss, recons_loss, -KLD


def Anomaly_score(x, G_z, model_d, Lambda=0.1):
    _, x_feature = model_d(x)
    _, G_z_feature = model_d(G_z)

    residual_loss = torch.sum(torch.abs(x - G_z), dim=1)
    discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature), dim=1)

    total_loss = (1 - Lambda) * residual_loss + Lambda * discrimination_loss
    return total_loss


def f_anogan_score(x, rx, model_d):
    dis_x = model_d(x)
    dis_rx = model_d(rx)
    loss = torch.nn.MSELoss(reduce=False)
    total_loss = loss(x, rx) + loss(dis_rx, dis_x)
    return total_loss
