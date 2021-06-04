# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2021/1/1215:41  
# 文件      ：NICE.py
# IDE       ：PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.distributions import Distribution, Uniform


class CouplingLayer(nn.Module):
    """
    分块耦合层
    """

    def __init__(self, data_dim, hidden_dim, mask, num_layer=4):
        super().__init__()
        assert data_dim % 2 == 0
        self.x_shape = data_dim
        self.mask = mask
        model_list = [nn.Linear(self.x_shape, hidden_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layer - 2):
            model_list.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2)
                )
            )
        model_list.append(nn.Linear(hidden_dim, data_dim))
        self.layer1 = nn.Sequential(*model_list)

    def forward(self, input, logdet, invert=False):
        if not invert:
            x1, x2 = self.mask * input, (1 - self.mask) * input
            y1, y2 = x1, x2 + (self.layer1(x1) * (1. - self.mask))
            return y1 + y2, logdet

        # inverse additive coupling layer
        y1, y2 = self.mask * input, (1 - self.mask) * input
        x1, x2 = y1, y2 - (self.layer1(y1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

    def forward(self, input, logdet, invert=False):
        log_det_jacobian = torch.sum(self.log_scale_vector)
        if invert:
            return torch.exp(- self.log_scale_vector) * input, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * input, logdet + log_det_jacobian


class LogisticDistribution(Distribution):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        if self.device:
            z = Uniform(torch.cuda.FloatTensor([0.]), torch.cuda.FloatTensor([1.])).sample(size)
        else:
            z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)
        return torch.log(z) - torch.log(1. - z)


class NICE(nn.Module):
    def __init__(self, data_dim, device, hidden_dim=1000, samples=128, num_net_layer=6, num_coupling_layers=3):
        super().__init__()

        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.samples = samples
        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                 for i in range(num_coupling_layers)]
        self.coupling_layers = nn.ModuleList(
            [CouplingLayer(data_dim=data_dim, hidden_dim=hidden_dim,
                           mask=masks[i], num_layer=num_net_layer)
             for i in range(num_coupling_layers)]
        )
        self.scaling_layer = ScalingLayer(data_dim=data_dim)
        self.prior = LogisticDistribution()

    def _get_mask(self, dim, orientation=True):
        mask = torch.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask
        if self.device:
            mask = mask.to(device=self.device)
        return mask

    def f(self, x):
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(self.coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def f_inverse(self, z):
        x = z
        x, _ = self.scaling_layer(x, 0, invert=True)
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, num_sample):
        z = self.prior.sample([num_sample, self.data_dim]).view(self.samples, self.data_dim)
        return self.f_inverse(z)

    def forward(self, x, invert=False):
        if not invert:
            z, log_det_jacobian = self.f(x)
            log_likelhood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
            return z, log_likelhood
        return self.f_inverse(x)


if __name__ == '__main__':
    x = torch.randn(100, 512)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NICE(data_dim=512, device=device, num_coupling_layers=4).to(device)
    model.train()

    opt = Adam(params=model.parameters(), lr=5e-5)
    mean_likelihood = 0.0
    num_minibatches = 0

    x = x.view(-1, 512) + torch.rand(512) / 256
    x = x.to(device)
    x = torch.clamp(x, 0, 1)  # 将上下游进行处理
    z, likelihood = model(x)
    loss = torch.mean(likelihood)

    print(z.size())
    print(loss)
    loss.backward()
    opt.step()
    model.zero_grad()
    mean_likelihood += loss
    num_minibatches += 1
