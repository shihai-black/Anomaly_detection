# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2021/1/216:11  
# 文件      ：f_anogan_train.py
# IDE       ：PyCharm
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from torch import optim
from torch import nn,Tensor
import numpy as np
from torch.autograd import Variable
from utils.losses import loss_function, ae_loss, cvae_loss, Anomaly_score
import torch
import os
from modules_gan.f_AnoGAN import Generator, Discriminator,Encoder
from preprocessing.data_load import ImageDataIter, FaceImage, FaceTransData
from base_config.arg_conf import arguments
from torchvision.utils import save_image
from callback.tensorboard_pytorch import board
from utils.logger import init_logger
from callback.modelcheckpoint import ModelCheckPoint

log = init_logger(logger_name='face_test',
                  logging_path='output/log/')


class GANTrain(object):
    def __init__(self, args, model_gen, model_dis,model_enc,train_data, test_data,lambda_gp,z_dim,
                 optim_gen, optim_dis,optim_enc, device, x_shape):
        self.args = args
        self.model_gen = model_gen
        self.model_dis = model_dis
        self.model_enc = model_enc
        self.optim_gen = optim_gen
        self.optim_dis = optim_dis
        self.optim_enc = optim_enc
        self.z_dim = z_dim
        self.train_data = train_data
        self.test_data = test_data
        self.lambda_gp = lambda_gp
        self.device = device
        self.x_shape = x_shape

    def compute_gradient_penalty(self,model_dis,real_samples,fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.rand((real_samples.size()[0],1),requires_grad=True).to(self.device)
        ones = torch.ones(real_samples.size()[0],1).to(self.device)
        interpolates = (alpha * real_samples+((ones-alpha)*fake_samples)).to(self.device)
        d_interpolates = model_dis(interpolates)
        fake = torch.ones(real_samples.size()[0],1,requires_grad=False).to(self.device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0),-1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def gan_train(self, epoch):
        self.model_gen.train()
        self.model_dis.train()
        train_dis_loss = 0
        train_gen_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_data):
            # -------------------
            # train discriminator
            # -------------------
            self.optim_dis.zero_grad()

            # get real and fake image
            real_image = data.float().view(data.size()[0],-1).to(self.device)
            z = torch.randn(real_image.size()[0], self.z_dim).to(self.device)
            fake_image = self.model_gen(z)

            # Send the image to discriminator
            real_validity = self.model_dis(real_image)
            fake_validity = self.model_dis(fake_image)

            # WGAN_GP
            gradient_penalty = self.compute_gradient_penalty(self.model_dis,real_image.data,fake_image.data)
            d_loss = -torch.mean(real_validity)+torch.mean(fake_validity)+self.lambda_gp*gradient_penalty
            d_loss.backward()
            train_dis_loss+=d_loss

            self.optim_dis.step()

            # -------------------
            # train generator
            # -------------------
            self.optim_gen.zero_grad()

            z = torch.randn(real_image.size()[0], self.z_dim).to(self.device)
            fake_images = self.model_gen(z)
            fake_validity = self.model_dis(fake_images)

            # generator loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            train_gen_loss +=g_loss
            self.optim_gen.step()

            # print result
            if batch_idx % self.args.log_interval == 0:
                log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tgen_Loss: {:.6f}\tdis_loss:{:.6f}'.format(
                    epoch, batch_idx * len(real_image), len(self.train_data.dataset),
                           100. * batch_idx / len(self.train_data),
                           g_loss.item() / len(real_image), d_loss.item() / len(real_image)))
        log.info('====> Epoch: {} Average gen loss: {:.4f} Average dis loss: {:.4f}'.format(
            epoch, train_gen_loss / len(self.train_data.dataset), train_dis_loss / len(self.train_data.dataset)))
        # Tensor board train result
        result_loss = train_dis_loss / len(self.train_data.dataset)
        board(comment_name=self.args.module, board_name='GAN_Train', loss=result_loss, epoch=epoch)
        return result_loss

    def enc_train(self,epoch):
        self.model_gen.eval()
        self.model_dis.eval()
        self.model_enc.train()
        kappa = 1
        train_enc_loss = 0
        loss = nn.MSELoss()
        for batch_idx, (data, _) in enumerate(self.train_data):
            self.optim_enc.zero_grad()

            real_image = data.float().view(data.size()[0],-1).to(self.device)
            z = self.model_enc(real_image)

            rec_image = self.model_gen(z)
            real_dis = self.model_dis(real_image)
            fake_dis = self.model_dis(rec_image)

            enc_loss = loss(real_image,rec_image)+kappa*loss(real_dis,fake_dis)
            train_enc_loss += enc_loss
            enc_loss.backward()

            self.optim_enc.step()
            # print result
            if batch_idx % self.args.log_interval == 0:
                log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tenc_loss: {:.6f}'.format(
                    epoch, batch_idx * len(real_image), len(self.train_data.dataset),
                           100. * batch_idx / len(self.train_data),
                           enc_loss.item() / len(real_image)))
        log.info('====> Epoch: {} Average enc loss: {:.4f}'.format(
            epoch, train_enc_loss / len(self.train_data.dataset)))
        # Tensor board train result
        result_loss = train_enc_loss / len(self.train_data.dataset)
        board(comment_name=self.args.module, board_name='Enc_train', loss=result_loss, epoch=epoch)
        return result_loss



def f_anogan_run():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_data = FaceTransData(path=data_path, data_dir='yddr_train.txt', log=log, batch_size=args.batch_size,
                               shuffle=True, num_works=4, is_normal=True)
    train_iter = train_data.data_load()
    if torch.cuda.is_available():
        if not args.cuda:
            log.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:1" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    model_g = Generator().to(device)
    model_d = Discriminator().to(device)
    model_e = Encoder().to(device)

    opt_g = Adam(model_g.parameters(), lr=5e-5)
    opt_d = Adam(model_d.parameters(), lr=5e-5)
    opt_e = RMSprop(model_e.parameters(), lr=5e-5)

    # Print network structure
    log.info(f"print {args.module} structure")
    train = GANTrain(args=args, model_gen=model_g, model_dis=model_d, model_enc=model_e,train_data=train_iter, test_data=None,
                     optim_gen=opt_g, optim_dis=opt_d, optim_enc=opt_e,device=device, x_shape=512,lambda_gp=10,z_dim=100)
    if args.save == 'y':
        save_module_g = ModelCheckPoint(model=model_g, optimizer=opt_g, args=args, log=log, name='f_gen',
                                        save_best_model=False, dir='output/checkpoint/')
        state_g = save_module_g.save_info(epoch=0)

        save_module_d = ModelCheckPoint(model=model_d, optimizer=opt_d, args=args, log=log, name='f_dis',
                                        save_best_model=False, dir='output/checkpoint/')
        state_d = save_module_d.save_info(epoch=0)

        save_module_e = ModelCheckPoint(model=model_e, optimizer=opt_e, args=args, log=log, name='f_enc',
                                        save_best_model=False, dir='output/checkpoint/')
        state_e = save_module_e.save_info(epoch=0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.gan_train(epoch)
        if args.save == 'y':
            state_g = save_module_g.step_save(state=state_g, current=train_loss)
            state_d = save_module_d.step_save(state=state_d, current=train_loss)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.enc_train(epoch)
        if args.save == 'y':
            state_e = save_module_e.step_save(state=state_e, current=train_loss)

def test():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_data = ImageDataIter(path=data_path, data_dir='train-images-idx3-ubyte', tag_dir='train-labels-idx1-ubyte',
                                  batch_size=args.batch_size,
                                  shuffle=True, num_works=0)
    test_data = ImageDataIter(path=data_path, data_dir='t10k-images-idx3-ubyte', tag_dir='t10k-labels-idx1-ubyte',
                                 batch_size=args.batch_size,
                                 shuffle=True, num_works=0)
    train_iter = train_data.data_load()
    test_iter = test_data.data_load()
    device = torch.device("cuda:1" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    model_g = Generator(x_shape=784).to(device)
    model_d = Discriminator(x_shape=784).to(device)
    model_e = Encoder(x_shape=784).to(device)

    opt_g = optim.Adam(model_g.parameters(), lr=5e-4)
    opt_d = optim.Adam(model_d.parameters(), lr=2e-4)
    opt_e = optim.Adam(model_d.parameters(), lr=2e-4)

    # Print network structure
    log.info(f"print {args.module} structure")

    train = GANTrain(args=args, model_gen=model_g, model_dis=model_d, model_enc=model_e,train_data=train_iter, test_data=test_iter,
                     optim_gen=opt_g, optim_dis=opt_d, optim_enc=opt_e,device=device, x_shape=784,lambda_gp=10,z_dim=100)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.gan_train(epoch)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.enc_train(epoch)



if __name__ == '__main__':
    # detect_run()
    test()



