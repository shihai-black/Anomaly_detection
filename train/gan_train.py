# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/12/1816:21  
# 文件      ：gan_train.py
# IDE       ：PyCharm
# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/2315:09
# 文件      ：detect_train.py
# IDE       ：PyCharm
from torch import optim
from torch import nn
from torch.autograd import Variable
from utils.losses import loss_function, ae_loss, cvae_loss, Anomaly_score
from utils.summary import print_summary
import torch
import os
from modules_gan.AnoGAN import Generator, Discriminator
from modules.cvae import CondiationalVAE
from preprocessing.data_load import ImageDataIter, FaceImage, FaceTransData
from base_config.arg_conf import arguments
from torchvision.utils import save_image
from callback.tensorboard_pytorch import board
from utils.logger import init_logger
from callback.modelcheckpoint import ModelCheckPoint

log = init_logger(logger_name='face_test',
                  logging_path='output/log/')


class GANTrain(object):
    def __init__(self, args, model_gen, model_dis, train_data, test_data, optim_gen, optim_dis, device, x_shape):
        self.args = args
        self.model_gen = model_gen
        self.model_dis = model_dis
        self.optim_gen = optim_gen
        self.optim_dis = optim_dis
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.x_shape = x_shape

    def train(self, epoch):
        self.model_gen.train()
        self.model_dis.train()
        train_dis_loss = 0
        train_gen_loss = 0
        loss_func = nn.MSELoss()
        for batch_idx, (data, _) in enumerate(self.train_data):
            data1 = data.float().to(self.device)
            ones_label = torch.ones(data1.size()[0], 1).to(self.device)
            zeros_label = torch.zeros(data1.size()[0], 1).to(self.device)

            # discriminator
            self.optim_dis.zero_grad()
            z = torch.randn(data1.size()[0], 512).to(self.device)
            gen_fake = self.model_gen(z)
            dis_fake, _ = self.model_dis(gen_fake)

            dis_real, _ = self.model_dis(data1)
            dis_loss = torch.sum(loss_func(dis_fake, zeros_label)) + torch.sum(loss_func(dis_real, ones_label))
            dis_loss.backward()
            train_dis_loss += dis_loss.item()
            self.optim_dis.step()

            # generator
            self.optim_gen.zero_grad()
            z = torch.randn(data1.size()[0], 512).to(self.device)
            gen_fake = self.model_gen(z)
            dis_fake, _ = self.model_dis(gen_fake)

            gen_loss = torch.sum(loss_func(dis_fake, ones_label))  # fake classified as real
            gen_loss.backward()
            train_gen_loss += gen_loss.item()
            self.optim_gen.step()
            if batch_idx % self.args.log_interval == 0:
                log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tgen_Loss: {:.6f}\tdis_loss:{:.6f}'.format(
                    epoch, batch_idx * len(data1), len(self.train_data.dataset),
                           100. * batch_idx / len(self.train_data),
                           gen_loss.item() / len(data1), dis_loss.item() / len(data1)))
        log.info('====> Epoch: {} Average gen loss: {:.4f} Average dis loss: {:.4f}'.format(
            epoch, train_gen_loss / len(self.train_data.dataset), train_dis_loss / len(self.train_data.dataset)))
        # Tensor board train result
        result_loss = train_dis_loss / len(self.train_data.dataset)
        board(comment_name=self.args.module, board_name='Train', loss=result_loss, epoch=epoch)

        return result_loss

    def test(self, epoch):
        self.model_gen.eval()
        self.model_dis.eval()
        z = torch.randn(1, 512).to(self.device)
        z.requires_grad = True
        z_optimizer = torch.optim.Adam([z], lr=1e-4)
        test_recon_loss = 0
        for i, (data, _) in enumerate(self.test_data):
            data1 = data.float().to(self.device)
            gen_fake = self.model_gen(z)
            recons_loss = Anomaly_score(data1, gen_fake, model_d=self.model_dis, Lambda=0.01)
            recons_loss.backward()
            test_recon_loss += recons_loss.item()
            z_optimizer.step()
        result_loss = test_recon_loss / len(self.test_data.dataset)
        log.info('====> Test set loss: {:.4f}'.format(result_loss))
        # Tensor board test result
        board(comment_name=self.args.module, board_name='Test', loss=result_loss, epoch=epoch)
        return result_loss, z


def gan_run():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_data = FaceTransData(path=data_path, data_dir='yddr_train.txt', log=log, batch_size=args.batch_size,
                               shuffle=True, num_works=2, is_normal=True)
    # test_data = FaceTransData(path=data_path, data_dir='yddr_test.txt', log=log, batch_size=args.batch_size,
    #                           shuffle=True, num_works=2, is_normal=True)
    train_iter = train_data.data_load()
    # test_iter = test_data.data_load()
    if torch.cuda.is_available():
        if not args.cuda:
            log.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    model_g = Generator().to(device)
    model_d = Discriminator().to(device)

    opt_g = optim.Adam(model_g.parameters(), lr=5 * 1e-4)
    opt_d = optim.Adam(model_d.parameters(), lr=1e-5)

    # Print network structure
    log.info(f"print {args.module} structure")
    train = GANTrain(args=args, model_gen=model_g, model_dis=model_d, train_data=train_iter, test_data=None,
                     optim_gen=opt_g, optim_dis=opt_d, device=device, x_shape=512)
    if args.save == 'y':
        save_module_g = ModelCheckPoint(model=model_g, optimizer=opt_g, args=args, log=log, name='gen',
                                        save_best_model=False, dir='output/checkpoint/')
        state_g = save_module_g.save_info(epoch=0)

        save_module_d = ModelCheckPoint(model=model_d, optimizer=opt_d, args=args, log=log, name='dis',
                                        save_best_model=False, dir='output/checkpoint/')
        state_d = save_module_d.save_info(epoch=0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train(epoch)
        if args.save == 'y':
            state_g = save_module_g.step_save(state=state_g, current=train_loss)
            state_d = save_module_d.step_save(state=state_d, current=train_loss)



def test():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_data = FaceTransData(path=data_path, data_dir='yddr_trans_data.txt', log=log, batch_size=args.batch_size,
                               shuffle=True, num_works=2, is_normal=True)
    test_data = FaceTransData(path=data_path, data_dir='yddr_trans_data.txt', log=log, batch_size=args.batch_size,
                              shuffle=True, num_works=2, is_normal=True)
    train_iter = train_data.data_load()
    test_iter = test_data.data_load()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    model_g = Generator().to(device)
    model_d = Discriminator().to(device)

    opt_g = optim.Adam(model_g.parameters(), lr=5 * 1e-4)
    opt_d = optim.Adam(model_d.parameters(), lr=1e-5)

    # Print network structure
    log.info(f"print {args.module} structure")

    train = GANTrain(args=args, model_gen=model_g, model_dis=model_d, train_data=train_iter, test_data=test_iter,
                     optim_gen=opt_g, optim_dis=opt_d, device=device, x_shape=512)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train(epoch)
        test_loss, z = train.test(epoch)


if __name__ == '__main__':
    # detect_run()
    test()
