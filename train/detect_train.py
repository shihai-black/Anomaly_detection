# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/2315:09  
# 文件      ：detect_train.py
# IDE       ：PyCharm
from torch import optim
from utils.losses import loss_function, ae_loss,cvae_loss
from utils.summary import print_summary
import torch
import os
from modules.vae_demo import VAE
from modules.ae_demo import AutoEncoder
from modules.cvae import CondiationalVAE
from preprocessing.data_load import ImageDataIter, FaceImage, FaceTransData
from base_config.arg_conf import arguments
from torchvision.utils import save_image
from callback.tensorboard_pytorch import board
from utils.logger import init_logger
from callback.modelcheckpoint import ModelCheckPoint

log = init_logger(logger_name='face_test',
                  logging_path='output/log/')


class Train(object):
    def __init__(self, args, models, train_data, test_data, optimizer, device, x_shape):
        self.args = args
        self.models = models
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.device = device
        self.x_shape = x_shape

    def train(self, epoch):
        self.models.train()
        train_loss = 0
        train_recon_loss=0
        for batch_idx, (data, _) in enumerate(self.train_data):
            data1 = data.float().to(self.device)
            # data1 = data / 255
            self.optimizer.zero_grad()
            # choose model
            loss, recons_loss, decoded = self.model_choose(data1)
            loss.backward()
            train_loss += loss.item()
            train_recon_loss+=recons_loss.item()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRecon_loss:{:.6f}'.format(
                    epoch, batch_idx * len(data1), len(self.train_data.dataset),
                           100. * batch_idx / len(self.train_data),
                           loss.item() / len(data1),recons_loss.item()/len(data1)))
        log.info('====> Epoch: {} Average loss: {:.4f} Average recon loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_data.dataset),train_recon_loss/len(self.train_data.dataset)))
        # Tensor board train result
        result_loss = train_loss / len(self.train_data.dataset)
        board(comment_name=self.args.module, board_name='Train', loss=result_loss, epoch=epoch)

        return result_loss

    def test(self, epoch):
        self.models.eval()
        test_loss = 0
        test_recon_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_data):
                data1 = data.float().to(self.device)
                # data1 = data / 255
                loss,recons_loss,decoded=self.model_choose(data1)
                test_loss += loss
                test_recon_loss+=recons_loss
                if i == 0:
                    n = min(data1.size(0), 8)
                    m = 16
                    comparison = torch.cat(
                        [data1.view(self.args.batch_size, 1, int(512 / m), m)[:n],
                         decoded.view(self.args.batch_size, 1, int(512 / m), m)[:n]])
                    save_image(comparison.cpu(),
                               self.args.image_path + self.args.module + '_face' + str(epoch) + '.png',
                               nrow=n)
        result_loss = test_loss / len(self.test_data.dataset)
        reslut_recon_loss = test_recon_loss/len(self.test_data.dataset)
        log.info('====> Test set loss: {:.4f}\trecon loss: {:.4f}'.format(result_loss,reslut_recon_loss))
        # Tensor board test result
        board(comment_name=self.args.module, board_name='Test', loss=result_loss, epoch=epoch)
        return result_loss

    def model_choose(self,data1):
        if self.args.module == 'vae':
            decoded, mu, log_var, encoded = self.models(data1)
            loss, recons_loss = loss_function(decoded, data1, mu, log_var, x_shape=self.x_shape)
        elif self.args.module == 'cvae':
            decoded, mu, log_var, encoded = self.models(data1)
            loss, recons_loss, kl_loss = cvae_loss(decoded, data1, mu, log_var, x_shape=self.x_shape)
        else:
            encoded, decoded = self.models(data1)
            loss, recons_loss = ae_loss(decoded, data1, x_shape=self.x_shape)

        return loss,recons_loss,decoded

def detect_run():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_data = FaceTransData(path=data_path, data_dir='yddr_train.txt', log=log, batch_size=args.batch_size,
                               shuffle=True, num_works=2, is_normal=True)
    test_data = FaceTransData(path=data_path, data_dir='yddr_test.txt', log=log, batch_size=args.batch_size,
                              shuffle=True, num_works=2, is_normal=True)
    train_iter = train_data.data_load()
    test_iter = test_data.data_load()
    if torch.cuda.is_available():
        if not args.cuda:
            log.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if args.module == 'vae':
        model = VAE(x_shape=512).to(device)
    elif args.module == 'cvae':
        model = CondiationalVAE(batch_size=args.batch_size,start_dim=32,latent_dim=16).to(device)
    else:
        model = AutoEncoder(x_shape=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Print network structure
    log.info(f"print {args.module} structure")
    print_summary(model=model,input_size=(128,512))
    train = Train(args=args, models=model, train_data=train_iter, test_data=test_iter, optimizer=optimizer,
                  device=device,
                  x_shape=512)
    if args.save == 'y':
        save_module = ModelCheckPoint(model=model, optimizer=optimizer, args=args, log=log,name='ae',
                                      dir='output/checkpoint/')
        state = save_module.save_info(epoch=0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train(epoch)
        test_loss = train.test(epoch)
        if args.save == 'y':
            state = save_module.step_save(state=state, current=train_loss)


def test():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_data = FaceTransData(path=data_path, data_dir='yddr_trans_data.txt', log=log, batch_size=args.batch_size,
                               shuffle=True, num_works=2, is_normal=True)
    train_iter = train_data.data_load()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    model = CondiationalVAE(batch_size=args.batch_size, start_dim=32, latent_dim=16,hidden_dim=[32,16,8,4]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    log.info(f"print {args.module} structure")
    print_summary(model=model,input_size=(128,512))
    train = Train(args=args, models=model, train_data=train_iter, test_data=None, optimizer=optimizer,
                  device=device,
                  x_shape=512)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train(epoch)


if __name__ == '__main__':
    # detect_run()
    test()