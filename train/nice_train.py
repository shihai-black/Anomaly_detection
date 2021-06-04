# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2021/1/1315:26  
# 文件      ：nice_train.py
# IDE       ：PyCharm
from torch import optim
import os
import torch
from modules_flow.NICE import NICE
from utils.summary import print_summary
from preprocessing.data_load import ImageDataIter, FaceImage, FaceTransData
from base_config.arg_conf import arguments
from torchvision.utils import save_image
from callback.tensorboard_pytorch import board
from utils.logger import init_logger
from callback.modelcheckpoint import ModelCheckPoint

log = init_logger(logger_name='NICE',
                  logging_path='output/log/')


class NICETrain(object):
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
        num_minibatches = 0
        for batch_idx, (data, _) in enumerate(self.train_data):
            data = data.view(-1,self.x_shape)
            data1 = (data + torch.rand(self.x_shape)) / 255
            data1 = data1.to(self.device)
            data1 = torch.clamp(data1, 0, 1)
            z, likelihood = self.models(data1)
            loss = -torch.mean(likelihood)
            loss.backward()
            self.optimizer.step()
            self.models.zero_grad()
            train_loss -=loss.item()
            num_minibatches+=1
        # Tensor board train result
        result_loss = train_loss / num_minibatches
        log.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, result_loss))
        board(comment_name=self.args.module, board_name='Train', loss=result_loss, epoch=epoch)

        return result_loss

    def test(self, epoch):
        self.models.eval()
        test_loss = 0
        num_minibatches = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_data):
                data = data.view(-1, self.x_shape)/255
                data1 = data + torch.rand(self.x_shape) / 255
                data1 = data1.to(self.device)
                data1 = torch.clamp(data1, 0, 1)
                z, likelihood = self.models(data1)
                loss = -torch.mean(likelihood)
                test_loss -= loss.item()
                num_minibatches += 1
                if i == 0:
                    n = min(data1.size(0), 8)
                    comparison = torch.cat(
                        [data1.view(self.args.batch_size, 1, 28, 28)[:n], z.view(self.args.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                               self.args.image_path + self.args.module + '_reconstruction_' + str(epoch) + '.png',
                               nrow=n)
            result_loss = test_loss / num_minibatches
            log.info('====> Test set loss: {:.4f}\t'.format(result_loss))
            # Tensor board test result
            board(comment_name=self.args.module, board_name='Test', loss=result_loss, epoch=epoch)
            return result_loss


def nice_run():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_dataset = ImageDataIter(path=data_path, data_dir='train-images.idx3-ubyte', tag_dir='train-labels.idx1-ubyte',
                                  batch_size=args.batch_size,
                                  shuffle=True, num_works=2)
    test_dataset = ImageDataIter(path=data_path, data_dir='t10k-images.idx3-ubyte', tag_dir='t10k-labels.idx1-ubyte',
                                 batch_size=args.batch_size,
                                 shuffle=True, num_works=2)
    train_iter = train_dataset.data_load()
    test_iter = test_dataset.data_load()

    if torch.cuda.is_available():
        if not args.cuda:
            log.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:1" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    model = NICE(data_dim=784, device=device, hidden_dim=1000,num_net_layer=6,num_coupling_layers=4).to(device)
    optimizer = optim.Adam(model.parameters())
    # Print network structure
    log.info(f"print {args.module} structure")
    # print_summary(model=model, input_size=(128, 784))
    train = NICETrain(args=args, models=model, train_data=train_iter, test_data=test_iter, optimizer=optimizer,
                      device=device, x_shape=784)
    if args.save == 'y':
        save_module = ModelCheckPoint(model=model, optimizer=optimizer, args=args, log=log,name='nice',mode='max',
                                      dir='output/checkpoint/')
        state = save_module.save_info(epoch=0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train.train(epoch)
        test_loss = train.test(epoch)
        if args.save == 'y':
            state = save_module.step_save(state=state, current=train_loss)

