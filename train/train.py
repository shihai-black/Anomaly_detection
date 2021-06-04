# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1411:31
# 文件      ：train.py
# IDE       ：PyCharm
from torch import optim
from utils.losses import loss_function, ae_loss,cvae_loss
import torch
import os
from modules.vae_demo import VAE
from modules.ae_demo import AutoEncoder
from modules.cvae import CondiationalVAE
from preprocessing.data_load import ImageDataIter, FaceImage
from base_config.arg_conf import arguments
from torchvision.utils import save_image
from callback.tensorboard_pytorch import board
from utils.logger import init_logger
from callback.modelcheckpoint import ModelCheckPoint

log = init_logger(logger_name='face_test',
                  logging_path='output/log/')


def model_choose(args, data1, models, x_shape):
    if args.module == 'vae':
        decoded, mu, log_var, encoded = models(data1)
        loss, recons_loss = loss_function(decoded, data1, mu, log_var, x_shape=x_shape)
    elif args.module == 'cvae':
        decoded, mu, log_var, encoded = models(data1)
        loss, recons_loss, kl_loss = cvae_loss(decoded, data1, mu, log_var, x_shape=x_shape)
    else:
        encoded, decoded = models(data1)
        loss, recons_loss = ae_loss(decoded, data1, x_shape=x_shape)

    return loss, recons_loss, decoded


def train(args, models, train_data, epoch, optimizer, device, x_shape):
    models.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_data):
        data = data.float().to(device)
        data = data / 255
        optimizer.zero_grad()
        loss, recons_loss, decoded = model_choose(args, data, models, x_shape)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.dataset),
                       100. * batch_idx / len(train_data),
                       loss.item() / len(data)))
    log.info('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_data.dataset)))
    # Tensor board train result
    result_loss = train_loss / len(train_data.dataset)
    board(comment_name=args.module, board_name='Train', loss=result_loss, epoch=epoch)

    return result_loss


def test(args, models, test_data, epoch, device, x_shape):
    models.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_data):
            data = data.float().to(device)
            data = data / 255
            loss, recons_loss, decoded = model_choose(args, data, models, x_shape)
            test_loss += loss
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data.view(args.batch_size, 1, 28, 28)[:n], decoded.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), args.image_path + args.module + '_reconstruction_' + str(epoch) + '.png',
                           nrow=n)
    result_loss = test_loss / len(test_data.dataset)
    log.info('====> Test set loss: {:.4f}'.format(result_loss))
    # Tensor board test result
    board(comment_name=args.module, board_name='Train', loss=result_loss, epoch=epoch)
    return result_loss


def run():
    args = arguments()
    data_path = os.path.join(args.path, 'data/')
    train_dataset = ImageDataIter(path=data_path, data_dir='train-images-idx3-ubyte', tag_dir='train-labels-idx1-ubyte',
                                  batch_size=args.batch_size,
                                  shuffle=True, num_works=2)
    test_dataset = ImageDataIter(path=data_path, data_dir='t10k-images-idx3-ubyte', tag_dir='t10k-labels-idx1-ubyte',
                                 batch_size=args.batch_size,
                                 shuffle=True, num_works=2)
    train_iter = train_dataset.data_load()
    test_iter = test_dataset.data_load()

    if torch.cuda.is_available():
        if not args.cuda:
            log.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if args.module == 'vae':
        model = VAE(x_shape=784).to(device)
    elif args.module == 'cvae':
        model = CondiationalVAE(start_dim=28, latent_dim=28, hidden_dim=[28,14,7,2],batch_size=args.batch_size).to(device)
    else:
        model = AutoEncoder(x_shape=784).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(args=args, models=model, train_data=train_iter, epoch=epoch, optimizer=optimizer, device=device,
              x_shape=784)
        test(args=args, models=model, test_data=test_iter, epoch=epoch, device=device,
             x_shape=784)


if __name__ == '__main__':
    face_run()
