# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1414:15  
# 文件      ：arg_conf.py
# IDE       ：PyCharm
import argparse


def arguments():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1111, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save', type=str, default='y', metavar='S',
                        help='Whether or not to save(default: y)')
    parser.add_argument('--train', type=str, default='y', metavar='S',
                        help='Train or predict(default: y)')
    parser.add_argument('--module', type=str, default='ae', metavar='N',
                        help='Which model to choose(default: ae)')
    parser.add_argument('--path', type=str, default='/home/caojinlei/solve_problem/Unsupervised_clustering/',
                        metavar='S',
                        help='data path')
    parser.add_argument('--image_path', type=str,
                        default='/home/caojinlei/solve_problem/Unsupervised_clustering/output/image/',
                        metavar='S',
                        help='image path')
    parser.add_argument('--log-interval', type=int, default=128, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
