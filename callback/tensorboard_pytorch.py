# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1615:01  
# 文件      ：tensorboard_pytorch.py
# IDE       ：PyCharm
from tensorboardX import SummaryWriter, FileWriter


def board(comment_name, board_name, loss, epoch):
    with SummaryWriter(comment=comment_name) as write:
        write.add_scalar(board_name, loss, epoch)

