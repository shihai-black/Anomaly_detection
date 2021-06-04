# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/12/1414:38  
# 文件      ：summary.py
# IDE       ：PyCharm
from torchsummary import summary


def print_summary(model, input_size,device='cuda'):
    return summary(model=model, input_size=input_size,device=device)
