# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2021/1/1316:13  
# 文件      ：nice_run.py
# IDE       ：PyCharm
from base_config.arg_conf import arguments
from train.nice_train import nice_run
if __name__ == '__main__':
    arg = arguments()
    # ************train*******************
    if arg.train == 'y':
        nice_run()
