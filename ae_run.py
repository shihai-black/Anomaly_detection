# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1411:29  
# 文件      ：run.py
# IDE       ：PyCharm
from train.train import run
from train.detect_train import detect_run
from predict.detect_predict import get_result
from base_config.arg_conf import arguments
from train.gan_train import gan_run
from predict.gan_predict import get_result
if __name__ == '__main__':
    arg = arguments()
    # ************train*******************
    # run()
    # face_run()
    if arg.train == 'y':
        # gan_run()
        detect_run()
    else:
        get_result(original_dir='yddr_original_data.txt', trans_dir='yddr_trans_data.txt')
