# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/12/319:40  
# 文件      ：gan_run.py
# IDE       ：PyCharm
from base_config.arg_conf import arguments
from train.gan_train import gan_run
from train.f_anogan_train import f_anogan_run
from predict.gan_predict import get_result
from predict.f_anogan_predict import f_anogan_result

if __name__ == '__main__':
    arg = arguments()
    # ************train*******************
    # run()
    # face_run()
    if arg.train == 'y':
        # gan_run()
        f_anogan_run()
    else:
        # get_result(original_dir='yddr_original_data.txt', trans_dir='yddr_trans_data.txt')
        f_anogan_result(original_dir='yddr_original_data.txt', trans_dir='yddr_trans_data.txt')
