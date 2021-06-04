# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2021/1/417:12  
# 文件      ：f_anogan_predict.py
# IDE       ：PyCharm
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.losses import f_anogan_score
from utils.logger import init_logger
from modules_gan.f_AnoGAN import Generator, Discriminator, Encoder
from utils.losses import ae_loss, loss_function, cvae_loss
from preprocessing.data_load import FaceTransData
import torch
import os
from base_config.arg_conf import arguments

log = init_logger(logger_name='face_test',
                  logging_path='/home/caojinlei/solve_problem/Unsupervised_clustering/output/log/')


class GANPredict(object):
    def __init__(self, args, data_path, data_dir, device, log, x_shape):
        """
        Use detect algorithm for data predict
        :param args: External configuration parameters
        :param data_path: path of the predict data file
        :param data_dir: Name of the predict data file
        :param device:
        :param log:
        :param x_shape:
        """
        self._args = args
        self._data_path = data_path
        self._data_dir = data_dir
        self._device = device
        self._x_shape = x_shape
        self._log = log

    def pd_data(self):
        """
        Use pandas load the predict data
        :return:predict data
        """
        data = pd.read_csv(self._data_path + self._data_dir)
        return data

    def data_load(self, is_normal=True):
        """
        The iterator used to load the data
        :param is_normal: Whether to select normal data
        :return:The iterator
        """
        pre_data = FaceTransData(path=self._data_path, data_dir=self._data_dir, batch_size=1,
                                 shuffle=True, num_works=4, is_normal=is_normal, log=self._log)
        pre_iter = pre_data.data_load()
        return pre_iter

    def model_load(self):
        """
        Load module
        :return:
        """
        # generate
        model_g = Generator().to(self._device)
        file_name_g = os.path.join(self._args.path, f'output/checkpoint/f_gen-checkpoint-epoch.pth')
        state_g = torch.load(file_name_g)
        model_g.load_state_dict(state_g['state_dict'])
        model_g.eval()

        # discriminant
        model_d = Discriminator().to(self._device)
        file_name_d = os.path.join(self._args.path, f'output/checkpoint/f_dis-checkpoint-epoch.pth')
        state_d = torch.load(file_name_d)
        model_d.load_state_dict(state_d['state_dict'])
        model_d.eval()

        # encoder
        model_e = Encoder().to(self._device)
        file_name_e = os.path.join(self._args.path, f'output/checkpoint/f_enc-checkpoint-epoch.pth')
        state_e = torch.load(file_name_e)
        model_e.load_state_dict(state_e['state_dict'])
        model_e.eval()

        return model_g, model_d, model_e

    def predict(self, is_normal=True):
        """
        :param is_normal: Whether to select normal data
        :return:
        """
        self._log.info('load data')
        pre_data = self.data_load(is_normal=is_normal)
        self._log.info('load model')
        model_g, model_d, model_e = self.model_load()
        label_list = []
        error_list = []
        tk1 = tqdm(pre_data, total=len(pre_data))
        for batch_idx, (data, label) in enumerate(tk1):
            real_image = data.float().to(self._device)
            z = model_e(real_image)
            rec_image = model_g(z)
            recons_loss = f_anogan_score(real_image, rec_image, model_d)
            label_list.extend(label.cpu().numpy().tolist())
            error_list.extend(recons_loss.cpu().detach().numpy().tolist())
        result_list = np.array([label_list, error_list]).T
        df = pd.DataFrame(result_list, columns=['label', 'reconstruction_error'])
        return df


def f_anogan_result(original_dir, trans_dir):
    # The front part
    args = arguments()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    data_path = os.path.join(args.path, 'data/')

    # Load the predict data
    predict = GANPredict(args=args, data_path=data_path, data_dir=trans_dir, device=device, x_shape=512, log=log)
    df_base = predict.pd_data()  # detect input data
    df_normal = predict.predict(is_normal=True)  # normal predict data
    df_abnormal = predict.predict(is_normal=False)  # abnormal predict data
    df_all = pd.concat([df_abnormal, df_normal])
    df = pd.merge(df_base, df_all, how='left', on='label')
    df_n = df[['label', 'id1', 'id2', 's_label', 't1', 't2', 'sample_type', 'reconstruction_error']]
    # load original data and get photo_id list
    original_path = os.path.join(data_path, original_dir)
    df_4 = pd.read_csv(original_path)
    id_list = df_4['photo_id'].unique().tolist()

    # Get the error mean and error list for each photo_ID in the predict data
    result_list = []
    for i in range(len(id_list)):
        photo_id = id_list[i]
        s_label = df_n[(df_n['id1'] == photo_id) | (df_n['id2'] == photo_id)]['s_label'].values[0]
        error_list = df_n[(df_n['id1'] == photo_id) | (df_n['id2'] == photo_id)]['reconstruction_error'].tolist()
        error_mean = np.mean(error_list)
        result_list.append([photo_id, s_label, error_mean, error_list])
    df_5 = pd.DataFrame(result_list, columns=['photo_id', 's_label', 'error_mean', 'error_list'])

    # Add the error mean and error list to the original data
    df_result = pd.merge(df_4, df_5, on=['photo_id', 's_label'])

    # Output data set
    out_path = os.path.join(args.path, 'output/predict/')
    df_result.to_csv(out_path + args.module + '_result2.csv', index=None)
    df_result['photo_id'].to_csv(out_path + args.module + '_photo_id2.txt', index=None)
    df_result['s_label'].to_csv(out_path + args.module + '_s_label2.txt', index=None)


if __name__ == '__main__':
    f_anogan_result(original_dir='yddr_original_data.txt', trans_dir='yddr_trans_data.txt')
