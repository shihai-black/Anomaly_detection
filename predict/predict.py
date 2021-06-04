# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/189:06  
# 文件      ：predict.py
# IDE       ：PyCharm
import pandas as pd
import numpy as np
from modules.ae_demo import AutoEncoder
from modules.vae_demo import VAE
from preprocessing.data_load import FaceImage
import torch
import os
from base_config.arg_conf import arguments


class Predict(object):
    def __init__(self, args, data_dir, device,x_shape):
        self._args = args
        self._data_dir = data_dir
        self._device = device
        self._x_shape = x_shape

    def data_load(self):
        data_path = os.path.join(self._args.path, 'data/')
        pre_data = FaceImage(path=data_path,
                             data_dir=self._data_dir,
                             batch_size=args.batch_size, shuffle=True,
                             num_works=2)
        pre_iter = pre_data.data_load()
        return pre_iter

    def module_load(self):
        if args.module =='vae':
            model = VAE(self._x_shape)
        else:
            model = AutoEncoder(self._x_shape)

        file_name = os.path.join(args.path, f'output/checkpoint/{args.module}-model_best.pth')
        state = torch.load(file_name)
        model.load_state_dict(state['state_dict'])
        model.eval()
        return model

    def predict(self):
        pre_data = self.data_load()
        model = self.module_load()
        result_list = []

        for batch_idx, (data, label) in enumerate(pre_data):
            data = data.float().to(self._device)
            # data = data / 255
            if self._args.module == 'vae':
                decoded, mu, log_var, encoded = model(data)
            else:
                encoded, decoded = model(data)
            zip_list = zip(label.numpy().tolist(), data.numpy().tolist(), encoded.detach().numpy().tolist())
            for x, y, z in zip_list:
                result_list.append([x, y, z])

        df = pd.DataFrame(result_list, columns=['photo_id', 'feature', 'trans_feature'])
        return df


if __name__ == '__main__':
    args = arguments()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    data_dir = args.input
    predict = Predict(args=args,data_dir=data_dir,device=device,x_shape=512)
    result = predict.predict()
    result.to_csv('trans_sample_300.txt', index=None)
    print(np.array(result['trans_feature'][0]).shape)
    print(result)
