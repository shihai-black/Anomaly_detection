# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1411:27  
# 文件      ：data_load.py
# IDE       ：PyCharm

import os
import numpy as np
import pandas as pd
from base_config.arg_conf import arguments
from torchvision.transforms import transforms
import torch
from utils.trans_tools import base64toint, minmaxscaler, cal_dis, str2timestamp
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset


class ImageDataIter(object):

    def __init__(self, path, data_dir, tag_dir, batch_size, shuffle, num_works, pin_memory=True):
        self.path = path
        self.data_dir = data_dir
        self.tag_dir = tag_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_works = num_works
        self.pin_memory = pin_memory

    def data_load(self):
        image = mnist.read_image_file(os.path.join(self.path, self.data_dir))
        label = mnist.read_label_file(os.path.join(self.path, self.tag_dir))
        dataset = TensorDataset(image, label)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_works,
            pin_memory=self.pin_memory
        )

        return data_loader


class FaceImage(object):
    def __init__(self, path, data_dir, batch_size, shuffle, num_works, pin_memory=True):
        self.path = path
        self.data_dir = data_dir
        self.id_list = []
        self.feature_list = []
        path = os.path.join(self.path, self.data_dir)

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split(',')
                feature = base64toint(info[1])

                self.id_list.append(int(info[0]))
                self.feature_list.append(feature)
        assert len(self.id_list) == len(self.feature_list)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_works = num_works
        self.pin_memory = pin_memory

    def data_load(self):
        label = torch.tensor(self.id_list)
        img = minmaxscaler(self.feature_list)
        img = torch.tensor(img)
        dataset = TensorDataset(img, label)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_works,
            pin_memory=self.pin_memory
        )
        return data_loader


class FaceTransData(object):
    def __init__(self, path, data_dir,log, batch_size, shuffle, num_works, pin_memory=True, is_normal=True,
                 is_space=True, is_speed=True):
        self.path = path
        self.data_dir = data_dir
        self.log = log
        self.label_list = []
        self.feature_list = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_works = num_works
        self.pin_memory = pin_memory
        self.is_normal = is_normal
        self.is_space = is_space
        self.is_speed = is_speed

    def data_preprocess(self):
        path = os.path.join(self.path, self.data_dir)
        df = pd.read_csv(path)
        self.log.info('Data read out')
        if self.is_normal:
            df = df[df['sample_type'] == 1].reset_index(drop=True)
        else:
            df = df[df['sample_type'] == 0].reset_index(drop=True)
        # 转化图片特征
        df['trans1'] = df.apply(lambda rows: base64toint(rows['feature1']), axis=1)
        df['trans2'] = df.apply(lambda rows: base64toint(rows['feature2']), axis=1)

        # 获取图片特征差异值
        df['image_feature'] = df.apply(lambda rows: np.array(rows['trans1']) - np.array(rows['trans2']), axis=1)

        # 获取gps特征差异值
        df['gps_x_dif'] = df.apply(lambda rows: rows['x1'] - rows['x2'], axis=1)
        df['gps_y_dif'] = df.apply(lambda rows: rows['y1'] - rows['y2'], axis=1)

        # 获取speed特征
        df['t1'] = df.apply(lambda rows: str2timestamp(rows['time1']), axis=1)
        df['t2'] = df.apply(lambda rows: str2timestamp(rows['time2']), axis=1)
        df['t'] = abs(df['t1'] - df['t2'])
        df['dis'] = df.apply(lambda rows: cal_dis(rows['x1'], rows['y1'], rows['x2'], rows['y2']), axis=1)
        df['speed'] = df.apply(lambda rows: rows['dis'] / (rows['t'] + 30), axis=1)

        return df

    def data_load(self):
        df = self.data_preprocess()
        self.log.info('Data processing completed')
        for i in range(len(df)):
            feature = df['image_feature'][i]
            label = df['label'][i]
            self.label_list.append(label)
            self.feature_list.append(feature)

        img = minmaxscaler(self.feature_list)
        img = torch.tensor(img)
        labels = torch.tensor(self.label_list)
        dataset = TensorDataset(img, labels)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_works,
            pin_memory=self.pin_memory
        )
        return data_loader


if __name__ == '__main__':
    args = arguments()
    # train_dataset = ImageDataIter(path=args.path, data_dir='train-images-idx3-ubyte', tag_dir='train-labels-idx1-ubyte',
    #                               batch_size=args.batch_size,
    #                               shuffle=True, num_works=2)
    # train_iter = train_dataset.data_load()

    # face = FaceImage(path='/home/caojinlei/solve_problem/Unsupervised_clustering/data/',
    #                  data_dir='gold_sample_feature.txt',
    #                  batch_size=32, shuffle=True,
    #                  num_works=2)
    # data = face.data_load()
    data_path = os.path.join(args.path, 'data/')
    face_trans = FaceTransData(path=data_path, data_dir='detection_sample_over.txt', batch_size=64, shuffle=True,
                               num_works=2,is_normal=True)
    data_iter = face_trans.data_load()
