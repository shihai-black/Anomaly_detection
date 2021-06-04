# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1515:01  
# 文件      ：trans_tools.py
# IDE       ：PyCharm
import base64
import struct
import numpy as np
import time
import math
import os


def base64toint(base64Str: str) -> list:
    """
    Convert Base64 to int
    @param base64Str:base64 str
    @return:
    """
    byte_temp = base64.b64decode(base64Str)
    # Specify that the feature is 512 dimensions, integer type
    array_temp = struct.unpack('512H', byte_temp)
    return list(array_temp)


def minmaxscaler(data):
    """
    Normalized
    :param data: input data
    :return: Normalized results
    """
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)


def ensure_dir(path):
    """
    Make sure the path exists
    :param path:
    :return: makedir path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def str2timestamp(s: str) -> int:
    timestamp = int(time.mktime(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S")))
    return timestamp


def cal_dis(lng1, lat1, lng2, lat2):
    """
    Calculate the distance between the two points
    :param lng1, lat1: The latitude and longitude of the first point
    :param lng2, lat2: The latitude and longitude of the second point
    :return: distance, float
    """
    dist_angle = math.sin(lat1 * math.pi / 180) * math.sin(lat2 * math.pi / 180) + \
                 math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * \
                 math.cos((lng1 - lng2) * math.pi / 180)
    if dist_angle > 1:
        dist_angle = 1
    return 6371000 * math.acos(dist_angle)
