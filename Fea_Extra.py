# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:24:13 2022

@author: 29792
"""

''' ============== 特征提取的类 =====================
时域特征 ：11类
频域特征 : 13类
总共提取特征 ： 24类

参考文献 英文文献 016_C_(Q1 时域和频域共24种特征参数 )  Fault diagnosis of rotating machinery based on multiple ANFIS combination with GAs

'''
import math
import numpy as np
import scipy.stats

class Fea_Extra():
    def __init__(self, Signal, Fs = 25600):
        self.signal = Signal
        self.Fs = Fs



    def Both_Fea(self):
        """
        :return: 时域、频域特征 array
        """
        
        tf_fea = get_wavelet_packet_feature(self.signal)
        fea = tf_fea

        return fea







import pywt
import numpy as np


def get_wavelet_packet_feature(data, wavelet='db3', mode='symmetric', maxlevel=6):  # db2         1 2 3  4  5 6   7 
    """                                                                                           2 4 8 16 32 64 128
    提取 小波包特征
    
    @param data: shape 为 (n, ) 的 1D array 数据，其中，n 为样本（信号）长度
    @return: 最后一层 子频带 的 能量百分比
    """
    wp = pywt.WaveletPacket(data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    
    nodes = [node.path for node in wp.get_level(maxlevel, 'freq')]  # 获得最后一层的节点路径   freq    natural
    
    e_i_list = []  # 节点能量
    for node in nodes:
        e_i = np.linalg.norm(wp[node].data, ord=None) ** 2  # 求 2范数，再开平方，得到 频段的能量（能量=信号的平方和）
        e_i_list.append(e_i)
    
    # 以 频段 能量 作为特征向量
    # features = e_i_list
        
    # 以 能量百分比 作为特征向量，能量值有时算出来会比较大，因而通过计算能量百分比将其进行缩放至 0~100 之间
    e_total = np.sum(e_i_list)  # 总能量
    features = []
    for e_i in e_i_list:
        features.append(e_i / e_total * 100)  # 能量百分比
    
    return np.array(features)




