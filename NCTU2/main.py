import dataloader as data
import numpy as np
import os
import torch.nn as nn

def hello():
    print('hello')
def ioio():
    print('ioio')
def kf():
    print('kfjdk')

option = {'data_path' : './lab2',
          'Batch_size' : 64,
          'Learning_rate' : 1e-2,
          'Epochs' : 150,
          'activation': ['nn.LeakyReLU()','nn.ReLU()','nn.ELU()'],
          'ac': ['hello()','ioio()','kf()'],
          #'aec': [hello(),ioio(),kf()],
          #'hello' : 'hello()',
          #'ioio' : ioio()

         }

# Read file
#data.read_bci_data(option['data_path'])
#func = option['ac']
func = hello()

#func
#print(option['cool'][0])

# pytorch optional activation in dict方式 , 用下載下來的嘗試 optional 方式
#nn.LeakyReLU()
#nn.ReLU()
#nn.ELU()

# 求出檔案 在畫圖