from trainer import Trainer
from model import *
import torch 
import numpy as np
import math
from torchvision.utils import save_image
from torch.autograd import Variable
import fuction as f

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class test:
    def __init__(self, G):
        self.G = G
        self.batch_size = 100


    def change_num(self, produce):
        #dic = {'0':8, '1':1, '2':4, '3':5, '4':3, '5':6, '6':2, '7':9, '8':0, '9':7}
        dic = [8, 1, 4, 5, 3, 6, 2, 9, 0, 7]
        for i in range(len(dic)):
            if dic[i] == produce:
                break
        return i


    def testing(self, produce):
        word = 'training'
        word = 'testing'
        #z = f.fix_noise_cat(self.batch_size, word, produce)
        z = f.fixedNoise( number = produce).to(device)
        #print(z)
        name = './pic/'+ '#produce_' + str(produce)  +'.png'
        img = self.G(z)
        #prod = int(self.change_num(produce))
        # img.data[produce][:] [produce * 10 : produce * 10+10]
        #print(img.size())
        save_image(img.data, name, nrow = 10)
        return z



