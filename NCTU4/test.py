from trainer import Trainer
from model import *
import torch 

class test:
    def __init__(self, G, FE, D, Q):
        self.FE = FE.apply(weights_init)
        self.G = self.load_w(G,'G')
        self.D = self.load_w(D,'D')
        self.Q = self.load_w(Q,'Q')
        self.batch_size = 100

    def load_w(self, model, name):
        return model.load_state_dict(torch.load(name + '.pt'))


    def testing(self):
        real_x, label, dis_c, con_c, noise = Trainer.setup()
        z, idx = Trainer._noise_sample(dis_c, con_c, noise, self.batch_size)
        print(z)
        print(idx)

    def print(self):
        print('hello')


