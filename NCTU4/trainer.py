import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import random
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np

import fuction as f

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class log_gaussian:
  def __call__(self, x, mu, var):
    logli = -0.5 * (var.mul(2*np.pi) + 1e-6).log() - (x-mu).pow(2).div(var.mul(2.0) + 1e-6)
    return -(logli.sum(1).mean())


class Trainer:
  def __init__(self, d, dh, qh, g, Ep):

# initialize model variable
    self.D = d
    self.G = g  
    self.DH = dh  
    self.QH = qh  
    self.Ep = Ep
    self.batch_size = 100

# initialize loss function
    self.criterionD = nn.BCELoss()
    self.criterionQ_dis = nn.CrossEntropyLoss()
    self.criterionQ_con = log_gaussian()

# initialize the optimizer
    self.optimD = optim.Adam([{'params':self.D.parameters()}, {'params':self.DH.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    self.optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.QH.parameters()}], lr=0.001, betas=(0.5, 0.99))


  def get_path(self, name):
    return name + '.pt'  


  def train(self):
    real_x, label, dis_c, con_c, noise = f.setup(self.batch_size)

    dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    D_L = []
    G_L = []

    for epoch in range(self.Ep):
      G_record_loss = 0
      D_record_loss = 0

      for num_iters, batch_data in enumerate(dataloader, 0):
      # Training Discriminater optimizer
        # real part
        self.optimD.zero_grad()
        
        x, _ = batch_data

        bs = x.size(0)
        real_x.data.resize_(x.size()) 
        label.data.resize_(bs, 1)
        dis_c.data.resize_(bs, 10)
        con_c.data.resize_(bs, 2)
        noise.data.resize_(bs, 62)
        
        
        real_x.data.copy_(x)
        fe_out1 = self.D(real_x)

        probs_real = self.DH(fe_out1)
        label.data.fill_(1)
        loss_real = self.criterionD(probs_real, label)

        loss_real.backward()

        # fake part
        z, idx = f._noise_sample(dis_c, con_c, noise, bs)
        fake_x = self.G(z)
        fe_out2 = self.D(fake_x.detach())
        probs_fake = self.DH(fe_out2)
        label.data.fill_(0)
        loss_fake = self.criterionD(probs_fake, label)

        loss_fake.backward()

        D_loss = loss_real + loss_fake
        D_record_loss += loss_real + loss_fake

        self.optimD.step()
        

        # Training generator optimizer
        self.optimG.zero_grad()

        fe_out = self.D(fake_x)
        probs_fake = self.DH(fe_out)
        label.data.fill_(1.0)

        reconstruct_loss = self.criterionD(probs_fake, label)
        
        q_logits, q_mu, q_var = self.QH(fe_out)
        class_ = torch.LongTensor(idx).to(device)
        target = Variable(class_)

        dis_loss = self.criterionQ_dis(q_logits, target)
        con_loss = self.criterionQ_con(con_c, q_mu, q_var)*0.1 # lower bound
        
        G_loss = reconstruct_loss + dis_loss + con_loss
        G_record_loss += reconstruct_loss + dis_loss + con_loss
        G_loss.backward()
        
        self.optimG.step()


        if num_iters % 100 == 0:

          print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
            epoch+1, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy())
          )


      z = f.fix_noise_cat(100, 'training', epoch)
      x_save = self.G(z)
      name = './try_ans/'+ '#epoch' + str(epoch+1)  +'.png'
      save_image(x_save.data, name, nrow=10)

      torch.save(self.G.state_dict(), self.get_path('G'))

      D_L.append(D_record_loss/len(dataset))
      G_L.append(G_record_loss/len(dataset))

    torch.cuda.empty_cache()
    return  D_L ,G_L