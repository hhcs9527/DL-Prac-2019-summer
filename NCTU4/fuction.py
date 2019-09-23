import torch
import numpy as np
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the variable size
def setup(batch):
  real_x = torch.FloatTensor(batch, 1, 28, 28).to(device)
  label = torch.FloatTensor(batch, 1).to(device)
  dis_c = torch.FloatTensor(batch, 10).to(device)
  con_c = torch.FloatTensor(batch, 2).to(device)
  noise = torch.FloatTensor(batch, 62).to(device) 

  real_x = Variable(real_x)
  label = Variable(label, requires_grad=False)
  dis_c = Variable(dis_c)
  con_c = Variable(con_c)
  noise = Variable(noise)

  return real_x, label, dis_c, con_c, noise




# Return concated noise, for generating picture
def fix_noise_cat(batch, instruction, produce):
    c = (torch.rand(batch, 2).to(device)* 2 - 1).to(device)

    if instruction == 'training':
        idx = np.arange(10).repeat(batch/10)
        #idx = [produce] * batch
    else:
        idx = [produce] * batch            # doesn't work
        #idx = [9]*batch      #-> work
        #idx = np.arange(10)  
        

    one_hot = np.zeros((batch, 10))
    one_hot[range(batch), idx] = 1


    fix_noise = torch.randn(batch, 62).to(device) 

    real_x, label, dis_c, con_c, noise = setup(batch)

    noise.data.copy_(fix_noise)
    dis_c.data.copy_(torch.Tensor(one_hot))
    con_c.data.copy_(c)
    return torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)



# Return the noise sample for the training part
def _noise_sample(dis_c, con_c, noise, batch):

    idx = np.random.randint(10, size = batch)
    c = np.zeros((batch, 10))
    c[range(batch),idx] = 1.0

    constant = (torch.rand(batch, 2).to(device)* 2 - 1).to(device) 

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.copy_(constant)

    noise = torch.randn(batch, 62).to(device)

    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

    return z, idx



def fixedNoise(count = 100, number = None):
  num_dis_c = 1
  dis_c_dim = 10
  num_z = 62
  num_con_c = 2
  fixed_noise = torch.randn(count, num_z, 1, 1)

  if number == None:
    idx = np.arange(dis_c_dim).repeat(10)
  else:
    number = [number]*100 #[changeIndex(number[x]) for x in range(len(number))]
    print(number)
    idx = number

  dis_c = torch.zeros(count, num_dis_c, dis_c_dim)
  for i in range(num_dis_c):
      dis_c[torch.arange(0, count), i, idx] = 1.0
  dis_c = dis_c.view(count, -1, 1, 1)          

  con_c = torch.rand(count, num_con_c, 1, 1) * 2  - 1
        
  fixed_noise = torch.cat((fixed_noise, dis_c), 1)
  fixed_noise = torch.cat((fixed_noise, con_c), 1)

  fixed_noise.to(device)
  return fixed_noise

#def getNoiseSample(self, batch_size):
#    z = torch.randn(batch_size, num_z, 1, 1)
#    idx = np.zeros((num_dis_c, batch_size))
#    dis_c = torch.zeros(batch_size, num_dis_c, dis_c_dim)
#
#    for i in range(num_dis_c):
#        idx[i] = np.random.randint(dis_c_dim, size = batch_size)
#        dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0        
#    dis_c = dis_c.view(batch_size, -1, 1, 1)
#
#    con_c = torch.rand(batch_size, num_con_c, 1, 1) * 2 - 1 
#
#    z = torch.cat((z, dis_c), 1)
#    z = torch.cat((z, con_c), 1)
#    z = z.to(device)
#
#    return z, idx
def changeIndex(number):
  if number == 0:
      return 7
  elif number == 1:
      return 2
  elif number == 2:
      return 0
  elif number == 3:
      return 5
  elif number == 4:
      return 8
  elif number == 5:
      return 6
  elif number == 6:
      return 3
  elif number == 7:
      return 4
  elif number == 8:
      return 1
  elif number == 9:
      return 9

if __name__ == '__main__' :
    batch = 100
    instruction = 'trainin'
    produce = 5

    a =  fix_noise_cat(batch, instruction, produce)
    print(a[0])
    print(a[1])
    #tmp = a[0]
    #for i in range(100-1):
    #    diff = a[i+1] - a[i]
    #    print(diff)