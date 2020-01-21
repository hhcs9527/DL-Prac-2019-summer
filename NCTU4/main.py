# where the file should be 
# /home/viplab/Desktop/Huang/DL-Prac-2019-summer/NCTU4_new
from model import *
from trainer import Trainer
from test import test
import torch
import plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = Discriminator().to(device)
dh = D_head().to(device)
qh = Q_head().to(device)
g = Generator().to(device)

train = False
#train = True

# Training 
if train == True:
  for i in [d, dh, qh, g]:
    i.apply(weights_init)

  trainer = Trainer(d, dh, qh, g, 60)
  D_L ,G_L = trainer.train()
  D_L = [D_L[i].cpu() for i in range(len(D_L))]

  G_L = [G_L[i].cpu() for i in range(len(G_L))]
  #plot.PlotComp(len(D_L), D_L ,G_L)


# Testing 
else:
  g.load_state_dict(torch.load('G.pt'))
  g.eval()
  gotest = test(g)

  for produce in range(10):
    gotest.testing(produce)

  print('Testing is done !!')
torch.cuda.empty_cache()
