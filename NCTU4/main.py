from model import *
from trainer import Trainer
from test import test

fe = FrontEnd()
d = D()
q = Q()
g = G()

train = False
train = True

if train == True:
  for i in [fe, d, q, g]:
    i.cuda()
    i.apply(weights_init)

  trainer = Trainer(g, fe, d, q)
  trainer.train()

else:
  gotest = test(g, fe, d, q)
  gotest.testing()
