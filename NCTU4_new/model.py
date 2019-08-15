import torch.nn as nn

Change = True
class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()
#####
    if not Change:
      self.main = nn.Sequential(
        nn.Conv2d(1024, 1, 1),
        nn.Sigmoid()
      )
    else:
      self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv2d(1024, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. 100, 64, 1, 1
        nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. 100, 128, 1, 1
        nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. 100, 256, 1, 1
        nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 8),
        # state size. 100, 512, 1, 1
        nn.Conv2d(64 * 8, 1, 4, 2, 1, bias=False),
        nn.Sigmoid()
      )
      # kernel = 4, stride = 2, pad = 1 -> maintain the size
#####

  def forward(self, x):
    output = self.main(x).view(-1, 1)
    print('after D size : ', self.main(x).size())
    return output


class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()
#####
    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, 10, 1)
    self.conv_mu = nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Conv2d(128, 2, 1)
#####
  def forward(self, x):

    y = self.conv(x)
    print('y : ',y.size())
    disc_logits = self.conv_disc(y).squeeze()
    print('disc_logits : ',disc_logits.size())
    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var 


class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()
#####
    if not Change:
      self.main = nn.Sequential(
      # state size. 100, 74, 1, 1
        nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(True),
      # state size. 100, 1024, 1, 1
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      # state size. 100, 128, 7, 7
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      # state size. 100, 64, 14, 14
      nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
      nn.Sigmoid()
      # state size. 100, 1, 28, 28
      )

    # Substitute Generator of DCGAN here
    else:
      self.main = nn.Sequential(
        # state size. 100, 74, 1, 1
        nn.ConvTranspose2d(74, 64 * 8, 4, 1, 1, bias=False),
        nn.BatchNorm2d(64 * 8),
        nn.ReLU(True),
        # state size. 100, 512, 2, 2
        nn.ConvTranspose2d(64 * 8, 64 * 4, 2, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.ReLU(True),
        # state size. 100, 256, 2, 2
        nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 1, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.ReLU(True),
      )
      self.main1 = nn.Sequential(
        # state size. 100, 512, 4, 4
        nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.ReLU(True),
        # state size. 100, 256, 8, 8
        nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 2),
        nn.ReLU(True),
        # state size. (64*4) x 8 x 8

        # state size. (64*2) x 16 x 16
        nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # state size. (64) x 32 x 32
        nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (1) x 64 x 64 -> generate a grey scale picture
    )
  ######

  def forward(self, x):
    print('x size : ', x.size())
    output = self.main(x)
    print('gen size : ',output.size())
    return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)