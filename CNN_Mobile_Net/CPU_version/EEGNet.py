import torch
import torch.nn as nn
import dataloader as data

class EEGNet(nn.Module):
    def __init__(self, activation):
        # Redifine the init funcion into the following
        super(EEGNet, self).__init__()
        self.activation = activation
        self.Firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (1, 51), stride = (1,1), padding = (0, 25), bias = False ),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        self.Depthwiseconv = nn.Sequential(
            # Gruop = input channel refer to conv to each layer seperately
            nn.Conv2d(16, 32, kernel_size = (2, 1), stride = (1,1), groups = 16, bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        if(self.activation == 'LeakyReLU'):
            self.Depthwiseconv.add_module('LeakyReLU',nn.LeakyReLU(0.2))
        if(self.activation == 'ReLU'):
            self.Depthwiseconv.add_module('ReLU',nn.ReLU())
        if(self.activation == 'ELU'):
            self.Depthwiseconv.add_module('ELU',nn.ELU(alpha = 1.0))
        self.Depthwiseconv.add_module('AvgPool2d',nn.AvgPool2d(kernel_size = (1, 4), stride = (1, 4), padding = 0))
        self.Depthwiseconv.add_module('Dropout',nn.Dropout(0.25))
            
        self.Seperableconv = nn.Sequential(
            # Reference :
            # 1. https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2
            # 2. https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
            #nn.Conv2d(32, 32, kernel_size = (15, 1), stride = (1,1), padding = (0, 7), bias = False),
            nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1,1), padding = (0, 7), bias = False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #nn.Conv2d(32, 32, kernel_size = (1, 1), stride = (1,1), padding = (0, 7), groups = 32, bias = False)
            )
        if(self.activation == 'LeakyReLU'):
            self.Seperableconv.add_module('LeakyReLU', nn.LeakyReLU(0.2))
        if(self.activation == 'ReLU'):
            self.Seperableconv.add_module('ReLU',nn.ReLU())
        if(self.activation == 'ELU'):
            self.Seperableconv.add_module('ELU', nn.ELU(alpha = 1.0))
        self.Seperableconv.add_module('AvgPool2d',nn.AvgPool2d(kernel_size = (1, 8), stride = (1, 8), padding = 0))
        self.Seperableconv.add_module('Dropout',nn.Dropout(0.25))
            
        self.Classfier = nn.Sequential(
            nn.Linear(in_features = 736, out_features = 2, bias = True)
            )

    def forward(self, x):
        out = self.Firstconv(x)
        out = self.Depthwiseconv(out)
        out = self.Seperableconv(out)
        out = out.view(out.size()[0], -1) # flatten, out.size()[0] means the number of the data, and flatten each into one dim feature
        return self.Classfier(out)

'''
Criteria = {
    'data_path' : './lab2',
    'Batch_size' : 128,
    'Learning_rate' : 1e-2,
    'Epochs' : 3,
    'activation': ['LeakyReLU','ReLU','ELU'],
    'model' : ['EEGNet', 'DeepConv']
}
train_loader, test_loader, train_size, test_size = data.read_bci_data(Criteria['data_path'],Criteria['Batch_size'])

for i, data in enumerate(train_loader):
    if i == 5:
        DD = data[0]
a = EEGNet('ELU')
q = a.forward(DD)
print(q.size())
'''










