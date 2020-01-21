import torch
import torch.nn as nn
import dataloader as data

class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        self.activation = activation
        self.FirstConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = (1,5)),
            nn.Conv2d(25, 25, kernel_size = (2, 1)),
            nn.BatchNorm2d(25)
            )
        if(self.activation == 'LeakyReLU'):
            self.FirstConv.add_module('LeakyReLU',nn.LeakyReLU(0.2))
        if(self.activation == 'ReLU'):
            self.FirstConv.add_module('ReLU',nn.ReLU())
        if(self.activation == 'ELU'):
            self.FirstConv.add_module('ELU',nn.ELU(alpha = 1.0))
        self.FirstConv.add_module('MaxPool', nn.MaxPool2d(kernel_size = (1,2)))
        self.FirstConv.add_module('Dropout', nn.Dropout(0.5))

        self.SecondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = (1,5)),
            nn.BatchNorm2d(50)
            )
        if(self.activation == 'LeakyReLU'):
            self.SecondConv.add_module('LeakyReLU',nn.LeakyReLU(0.2))
        if(self.activation == 'ReLU'):
            self.SecondConv.add_module('ReLU',nn.ReLU())
        if(self.activation == 'ELU'):
            self.SecondConv.add_module('ELU',nn.ELU(alpha = 1.0))
        self.SecondConv.add_module('MaxPool', nn.MaxPool2d(kernel_size = (1,2)))
        self.SecondConv.add_module('Dropout', nn.Dropout(0.5))

        self.ThirdConv = nn.Sequential(
            nn.Conv2d(50,100, kernel_size = (1,5)),
            nn.BatchNorm2d(100)
            )
        if(self.activation == 'LeakyReLU'):
            self.ThirdConv.add_module('LeakyReLU',nn.LeakyReLU(0.2))
        if(self.activation == 'ReLU'):
            self.ThirdConv.add_module('ReLU',nn.ReLU())
        if(self.activation == 'ELU'):
            self.ThirdConv.add_module('ELU',nn.ELU(alpha = 1.0))
        self.ThirdConv.add_module('MaxPool', nn.MaxPool2d(kernel_size = (1,2)))
        self.ThirdConv.add_module('Dropout', nn.Dropout(0.5))

        self.FourthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size = (1,5)),
            nn.BatchNorm2d(200)
            )
        if(self.activation == 'LeakyReLU'):
            self.FourthConv.add_module('LeakyReLU',nn.LeakyReLU(0.2))
        if(self.activation == 'ReLU'):
            self.FourthConv.add_module('ReLU',nn.ReLU())
        if(self.activation == 'ELU'):
            self.FourthConv.add_module('ELU',nn.ELU(alpha = 1.0))
        self.FourthConv.add_module('MaxPool', nn.MaxPool2d(kernel_size = (1,2)))
        self.FourthConv.add_module('Dropout', nn.Dropout(0.5))

        self.Classify = nn.Sequential(
            # Calculate by the output size
            nn.Linear(in_features = 200*43, out_features = 2, bias=True)
            )

    def forward(self, x):
        out = self.FirstConv(x)
        out = self.SecondConv(out)
        out = self.ThirdConv(out)
        out = self.FourthConv(out)
        out = out.view(out.size()[0], -1)
        return self.Classify(out)
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
a = DeepConvNet('ELU')
q = a.forward(DD)
print(q.size())
'''








