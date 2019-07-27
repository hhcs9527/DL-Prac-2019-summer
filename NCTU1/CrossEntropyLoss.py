import torch
import torch.nn as nn
class customLoss(nn.Module):
    def __init__(self, weight):
        super(customLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.weight = weight
        
    def forward(self, outputs, targets):        
        # transform targets to one-hot vector
        targets_onehot = torch.zeros_like(outputs)
        #targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.unsqueeze(-1), 1)
        print(targets.unsqueeze(-1))
        print(targets_onehot)
        
        # nn.CrossEntropyLoss 
        # combines nn.LogSoftmax() and nn.NLLLoss()
        outputs = self.softmax(outputs)
        self.weight = self.weight.expand_as(outputs) # expend as dim -> 發生機率
        # Cross Entropy的定義就是 log(發生機率的期望值)，所以才會是 -log＊(softmax result)*target
        loss = -targets_onehot.float() * torch.log(outputs) 
        return torch.mean(self.weight * loss)


# define CrossEntropyLoss with weights
weight = torch.Tensor([1, 5, 10])
# define inputs, official and custom loss
outputs = torch.Tensor([[0.9, 0.5, 0.05], [0.01, 0.2, 0.7]])
targets = torch.Tensor([0, 1]).long()
criterion = nn.CrossEntropyLoss(weight=weight)
custom_criterion = customLoss(weight=weight)
# run metrics
loss = criterion(outputs, targets)
custom_loss = custom_criterion(outputs, targets)
print ('official loss: ', loss.item())
print ('custom loss:   ', custom_loss.item())
