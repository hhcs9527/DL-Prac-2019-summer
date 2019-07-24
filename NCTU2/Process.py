import dataloader 
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from EEGNet import EEGNet
from DeepConvNet import DeepConvNet
import os



def Process(model, activation, batch_size, epoch, Learning_rate):
    Options = {
        'data_path' : './lab2',
        'model' : ['EEGNet', 'DeepConvNet']
    }

# Model initialization
    Run = model #int(input())
    if (Run == 'EEGNet'):
        model_name = Options['model'][0]
        activation_function = activation
        model = EEGNet(activation_function).cpu()
    else:
        model_name = Options['model'][1]
        activation_function = activation
        model = DeepConvNet(activation_function).cpu()

# Hyper Parameter setting 
    Batch_size = batch_size
    num_epochs = epoch
    lr = Learning_rate
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# plot file setting
    train_record = model_name + '_' + activation_function + '_train.csv'
    test_record = model_name + '_' + activation_function + '_test.csv'
    model_weight = model_name + '_' + activation_function + '.pkl'#'.pt'
    print('Dealing with ' + train_record)
    print('Dealing with ' + test_record)

# Read file, get the data which are wrapped into dataloader
    train_loader, test_loader, train_size, test_size = dataloader.read_bci_data(Options['data_path'],Batch_size)
    os.chdir('../'+ model_name)
    # Training & Testing & 寫入檔案
    with open(train_record, 'w', newline = '') as csvfile:
        write_train = csv.writer(csvfile)
        with open(test_record, 'w', newline = '') as csvFile:
            write_test = csv.writer(csvFile)    

            for epoch in range(num_epochs):
            # Training 
                train_acc = 0.0
                train_loss = 0.0
                model.train()
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()

                    train_pred = model(data[0].cpu())
                    batch_loss = loss(train_pred, data[1].cpu())
                    batch_loss.backward()
                    optimizer.step()

                    train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis = 1) == data[1].numpy())
                    train_loss += batch_loss.item()
                train_acc = train_acc/ train_size * 100.0
                print('# {} epoch, Train Accuracy : {:.2f}%'.format(epoch+1, train_acc))
                write_train.writerow([epoch, train_acc])

            # Testing 
                test_acc = 0.0
                test_loss = 0.0
                model.eval()
                for i, data in enumerate(test_loader):
                    test_pred = model(data[0].cpu())
                    batch_loss = loss(test_pred, data[1].cpu())

                    test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis = 1) == data[1].numpy())
                    test_loss += batch_loss.item()
                test_acc = test_acc/ test_size * 100.0
                print('# {} epoch, Test Accuracy : {:.2f}%'.format(epoch+1, test_acc))
                write_test.writerow([epoch, test_acc])
    torch.save(model.state_dict(), model_weight)

def Test_model(Run, act, batch_size, Learning_rate):
    Options = {
        'data_path' : '../lab2',
        'model' : ['EEGNet', 'DeepConvNet']

    }
    if (Run == 'EEGNet'):
        model_name = Run
        activation = act
        model = EEGNet(activation).cpu()
    else:
        model_name = Run
        activation = act
        model = DeepConvNet(activation).cpu()
    Path = './' + model_name 
    os.chdir(Path)
    PATH = model_name + '_' + activation + '.pkl'#'.pt'
    model.load_state_dict(torch.load(PATH))
    model.eval()
# Hyper Parameter setting 
    Batch_size = batch_size
    lr = Learning_rate
    loss = nn.CrossEntropyLoss()


    train_loader, test_loader, train_size, test_size = dataloader.read_bci_data(Options['data_path'],Batch_size)
    # Testing 
    test_acc = 0.0 
    test_loss = 0.0
    model.eval()
    for i, data in enumerate(test_loader):
        test_pred = model(data[0].cpu())
        batch_loss = loss(test_pred, data[1].cpu())
        test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis = 1) == data[1].numpy())
        test_loss += batch_loss.item()
    test_acc = test_acc/ test_size * 100.0
    print('Test Accuracy : {:.2f}%'.format(test_acc))


