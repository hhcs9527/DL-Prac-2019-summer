import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os

def read_bci_data(path, batch_size):
    print('Read file .... ')
    os.chdir(path)
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)


# numpy array -> Tensor -> TensorDataset -> DataLoader

    train_data = torch.Tensor(train_data).cuda()
    train_label = torch.LongTensor(train_label).cuda()
    test_data = torch.Tensor(test_data).cuda()
    test_label = torch.LongTensor(test_label).cuda()

    train_set = TensorDataset(train_data, train_label)
    test_set = TensorDataset(test_data, test_label)

    # num_workers = 0 must be zero, or can't run
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 0)

    return train_loader, test_loader, train_label.size()[0], test_label.size()[0]
