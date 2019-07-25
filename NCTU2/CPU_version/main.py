import Process as Proc
import plot
import os 
import matplotlib.pyplot as plt
import torch
from EEGNet import EEGNet
from DeepConvNet import DeepConvNet

cur_path = os.getcwd() 
if __name__ == '__main__':
    print('Training or Testing?')
    choice = input()
    
    if (choice == 'Training'):
        ##. Hyper Parameter
        Criteria = {
            'data_path' : './lab2',
            'Batch_size' : 128,
            'Learning_rate' : 1e-2,
            'Epochs' : 3,
            'activation': ['LeakyReLU','ReLU','ELU'],
            'model' : ['EEGNet', 'DeepConvNet'],
            'current_path' : cur_path 
        }

    # Training & Testing & Collecting Data
        for model in range(2):
            for act in range(3):
                os.chdir(Criteria['current_path'])
                model_name = Criteria['model'][model]
                print(model_name)
                Proc.Process(model = model_name, 
                    activation = Criteria['activation'][act], 
                    batch_size = Criteria['Batch_size'], 
                    epoch = Criteria['Epochs'], 
                    Learning_rate = Criteria['Learning_rate'])


    # Ploting
        for model in range(2):
            os.chdir(Criteria['current_path'])
            model_name = Criteria['model'][model]
            if model == 0:
                fig1 = plt.figure(figsize=(10,12))
                if os.path.isdir(model_name + '.png'):
                    os.remove(model_name + '.png')
                plot.PlotComp(path = './'+ model_name, model_name = model_name, epoch = Criteria['Epochs'],figure = fig1)
                fig1.savefig(model_name + '.png')
            else:
                fig2 = plt.figure(figsize=(10,12))
                if os.path.isdir(model_name + '.png'):
                    os.remove(model_name + '.png')
                os.chdir(Criteria['current_path'])
                plot.PlotComp(path = './'+ model_name, model_name = model_name, epoch = Criteria['Epochs'],figure = fig2)
                fig2.savefig(model_name + '.png')
        plt.show()

    # Show testing result
    else:
        ##. Hyper Parameter
        Options = {
            'data_path' : './lab2',
            'Batch_size' : 128,
            'Learning_rate' : 1e-2,
            'Epochs' : 3,
            'activation': ['LeakyReLU','ReLU','ELU'],
            'model' : ['EEGNet', 'DeepConvNet'],
            'current_path' : cur_path
        }
        print('Testing which model? With what activation fuction LeakyReLU(0),ReLU(1),ELU(2)?')
        model = input()
        activation_function =  int(input())
        if (model == 'EEGNet'):
            model_name = Options['model'][0]
            activation = Options['activation'][activation_function]
            #model = EEGNet(activation).cpu()
        else:
            model_name = Options['model'][1]
            activation = Options['activation'][activation_function]

        Proc.Test_model(Run = model_name , act = activation, batch_size = Options['Batch_size'], Learning_rate = Options['Learning_rate'])





