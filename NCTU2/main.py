import Process as Proc
import plot
import os 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    ##. Hyper Parameter
    Criteria = {
        'data_path' : './lab2',
        'Batch_size' : 128,
        'Learning_rate' : 1e-2,
        'Epochs' : 150,
        'activation': ['LeakyReLU','ReLU','ELU'],
        'model' : ['EEGNet', 'DeepConvNet'],
        'current_path' : '/tmp/DL-Prac-2019-summer/NCTU2'
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









