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


    print(Criteria['Learning_rate'])
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


fig1 = plt.figure(figsize=(10,12))
os.chdir(Criteria['current_path'])
model_name = Criteria['model'][0]
plot.PlotComp(path = './'+ model_name, model_name = model_name, epoch = Criteria['Epochs'],figure = fig1)
#fig1.savefig(model_name + '.png')
plt.show()

'''
fig2 = plt.figure(figsize=(10,12))
os.chdir(Criteria['current_path'])
model_name = Criteria['model'][1]
plot.PlotComp(path = './'+ model_name, model_name = model_name, epoch = Criteria['Epochs'],figure = fig2)
fig2.savefig(model_name + '.png')

'''







