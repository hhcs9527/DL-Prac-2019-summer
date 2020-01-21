import matplotlib.pyplot as plt
import csv
import numpy as np
from numpy import genfromtxt
import pandas as pd
import os

def PlotComp(path, model_name,epoch,figure):
    color = ['b-','g-','r-','c-','m-','y-']
    #figure = plt.axes()
    for dirPath, dirNames, fileNames in os.walk(path):
        os.chdir(path)
        index_color = 0
        for file in fileNames:
            if not '.csv' in file:
                print(file)
                continue
            with open(file, encoding="utf8", errors='ignore', newline='') as csvFile:
                #lines = len(open(file) .readlines()) 
                label = file.replace('.csv','')
                data = np.zeros((epoch,2), dtype = float)
                rows = csv.reader(csvFile, delimiter=',')
                index = 0
                #print(file)
                for row in rows:
                    #row =  [float(i) for i in row1]
                    data[index][0] = float(row[0])
                    data[index][1] = float(row[1])
                    index += 1
                plt.plot(data[:,0],data[:,1],color[index_color],label = label)
            index_color = index_color + 1
        title = 'Activation function comparison (' + model_name + ')'
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy %')
        plt.ylim(40,100) 
        plt.xlim(0, epoch) #epoch
        plt.legend()

#model_name = 'EEGNet'
#fig1 = plt.figure(figsize=(10,12))
#PlotComp(path = './'+ model_name, model_name = model_name, epoch = 3,figure = fig1)
