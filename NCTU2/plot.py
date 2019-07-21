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
            if file == '.DS_Store':
                continue
            if '.png' in file:
                continue
            with open(file, encoding="utf8", errors='ignore', newline='') as csvFile:
                #lines = len(open(file) .readlines()) 
                label = file.replace('.csv','')
                data = np.zeros((epoch,2))
                rows = csv.reader(csvFile, delimiter=',')
                index = 0
                #print(file)
                for row in rows:
                    data[index][0] = row[0]
                    data[index][1] = row[1]
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


