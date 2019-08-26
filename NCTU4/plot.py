import matplotlib.pyplot as plt
import csv
import numpy as np
from numpy import genfromtxt
import pandas as pd
import os

def PlotComp(epoch,D_L ,G_L):
    color = ['b-','r-']
    figure = plt.figure(figsize=(10,12))
    plt.plot(D_L, color[0], label = 'D_loss')
    plt.plot(G_L, color[1], label = 'G_loss')
    title = 'Loss Comparison'
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0,8) 
    plt.xlim(0, epoch) #epoch
    plt.legend()
    plt.show()