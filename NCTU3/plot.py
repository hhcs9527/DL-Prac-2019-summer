import os
import shutil

file = 'normal_BLEU.csv'

def splitter(tuple):
    return tuple[1].split(',') 

def file_finder():

with open(file, encoding="utf8", errors='ignore', newline='') as csvFile:
    for i in enumerate(csvFile):
        index = splitter(i)
        print(index)