# standard library
import os
import csv
import sys
import argparse
from multiprocessing import Pool

# optional library
import pandas as pd
from gensim.models import Word2Vec

# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Preprocess():
    def __init__(self, data_dir, label_dir):
        self.embed_dim = 20
        self.seq_len = 3
        self.wndw_size = 100
        self.word_cnt = 0
        self.save_name = 'word2vec'
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.label = []

        os.chdir(data_dir)
        with open('train.txt','r') as f:
            self.data = f.readlines()
            for line in iter(f):
                line = line.split( )
                for i in range(len(line)):
                    self.label.append(line[i])
        #self.data

    def tokenize(self, sentence):
        """ Use jieba to tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            tokens (list of str): List of tokens in a sentence.
        """
        # TODO
        tokens = []
        return tokens

    def get_embedding(self, load=False):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load(self.save_name)
        else:
            embed = Word2Vec(self.data, size=self.embed_dim, window=self.wndw_size, min_count=self.word_cnt, iter=16, workers=2)
            embed.save(self.save_name)
        #print("相似詞前 100 排序")
        #res = embed.most_similar('a',topn = 10)
        #print(res)
        #y1 = embed.similarity("abandon", "abandons")
        #print(y1)
        #bandon abandons 
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        for i, word in enumerate(embed.wv.vocab):
            print('=== get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['魯'] = 1 
            #e.g. self.index2word[1] = '魯'
            #e.g. self.vectors[1] = '魯' vector
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        self.vectors = torch.tensor(self.vectors)
        # Add special tokens
        self.add_embedding(self.pad)
        self.add_embedding(self.unk)
        print("=== total words: {}".format(len(self.vectors)))
        return self.vectors

    def add_embedding(self, word):
        # Add random uniform vector
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors = torch.cat([self.vectors, vector], 0)

    def get_indices(self,test=False):
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
        all_indices = []
        # Use tokenized data
        for i, sentence in enumerate(self.data):
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
                # if word in word2index append word index into sentence_indices
                # if word not in word2index append unk index into sentence_indices
                # TODO
                #print(word)
                sentence_indices.append(self.word2index[word])
            # pad all sentence to fixed length
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            all_indices.append(sentence_indices)
        if test:
            return torch.LongTensor(all_indices)         
        else:
            return torch.LongTensor(all_indices), torch.LongTensor(self.label) 
path = './lab3'
pro = Preprocess(path, path)
print(pro.data)
#vector = pro.get_embedding()
#index, label = indices.get_indices() 
#print(index[8])
#print(labels)

