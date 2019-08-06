from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import os
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu



# Get data ready
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
#The number of vocabulary
vocab_size = 28
teacher_forcing_ratio = 1.0
empty_input_ratio = 0.1
KLD_weight = 1.0
LR = 0.05
MAX_LENGTH = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Label == Training itself
def readLangs(lang1):
    print("Reading lines...")
    path = '/tmp/DL-Prac-2019-summer/NCTU3/lab3'#'./lab3'
    os.chdir(path)
    lines = []
    # Read the file and split into lines
    with open('train.txt', encoding='utf-8') as f:
            for line in iter(f):
                line = line.split( )
                for i in range(len(line)):
                    lines.append(line[i])
    #lines = file.read().strip().split('\n')
    pairs = [chr(i) for i in range(97, 123)]
    # Reverse pairs, make Lang instances
    input_lang = Lang(lang1)

    return input_lang, lines, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1):
    input_lang, lines, pairs = readLangs(lang1)
    #print(pairs)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        #print('yo ' + pair[0])
        #print('yo3 ' + pair[1])
        input_lang.addSentence(pair)
    #print("Counted words:")
    #print(input_lang.name, input_lang.n_words)
    #print(output_lang.name, output_lang.n_words)
    return input_lang, lines, pairs

#input_lang, lines, pairs = prepareData('character')



# Training Part
# Preparing Training Data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair)
    return (input_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # input -> 表示原來在幾維空間 / hidden 表示要壓到幾為空間
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # For the decoder 
        self.linear = nn.Linear(hidden_size, hidden_size) 

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        # Reparprint Part -> output of the Encoder
        mu = self.linear(output)
        logvar = self.linear(output)
        std = torch.exp(0.5*logvar)
        output = mu + torch.randn_like(std)
        #print('output:',output)
        #print('hidden:',hidden)
        return output, hidden, mu, logvar


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)




class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.lin = nn.Linear(output_size,1)

    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output = self.out(output)
        output = self.softmax(self.out(output[0]))
        output = self.lin(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)







def train(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    #----------sequence to sequence part for encoder----------#
    for i in range(input_length):
        encoder_output, encoder_hidden, mu, logvar = encoder(input_tensor[i], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False
    

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        #produce = []
        for di in range(input_length):
############### change here ##################################
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_input = input_tensor[di]  # Teacher forcing
            #produce.append(decoder_output)
            #torch.tensor([input_tensor[di]], dtype = torch.float64)
            loss += criterion(decoder_output, torch.tensor([input_tensor[di]], dtype = torch.float32))
            print('decoder_output ',decoder_output)


    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(input_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break
    # Reparprint Part brings the loss of the KLD
    KLD = -0.5 + torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print('KLD is ',KLD)
    loss += KLD
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    #print(input_tensor)
    #print(loss.item())

    return loss.item() #/ input_tensor



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(encoder, decoder, n_iters, print_every=1, plot_every=10, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    input_lang, lines, pairs = prepareData('train')

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang,random.choice(lines))for i in range(n_iters)]
    criterion = nn.MSELoss()#nn.CrossEntropyLoss()#nn.MSELoss()

    for iter in range(1, n_iters + 1):
        print_loss_total = 0
        plot_loss_total = 0
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair
        #print(input_tensor)
        loss = train(input_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)

        #print(loss)
        #print(print_loss_total)
        print_loss_total += loss
        #print(print_loss_total)
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            #'{} is {} years old.'.format("James", 5)
            msg = '{} ({} {}) {:.3f}'.format(timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg)
            #print('{} ({} {}) {:.4f}'.format(timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))
            msg = 'iter : {:.2f}%  Loss_avg : {:.5f}'.format(iter / n_iters * 100, print_loss_avg)
            print(msg)







#char = embedding(training_pairs)
encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)
condition = ['present','']
trainIters(encoder1, decoder1, 3)





