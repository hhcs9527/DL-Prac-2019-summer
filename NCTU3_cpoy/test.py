#################################################################
# This .py provide the training part can be used in this lab
#################################################################
import torch
import torch.nn as nn
from torch import optim
import time
import math
import string
from random import randrange
import random
import Function as F
import os


SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.9
empty_input_ratio = 0.1
KLD_weight = 0
LR = 0.05
MAX_LENGTH = 1
hidden_size,vocab_size  = 256, 28


# Training Part
# Preparing Training Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair)
    return (input_tensor)

def return_word(char, input_lang):
    return input_lang.index2word[char.item()]

def embed(char,input_size, hidden_size):
    embedding = nn.Embedding(input_size, hidden_size)
    return embedding(char).view(1, 1, -1)

def get_linear(input, input_size, output_size):
    linear = nn.Linear(input_size, output_size)
    return(linear(input))

def test(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, cond ,input_lang, max_length=MAX_LENGTH):
    # initialize the hidden with the condition
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_tensor)
    #print(cond)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    #embed = nn.Embedding(4,10)
    cond = embed(torch.tensor(cond, dtype = torch.long), 4, 10)
    encoder_hidden = torch.cat((encoder_hidden,cond),2)
    loss = 0

    input_word = ""
    predict_word = ""
    # embedding to another space
    #----------sequence to sequence part for encoder----------#
    for i in range(input_length):
        embed_input_tensor = embed(input_tensor[i], vocab_size, hidden_size)
        encoder_output, encoder_hidden, mu, logvar = encoder(embed_input_tensor, encoder_hidden, cond)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = get_linear(torch.cat((encoder_output, cond), dim = 2), hidden_size + 10, hidden_size + 10)

    # Testing 沒有 Teacher forcing
    # Embed SOS, since condition and the word should be in the same dimension
    embed_decoder_input = embed(decoder_input, vocab_size, hidden_size + 10)
    decoder_output, decoder_hidden, decoder_predict = decoder(embed_decoder_input, decoder_hidden)

    for di in range(20):
        embed_decoder_input = embed(decoder_output, vocab_size, hidden_size + 10)
        #print(embed_decoder_input)
        decoder_output, decoder_hidden, decoder_predict = decoder(embed_decoder_input, decoder_hidden)
        predict_word += return_word(decoder_output, input_lang)
        if decoder_output.item() == 1:
            break
    predict_word = predict_word.strip('EOS')

    return loss, input_word, predict_word, mu, logvar



def go_test(encoder, decoder, learning_rate=0.001):
    input_lang, char, data = F.prepareData('train','test_data')

    Condition = {0:'present', 1:'third_person', 2:'present_progressive', 3:'simple_past'}

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.8)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.8)
    criterion = nn.CrossEntropyLoss()
    os.chdir('../')

    last_loss = 0
    word = []
    tense = []

    print(data)

    # Reclassify the data with class/tense
    for i in range(len(data)):
        word.append(data[i][0])
        tense.append(data[i][2])
    total_data = []
    total_data.append(word)
    total_data.append(tense)



    for i in range(len(word)):
        training_pairs = tensorsFromPair(input_lang,word[i])
        cond = torch.tensor(int(tense[i]), dtype = torch.long)
        loss, input_word, predict_word, mu, logvar = test(training_pairs, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion, cond = cond, input_lang = input_lang)

        print('# Case : ', i + 1)
        print('input word : {}, condition is : {}'.format(data[i][0], Condition[int(tense[i])]))
        print('expected word : {}\npredict word : {} '.format(data[i][1], predict_word))

    print(' ')

    print('Guassin noise generation word :')
    for i in range(10):
        reparameterize = torch.randn_like(torch.exp(0.5*logvar)) + mu
        generate_set = []
        for c in range(4):
            #for di in range(20):
            decoder_input = torch.tensor([[SOS_token]], dtype = torch.long, device=device)
            embed_decoder_input = embed(decoder_input, vocab_size, hidden_size + 10)
            decoder_output = decoder_input

            for k in range(20):#(when model power enough)
                cond = embed(torch.tensor(c, dtype = torch.long),4, 10)
                decoder_hidden = torch.cat((reparameterize, cond), dim = 2)

                decoder_output, decoder_hidden, decoder_predict = decoder(embed_decoder_input, decoder_hidden)
                embed_decoder_input = embed(decoder_output, vocab_size, hidden_size + 10)
                #decoder_input = decoder_output  # detach from history as input  # Teacher forcing
                if return_word(decoder_output, input_lang) == 'EOS':
                    break
                #print(return_word(decoder_output, input_lang))
                predict_word += return_word(decoder_output, input_lang)

            #predict_word = predict_word.strip('EOS')
            generate_set.append(predict_word)
        print('# Case {} : {}'.format(i+1, generate_set))




