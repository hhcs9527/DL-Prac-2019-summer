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
import Function as F
import os


SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 1.0
empty_input_ratio = 0.1
KLD_weight = 0
LR = 0.05
MAX_LENGTH = 1


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

def test(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, cond ,input_lang, max_length=MAX_LENGTH):
    # initialize the hidden with the condition
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_tensor)
    #print(cond)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    embed = nn.Embedding(4,10)
    cond = embed(torch.tensor(cond, dtype = torch.long)).view(1, 1, -1)
    encoder_hidden = torch.cat((encoder_hidden,cond),2)
    loss = 0

    input_word = ""
    predict_word = ""
    # embedding to another space
    #----------sequence to sequence part for encoder----------#
    for i in range(input_length):
        #print(input_tensor[i])
        encoder_output, encoder_hidden, mu, logvar = encoder(input_tensor[i], encoder_hidden, cond)
        input_word += return_word(input_tensor[i], input_lang)
    input_word = input_word.strip('EOS')


    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = torch.cat((encoder_output, cond), dim = 2)


    use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False
    

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        #produce = []
        for di in range(input_length):
############### change here ##################################

            decoder_output, decoder_hidden, decoder_predict = decoder(decoder_input, decoder_hidden)
            decoder_input = input_tensor[di]  # Teacher forcing
            loss += criterion(decoder_predict, input_tensor[di])#torch.tensor([input_tensor[di]], dtype = torch.float32))
            predict_word += return_word(decoder_output, input_lang)
        predict_word = predict_word.strip('EOS')
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
    KLD = -0.5 * torch.sum(1 + 2*logvar - mu.pow(2) - logvar.exp() * logvar.exp())
    loss += KLD


    return loss, input_word, predict_word, mu, logvar



def go_test(encoder, decoder, learning_rate=0.005):
    input_lang, char, data = F.prepareData('train','test_data')

    Condition = {0:'present', 1:'third_person', 2:'present_progressive', 3:'simple_past'}

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.7)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.7)
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
        print('input word : {}, condition is : {}'.format(input_word, Condition[int(tense[i])]))
        print('expected word : {}\npredict word : {} '.format(data[i][1], predict_word))

    print(' ')

    print('Guassin noise generation word :')
    for i in range(10):
        reparameterize = torch.randn_like(torch.exp(0.5*logvar)) + mu
        generate_set = []
        for c in range(4):
            #for di in range(20):
            decoder_input = torch.tensor([[SOS_token]], dtype = torch.long, device=device)
            decoder_output = decoder_input

            for k in range(20):#(when model power enough)
                embed = nn.Embedding(4,10)
                cond = embed(torch.tensor(c, dtype = torch.long)).view(1, 1, -1)
                decoder_hidden = torch.cat((reparameterize, cond), dim = 2)
                decoder_output, decoder_hidden, decoder_predict = decoder(decoder_input, decoder_hidden)
                decoder_input = decoder_output  # detach from history as input  # Teacher forcing
                if return_word(decoder_output, input_lang) == 'EOS':
                    break
                #print(return_word(decoder_output, input_lang))
                predict_word += return_word(decoder_output, input_lang)

            predict_word = predict_word.strip('EOS')
            generate_set.append(predict_word)
        print('# Case {} : {}'.format(i+1, generate_set))




