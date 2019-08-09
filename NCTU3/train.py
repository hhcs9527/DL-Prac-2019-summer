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


def train(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, cond ,input_lang, max_length=MAX_LENGTH):
    # initialize the hidden with the condition
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
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
        encoder_output, encoder_hidden, mu, logvar = encoder(input_tensor[i], encoder_hidden, cond)
        input_word += return_word(input_tensor[i], input_lang)
    input_word = input_word.strip('EOS')


    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = torch.cat((encoder_hidden, cond), dim = 2)


    use_teacher_forcing = True #if random.random() < teacher_forcing_ratio else False
    

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        #produce = []
        for di in range(input_length):
############### change here ##################################

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_input = input_tensor[di]  # Teacher forcing
            loss += criterion(decoder_output, torch.tensor([input_tensor[di]], dtype = torch.float32))
            predict_word += return_word(decoder_output, input_lang)
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
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), input_word, predict_word #/ input_tensor



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



def trainIters(encoder, decoder, n_iters, epoch, print_every=5, plot_every=10, learning_rate=0.005):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    input_lang, lines, pairs = F.prepareData('train')

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)



    for i in range(epoch):
        #Choose index instead of real word
        #choose_index = [tensorsFromPair(condition_lang,randrange(len(lines))%4) for i in range(n_iters)]
        for j in range(int(len(lines)/n_iters)):
            choose_index = [randrange(len(lines)) for i in range(n_iters)]
            training_pairs = [tensorsFromPair(input_lang,lines[i]) for i in choose_index]
            criterion = nn.MSELoss()#nn.CrossEntropyLoss()#nn.MSELoss()
            #criterion = nn.CrossEntropyLoss()
            #criterion = nn.BCELoss()
            for iter in range(1, n_iters + 1):
                print_loss_total = 0
                plot_loss_total = 0
                training_pair = training_pairs[iter - 1]
                cond = torch.tensor(choose_index[iter - 1]%4, dtype = torch.long)
                input_tensor = training_pair
                #print(input_tensor)
                loss, input_word, predict_word = train(input_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion, cond = cond, input_lang = input_lang)

                print_loss_total += loss
                plot_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    KLD_weight = (iter % 10 /10) 
                    #'{} is {} years old.'.format("James", 5)
                    #msg = '{} ({} {}) {:.3f}'.format(timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg)
                    #print('{} ({} {}) {:.4f}'.format(timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))
                    msg = '# epoch : {} iter : {:.2f}%  Loss_avg : {:.5f}'.format(i,iter / n_iters * 100, print_loss_avg)
                    print_loss_total = 0
                    plot_loss_total = 0
                    print(msg)
                    print('input_word is : ',input_word)
                    print('predict_wo is : ',predict_word)
