import encoder
import decoder
import torch
import torch.nn as nn
import os
import csv
from torch import optim
from prepaer_data import Char2Dict
from encoder import EncoderRNN
from decoder import DecoderRNN 
from dataloader import Data
from CVAEmodel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder:
    def __init__(self, hidden_size, cond_embed_size, output_size, target_path, criterion, epoch, train_or_not, lr, input_embed_size, teacher_forcing_ratio, ratio_kind):
# initialize variable
        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size
        self.output_size = output_size
        self.target_path = target_path
        self.criterion = criterion
        self.train_or_not = train_or_not
        self.epoch = epoch
        self.learning_rate = lr
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.input_embed_size = input_embed_size
        self.ratio_kind = ratio_kind
        filename = self.get_bleuname()
        self.weight_name = 'CVAE_'+ filename.replace('.csv', '') +'.pt'
        
# initialize using class
        self.C2D = Char2Dict(cond_embed_size)
        self.DataLoader = Data(target_path)
        self.Encoder = EncoderRNN(input_embed_size, hidden_size, cond_embed_size).to(device)
        self.Decoder = DecoderRNN(input_embed_size, hidden_size, output_size).to(device)
        self.CVAE = CVAE(encoder = self.Encoder, decoder = self.Decoder, 
                            hidden_size = self.hidden_size, cond_embed_size = self.cond_embed_size, 
                            C2D = self.C2D, Train = self.train_or_not, output_size = self.output_size, 
                            teacher_forcing_ratio = self.teacher_forcing_ratio, input_embed_size = self.input_embed_size)
        self.CVAE_optimizer = optim.SGD(self.CVAE.parameters(), lr = self.learning_rate, momentum = 0.9)


    def loss_sum(self, loss, Encoder_output, ep):
        mu = self.CVAE.linearmu(Encoder_output.cpu()).to(device)
        logvar = self.CVAE.linearlogvar(Encoder_output.cpu()).to(device)
        KLD = (-0.5) * torch.sum(1 + 2*logvar - mu.pow(2) - logvar.exp().pow(2))
    # normal
        if self.ratio_kind == 0:
            return loss + KLD
    # cyckicak
        elif self.ratio_kind == 1:
            if ep % 5 == 0:
                return loss
            elif ep % 5 <= 3:
                return loss + (ep/3)*KLD  
            elif  ep % 5 > 3:
                return loss + KLD
    # monotonic
        elif self.ratio_kind == 2:
            if ep <= 5:
                return loss + (ep/5)*KLD  
            else:
                return loss + KLD
        else:
            return loss + 0.002*KLD

    def get_bleuname(self):
    # normal
        if self.ratio_kind == 0:
            return 'normal_BLEU.csv'
    # cyckicak
        elif self.ratio_kind == 1:
            return'cyckicak_BLEU.csv'
    # monotonic
        elif self.ratio_kind == 2:
            return 'monotonic_BLEU.csv'    
        else:
            return 'CVAE_train.csv'



    def get_train_set(self, c):
        present, third_person, present_progressive, simple_past = self.DataLoader.seperate_tense()
        if c == 0:
            data = present
        elif c == 1:
            data = third_person
        elif c == 2:
            data = present_progressive
        else:
            data = simple_past

        return data


    def writefile(self):
        filename = self.get_bleuname()
        write_list = self.train()
        with open(filename, 'w' ,newline = '') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(write_list)):
                writer.writerow(write_list[i])

# Training 
    def train(self):
        write_list = []
        Belu = Autoencoder(hidden_size = 256, cond_embed_size = 8, output_size = 28, 
               target_path = './lab3/test_data.txt', criterion = nn.CrossEntropyLoss(), 
               epoch = 30, train_or_not = False, lr = 0.001, input_embed_size = 64, 
               teacher_forcing_ratio = 1, ratio_kind = self.ratio_kind)

        self.CVAE.train()

        for ep in range(self.epoch):
            overall_loss = 0
            acc = 0
        # Training with different tense
            for c in range(4):
                data = self.get_train_set(c)
                for i in range(len(data)):

                    CVAE_case = data[i]
                    self.CVAE_optimizer.zero_grad()

                    # encode & decode
                    predict_word, decoded_word, pred_word, Encoder_output = self.CVAE(CVAE_case)

                    loss = self.criterion(pred_word, decoded_word)
                    total_loss = self.loss_sum(loss, Encoder_output, ep)                                
                    (total_loss).backward()
                    overall_loss += total_loss

                    self.CVAE_optimizer.step()                        
                    if predict_word == CVAE_case[1]:
                        acc += 1
                
            total_len = (len(self.get_train_set(0)) + len(self.get_train_set(1)) + len(self.get_train_set(2)) + len(self.get_train_set(3)))
            msg = '# Epoch : {}, Avg_loss = {}'.format(ep+1, overall_loss/total_len)
            print(msg)
            acc_msg = 'Accuracy : {:.3f}%'.format(acc/total_len*100)
            print(acc_msg)
            torch.save(self.CVAE.state_dict(), self.weight_name)
            score = Belu.test()

            write_list.append([ep, score])
        
        return write_list






    def test(self):
        self.CVAE.load_state_dict(torch.load(self.weight_name))
        self.CVAE.eval()
        acc = 0
        bleu_score = 0
        test_set = self.DataLoader.read_test_file()
        for i in range(len(test_set)):
            CVAE_case = test_set[i]
            predict_word = self.CVAE(CVAE_case)

            print('Given_word : {}'.format(CVAE_case[0]))
            print('Expected_word : {}'.format(CVAE_case[1]))
            print('Predict_word : {}'.format(predict_word))
            print(' ')

            if CVAE_case[1] == predict_word:
                acc += 1
            bleu_score += self.CVAE.compute_bleu(predict_word, CVAE_case[1])
        print('Test accuracy : {} %'.format(acc/len(test_set)*100))
        print('Test_Avg belu score : ', bleu_score/len(test_set))
        print()
        return bleu_score


    def generate(self):
        self.CVAE.load_state_dict(torch.load('CVAE.pt'))
        self.CVAE.eval()
        for i in range(10):
            noise = torch.randn(1,1,self.hidden_size).to(device)
            test_case1, test_case2, test_case3, test_case4 = self.CVAE.Decode_test(noise, 0), self.CVAE.Decode_test(noise, 1), self.CVAE.Decode_test(noise, 2), self.CVAE.Decode_test(noise, 3)
            print('# Case {}: [{}, {}, {}, {}]'.format(i+1, test_case1, test_case2, test_case3, test_case4))


if __name__ == '__main__':
    hidden_size, cond_embed_size, output_size = 256, 10, 28
    target_path = './lab3/train.txt'
    criterion = nn.CrossEntropyLoss()
    epoch = 5
    lr = 0.001
    train_or_not = True
    auto = Autoencoder(hidden_size, cond_embed_size, output_size, target_path, criterion, epoch, train_or_not, lr)
    auto.train()
    auto.test()