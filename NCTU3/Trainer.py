import encoder
import decoder
import torch
import torch.nn as nn
from torch import optim
from prepaer_data import Char2Dict
from encoder import EncoderRNN
from decoder import DecoderRNN 
from dataloader import Data
from CVAEmodel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder:
    def __init__(self, hidden_size, cond_embed_size, output_size, target_path, criterion, epoch, train_or_not, lr):
# initialize variable
        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size
        self.output_size = output_size
        self.target_path = target_path
        self.criterion = criterion
        self.train_or_not = train_or_not
        self.epoch = epoch
        self.learning_rate = lr
        
        
# initialize using class
        self.C2D = Char2Dict(hidden_size, cond_embed_size)
        self.DataLoader = Data(target_path)
        self.Encoder = EncoderRNN(hidden_size, cond_embed_size).to(device)
        self.Decoder = DecoderRNN(hidden_size, output_size).to(device)
        self.CVAE = CVAE(encoder = self.Encoder, decoder = self.Decoder, 
                            hidden_size = self.hidden_size, cond_embed_size = self.cond_embed_size, 
                            C2D = self.C2D, Train = self.train_or_not)



    def loss_sum(self, loss, Encoder_output):
        mu = self.CVAE.linearmu(Encoder_output.cpu()).to(device)
        logvar = self.CVAE.linearlogvar(Encoder_output.cpu()).to(device)
        KLD = (-0.5) * torch.sum(1 + 2*logvar - mu.pow(2) - logvar.exp().pow(2))
        return loss + KLD
  


# Setup variable
    def optimizer_setup(self):
        CVAE_optimizer = optim.SGD(self.Decoder.parameters(), lr = self.learning_rate, momentum = 0.9)

        return CVAE_optimizer


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



# Training 
    def train(self):
        CVAE_optimizer  = self.optimizer_setup()
        self.CVAE.apply(init_weights)
        for ep in range(self.epoch):
            overall_loss = 0
        # Training with different tense
            for c in range(4):
                data = self.get_train_set(c)
                for i in range(len(data)):

                    CVAE_case = data[i]
                    CVAE_optimizer.zero_grad()
                    self.CVAE.train()

                    # encode & decode
                    predict_word, decoded_word, pred_word, Encoder_output = self.CVAE(CVAE_case)

                    loss = self.criterion(pred_word, decoded_word)
                    total_loss = self.loss_sum(loss, Encoder_output)                                
                    (total_loss).backward()
                    overall_loss += total_loss

                    CVAE_optimizer.step()

            total_len = (len(self.get_train_set(0)) + len(self.get_train_set(1)) + len(self.get_train_set(2)) + len(self.get_train_set(3)))
            msg = '# Epoch : {}, Avg_loss = {}'.format(ep+1, overall_loss/total_len)
            print(msg)

            torch.save(self.CVAE.state_dict(), 'CVAE.pt')
            print('we save model !! ')




    def test(self):
        self.CVAE.load_state_dict(torch.load('CVAE.pt'))
        self.CVAE.eval()
        print('hello')
        acc = 0

        test_set = self.DataLoader.read_test_file()
        for i in range(len(test_set)):
            CVAE_case = test_set[i]
            predict_word, decoded_word, pred_word, Encoder_output = self.CVAE(CVAE_case)

            print('Given_word : {}'.format(CVAE_case[0]))
            print('Expected_word : {}'.format(CVAE_case[1]))
            print('Predict_word : {}'.format(predict_word))
            print(' ')

            if CVAE_case[1] == predict_word:
                acc += 1

        print('Test accuracy : {} %'.format(acc/len(test_set)*100))


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