import encoder
import decoder
import torch
import torch.nn as nn
from torch import optim
from prepaer_data import Char2Dict
from encoder import EncoderRNN
from decoder import DecoderRNN 
from dataloader import Data

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
        self.Decoder = DecoderRNN(hidden_size + cond_embed_size, output_size).to(device)


    def loss_sum(self, loss, mu, logvar):
        KLD = -0.5 * torch.sum(1 + 2*logvar - mu.pow(2) - logvar.exp() * logvar.exp())
        return loss + KLD
  


# Setup variable
    def optimizer_setup(self):
        encoder_optimizer = optim.SGD(self.Encoder.parameters(), lr = self.learning_rate)
        decoder_optimizer = optim.SGD(self.Decoder.parameters(), lr = self.learning_rate)

        return encoder_optimizer, decoder_optimizer 


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



# Training here
    def train(self):
        Encoder_optimizer, Decoder_optimizer  = self.optimizer_setup()
        for ep in range(self.epoch):
            overall_loss = 0
        # Training with different tense
            for c in range(4):
                data = self.get_train_set(c)
                for i in range(len(data)):

                    train_case = data[i]
                    Encoder_optimizer.zero_grad()
                    Decoder_optimizer.zero_grad()
                    self.Encoder.train()
                    self.Decoder.train()

                    # encode
                    Encoder_output, mu, logvar = encoder.DoEncode(self.Encoder, self.hidden_size, 
                                                                self.cond_embed_size, train_case[0], train_case[2])

                    # decode
                    predict_word, loss = decoder.DoDecode(self.Decoder, self.hidden_size, self.cond_embed_size, 
                                                    self.output_size, Encoder_output, 
                                                    condition = train_case[2], criterion = self.criterion, 
                                                    Training = True, decoding_word = train_case[1])
                    
                    total_loss = self.loss_sum(loss, mu, logvar)                                
                    (total_loss).backward()
                    overall_loss += total_loss

                    Encoder_optimizer.step()
                    Decoder_optimizer.step()

            total_len = (len(self.get_train_set(0)) + len(self.get_train_set(1)) + len(self.get_train_set(2)) + len(self.get_train_set(3)))
            msg = '# Epoch : {}, Avg_loss = {}'.format(ep, overall_loss/total_len)
            print(msg)

            torch.save(self.Encoder.state_dict(), 'Encoder.pt')
            torch.save(self.Decoder.state_dict(), 'Decoder.pt')



    def test(self):
        self.Encoder.load_state_dict(torch.load('Encoder.pt'))
        self.Decoder.load_state_dict(torch.load('Decoder.pt'))
        self.Encoder.eval()
        self.Decoder.eval()
        acc = 0
        test_set = self.DataLoader.read_test_file()
        for i in range(len(test_set)):
            test_case = test_set[i]
            # encode
            Encoder_output, mu, logvar = encoder.DoEncode(self.Encoder, self.hidden_size, 
                                                        self.cond_embed_size, test_case[0], int(test_case[2]))
            # decode
            predict_word, loss = decoder.DoDecode(self.Decoder, self.hidden_size, self.cond_embed_size, 
                                            self.output_size, Encoder_output, 
                                            condition = int(test_case[2]), criterion = self.criterion, 
                                            Training = True, decoding_word = test_case[1])
            print('Given_word : {}'.format(test_case[0]))
            print('Expected_word : {}'.format(test_case[1]))
            print('Predict_word : {}'.format(predict_word))
            print(' ')

            if test_case[1] == predict_word:
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