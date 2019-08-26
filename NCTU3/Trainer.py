import encoder
import decoder
import torch.nn as nn
from prepaer_data import Char2Dict
from encoder import EncoderRNN
from decoder import DecoderRNN 
from dataloader import Data

class Autoencoder:
    def __init__(self, hidden_size, cond_embed_size, output_size, target_path, criterion, epoch, train):
# initialize variable
        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size
        self.output_size = output_size
        self.target_path = target_path
        self.criterion = criterion
        self.train = train
        self.epoch = epoch
        
# initialize using class
        self.C2D = Char2Dict(hidden_size, cond_embed_size)
        self.DataLoader = Data(target_path)
        self.encoder = EncoderRNN(hidden_size, cond_embed_size)
        self.decoder = DecoderRNN(hidden_size + cond_embed_size, output_size)


    def train(self):
        present, third_person, present_progressive, simple_past = self.DataLoader.seperate_tense()


        test_case = present[1]


        # encode
        encoder_output, Encode_hidden, mu, logvar = encoder.DoEncode(hidden_size, 
                                                    cond_embed_size, test_case[0], test_case[2])

        # decode
        predict_word = decoder.DoDecode(hidden_size, cond_embed_size, 
                                        output_size, encoder_output, 
                                        condition = test_case[2], criterion = criterion, 
                                        Training = True, decoding_word = test_case[1])

if __name__ == '__main__':
    hidden_size, cond_embed_size, output_size = 256, 10, 28
    target_path = './lab3/train.txt'
    criterion = nn.CrossEntropyLoss()