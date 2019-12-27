import torch
import torch.nn as nn
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.09, 0.09)



class CVAE(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, cond_embed_size, C2D, Train, output_size, teacher_forcing_ratio, input_embed_size):
        super(CVAE, self).__init__()     
        #super(DecoderRNN, self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        self.C2D = C2D

        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size
        self.Train = Train
        self.output_size = output_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.input_embed_size = input_embed_size 

        self.SOS_token = 0
        self.EOS_token = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.linearmu = nn.Linear(hidden_size, hidden_size) 
        self.linearlogvar = nn.Linear(hidden_size, hidden_size) 
    
        self.word_embedding = nn.Embedding(28, self.input_embed_size)
        self.cond_embedding = nn.Embedding(4, self.cond_embed_size)
        self.linear = nn.Linear(self.hidden_size + self.cond_embed_size, self.hidden_size)


    # specific for the decoder
    def get_linear(self, Encoder_output, condition):
        Encoder_output = torch.cat((Encoder_output, self.embed_cond(condition)), dim = 2)
        return self.linear(Encoder_output.cpu())


# Embedding char to vector / condition to vector
    def embed_char(self, char):
        return self.word_embedding(char).view(1, 1, -1).to(self.device)

    def embed_cond(self, condition):
        cond = torch.tensor(condition, dtype = torch.long)
        return self.cond_embedding(cond).view(1, 1, -1).to(self.device)


    def reparameterize(self, Encoder_output):
        mu = self.linearmu(Encoder_output.cpu()).to(self.device)
        logvar = self.linearlogvar(Encoder_output.cpu()).to(self.device)
        std = torch.sqrt(torch.exp(logvar))
        sample = torch.randn(std.size()[0], std.size()[1], std.size()[2], device = self.device)
        return std.mul(sample).add(mu)


    def compute_bleu(self, output, reference):
        cc = SmoothingFunction()
        return sentence_bleu([reference], output,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method1)


    def Encode_word(self):
        encode_word = self.C2D.Word2Tensor(self.CVAE_case[0])
        Encoder_hidden = torch.cat((self.Encoder.initHidden(), self.embed_cond(int(self.CVAE_case[2]))), dim = 2)

        for i in range(encode_word.size()[0]):
            Encoder_input = self.embed_char(encode_word[i]) # before embed, must be in cpu
            Encoder_output, Encoder_hidden = self.Encoder(Encoder_input, Encoder_hidden)        
        return Encoder_output




    def Decode_word(self, Encoder_output):
    # initialize decoder
        decode_word = self.C2D.Word2Tensor(self.CVAE_case[1])

    # initialize variable
        loss = 0
        predict_word = ""

        TeacherForcing = False
        if (np.random.rand(1) < self.teacher_forcing_ratio):
            TeacherForcing = True 
        pred_word = torch.zeros(decode_word.size()[0], self.output_size, dtype = torch.float).to(self.device)
        decoded_word = torch.zeros(decode_word.size()[0], dtype = torch.long).to(self.device)


    # sequence to sequence part for decoder
        Decoder_hidden = self.get_linear(self.reparameterize(Encoder_output), int(self.CVAE_case[2])).to(self.device)
        Decoder_input = self.embed_char(torch.tensor(self.SOS_token, dtype = torch.long)).to(self.device)
        if self.Train:
            for i in range(decode_word.size()[0]):
                Decoder_output, Decoder_hidden, Decoder_predict = self.Decoder(Decoder_input, Decoder_hidden) 
                if TeacherForcing:
                    Decoder_input = self.embed_char(decode_word[i])
                else:
                    Decoder_input = self.embed_char(Decoder_output)
                
                pred_word[i] = Decoder_predict
                decoded_word[i] = decode_word[i]
                
                if Decoder_output.item() == self.EOS_token:
                    break
                
                predict_word += self.C2D.return_word(Decoder_output)
            return predict_word, decoded_word.to(self.device), pred_word, Encoder_output

        else:
            Decoder_output, Decoder_hidden, Decoder_predict = self.Decoder(Decoder_input, Decoder_hidden) 
            len = 0
            while len < 15 and Decoder_output.item() != self.EOS_token:
                Decoder_input = self.embed_char(Decoder_output)
                predict_word += self.C2D.return_word(Decoder_output)
                Decoder_output, Decoder_hidden, Decoder_predict = self.Decoder(Decoder_input, Decoder_hidden) 
                len +=1

            return predict_word    
            
            
    def Decode_test(self, Encoder_output, cond):
        predict_word = ""
    # sequence to sequence part for decoder
        Decoder_hidden = self.get_linear(self.reparameterize(Encoder_output), int(cond)).to(self.device)
        Decoder_input = self.embed_char(torch.tensor(self.SOS_token, dtype = torch.long)).to(self.device)

        print(Decoder_input.size())
        print(Decoder_hidden.size())

        Decoder_output, Decoder_hidden, Decoder_predict = self.Decoder(Decoder_input, Decoder_hidden) 
        len = 0
        while len < 15 and Decoder_output.item() != self.EOS_token:
            Decoder_input = self.embed_char(Decoder_output)
            predict_word += self.C2D.return_word(Decoder_output)
            Decoder_output, Decoder_hidden, Decoder_predict = self.Decoder(Decoder_input, Decoder_hidden) 
            len +=1

        return predict_word  

        
    def forward(self, CVAE_case):
        self.CVAE_case = CVAE_case
        Encoder_output = self.Encode_word()
        return self.Decode_word(Encoder_output)