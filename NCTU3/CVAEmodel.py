import torch
import torch.nn as nn

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

class CVAE(nn.Module):
    def __init__(self, encoder, decoder, hidden_size, cond_embed_size, C2D, Train):
        super(CVAE, self).__init__()     
        #super(DecoderRNN, self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        self.C2D = C2D

        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size
        self.Train = Train

        self.SOS_token = 0
        self.EOS_token = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.linearmu = nn.Linear(hidden_size, hidden_size) 
        self.linearlogvar = nn.Linear(hidden_size, hidden_size) 
    


    # specific for the decoder
    def get_linear(self, Encoder_output, condition):
        linear = nn.Linear(self.hidden_size + self.cond_embed_size, self.hidden_size)
        Encoder_output = torch.cat((Encoder_output, self.C2D.embed_cond(condition)), dim = 2)
        return linear(Encoder_output.cpu())


    def reparameterize(self, Encoder_output):
        mu = self.linearmu(Encoder_output.cpu()).to(self.device)
        logvar = self.linearlogvar(Encoder_output.cpu()).to(self.device)
        std = torch.sqrt(torch.exp(logvar))
        sample = torch.randn(std.size()[0], std.size()[1], std.size()[2], device = self.device)
        return std.mul(sample).add(mu)



    def Encode_word(self):
        encode_word = self.C2D.Word2Tensor(self.CVAE_case[0])
        Encoder_hidden = torch.cat((self.Encoder.initHidden(), self.C2D.embed_cond(int(self.CVAE_case[2]))), dim = 2)

        for i in range(encode_word.size()[0]):
            Encoder_input = self.C2D.embed_char(encode_word[i]) # before embed, must be in cpu
            Encoder_output, Encoder_hidden = self.Encoder(Encoder_input, Encoder_hidden)        
        return Encoder_hidden



    def Decode_word(self, Encoder_output):
    # initialize decoder
        decode_word = self.C2D.Word2Tensor(self.CVAE_case[1])

    # initialize variable
        loss = 0
        predict_word = ""
        TeacherForcing = True
        pred_word = torch.zeros(decode_word.size()[0], 28, dtype = torch.float).to(self.device)
        decoded_word = torch.zeros(decode_word.size()[0], dtype = torch.long).to(self.device)

    # sequence to sequence part for decoder
        if self.Train:
            Decoder_hidden = self.get_linear(self.reparameterize(Encoder_output), int(self.CVAE_case[2])).to(self.device)
            Decoder_input = self.C2D.embed_char(torch.tensor(self.SOS_token, dtype = torch.long)).to(self.device)
            for i in range(decode_word.size()[0]):
                Decoder_output, Decoder_hidden, Decoder_predict = self.Decoder(Decoder_input, Decoder_hidden) 
                if TeacherForcing:
                    Decoder_input = self.C2D.embed_char(decode_word[i])
                else:
                    if decode_word[i+1].item() == self.EOS_token:
                        break
                    Decoder_input = self.C2D.embed_char(Decoder_output)
                
                #loss += criterion(Decoder_predict, decode_word[i].to(device))
                pred_word[i] = Decoder_predict
                decoded_word[i] = decode_word[i]

                predict_word += self.C2D.return_word(Decoder_output)

        return predict_word, decoded_word.to(self.device), pred_word, Encoder_output

        
    def forward(self, CVAE_case):
        self.CVAE_case = CVAE_case
        Encoder_output = self.Encode_word()
        return self.Decode_word(Encoder_output)