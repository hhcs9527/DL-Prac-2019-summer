import torch
import torch.nn as nn
from prepaer_data import Char2Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.softmax2 = nn.Softmax(dim = 1)
        #self.lin = nn.Linear(output_size,1)

    def forward(self, input, hidden):
        output_pred, hidden = self.gru(input, hidden)
        hidden = self.linear(hidden)
        hidden = self.relu(hidden)
        predict = self.softmax2(self.out(self.softmax2(output_pred[0])))
        output = torch.argmax(self.softmax2(predict))
        output = torch.tensor(output, dtype = torch.long)
        return output, hidden, predict

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size , device = device)


def DoDecode(Decoder, hidden_size, cond_embed_size, output_size, 
            encoder_output, condition, criterion, Training, decoding_word):
# initialize decoder
    C2D = Char2Dict(hidden_size, cond_embed_size)
    decode_word = C2D.Word2Tensor(decoding_word)

# initialize variable
    loss = 0
    predict_word = ""
    TeacherForcing = True

# sequence to sequence part for decoder
    if Training:
        for i in range(decode_word.size()[0]):
            if i == 0:
                Decoder_hidden = Decoder.initHidden()
                Decoder_hidden.data.copy_(torch.cat((encoder_output, C2D.embed_cond(condition)), dim = 2))
                Decoder_input = C2D.embed_char(torch.tensor(SOS_token, dtype = torch.long), 'decode')

            Decoder_output, Decode_hidden, Decoder_predict = Decoder(Decoder_input, Decoder_hidden) 
            Decoder_hidden = Decode_hidden
            
            if TeacherForcing:
                Decoder_input = C2D.embed_char(decode_word[i], 'decode')
            else:
                Decoder_input = Decoder_output
            
            loss += criterion(Decoder_predict, decode_word[i].to(device))
            predict_word += C2D.return_word(Decoder_output)

            if decode_word[i+1].item() == EOS_token:
                break
    return predict_word, loss
# Testing
    #else:
        #while predict != EOS_token...

if __name__ == '__main__':
    print('how you doing???')
