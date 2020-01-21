import torch
import torch.nn as nn
from prepaer_data import Char2Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_embed_size, hidden_size, cond_embed_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size

        self.gru = nn.GRU(input_embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size) 
        self.linearmu = nn.Linear(hidden_size, hidden_size) 
        self.linearlogvar = nn.Linear(hidden_size, hidden_size) 
        

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden


    def initHidden(self):
        return torch.randn(1, 1, self.hidden_size - self.cond_embed_size, device = device)



if __name__ == '__main__':
    E = EncoderRNN(256, 10)
    E.initHidden()