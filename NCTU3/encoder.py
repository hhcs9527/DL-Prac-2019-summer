import torch
import torch.nn as nn
from prepaer_data import Char2Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, cond_embed_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size) 
        self.linearmu = nn.Linear(hidden_size, hidden_size) 
        self.linearlogvar = nn.Linear(hidden_size, hidden_size) 
        

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        hidden = self.linear(hidden)
        # Reparprint Part -> for Dncoder
        mu = self.linearmu(output)
        logvar = self.linearlogvar(output)
        std = torch.exp(0.5*logvar)
        output = mu + torch.randn_like(std)
        return output, hidden, mu, logvar

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size - self.cond_embed_size, device=device)


def DoEncode(hidden_size, cond_embed_size, encoding_word, condition):
# initalize encoder & concate hidden
    Encoder = EncoderRNN(hidden_size, cond_embed_size)
    C2D = Char2Dict(hidden_size, cond_embed_size)
    encode_word = C2D.Word2Tensor(encoding_word)

# sequence to sequence part for encoder
    for i in range(encode_word.size()[0]):
        if i == 0:
            Encoder_hidden = torch.cat((Encoder.initHidden(), C2D.embed_cond(condition)), dim = 2)

        Encode_input = C2D.embed_char(encode_word[i], 'encode')

# encoder output has already been reparameterize
        encoder_output, Encode_hidden, mu, logvar = Encoder(Encode_input, Encoder_hidden)
        Encoder_hidden = Encode_hidden
    
    return encoder_output, Encode_hidden, mu, logvar


if __name__ == '__main__':
    DoEncode(256, 10, 'apple', 2)