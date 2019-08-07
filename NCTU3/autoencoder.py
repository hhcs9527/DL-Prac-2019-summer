#################################################################
# This .py provide the model can be used in this lab
#################################################################

# torch library
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # input -> 表示原來在幾維空間 / hidden 表示要壓到幾為空間
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # For the decoder 
        self.linearmu = nn.Linear(hidden_size, hidden_size) 
        self.linearlogvar = nn.Linear(hidden_size, hidden_size) 
        self.embeddingcond = nn.Embedding(4, 10)

    def forward(self, input, hidden, cond):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        # Reparprint Part -> output of the Encoder
        mu = self.linearmu(output)
        logvar = self.linearlogvar(output)
        std = torch.exp(0.5*logvar)
        output = mu + torch.randn_like(std)
        return output, hidden, mu, logvar


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size - 10, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.softmax2 = nn.Softmax(dim = 1)
        #self.lin = nn.Linear(output_size,1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        output = torch.argmax(self.softmax2(output))
        output = torch.tensor(output, dtype = torch.float32)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size , device=device)


