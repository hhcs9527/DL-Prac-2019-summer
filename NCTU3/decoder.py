import torch
import torch.nn as nn
from prepaer_data import Char2Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.relu = nn.LeakyReLU(0.1)
        #self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.get_out = nn.Linear(hidden_size, output_size)
        self.soft = nn.LogSoftmax(dim = 1)
        self.softmax = nn.Softmax(dim = 1)
        self.activation = nn.ELU()

    def forward(self, input, hidden):
        input = self.relu(input)
        output_pred, hidden = self.gru(input, hidden)
        #hidden = self.linear(hidden)
        #hidden = self.relu(hidden)

        predict = self.soft(self.get_out(output_pred[0]))
        output = torch.tensor(torch.argmax(self.softmax(predict)), dtype = torch.long)
        
        return output, hidden, predict




if __name__ == '__main__':
    print('how you doing???')
