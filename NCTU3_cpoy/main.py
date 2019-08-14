

# import related class and .py file
from autoencoder import EncoderRNN, DecoderRNN
import train as T
import test as test
import torch
import os

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 256
    vocab_size = 28
    encoder1 = EncoderRNN(vocab_size, hidden_size ).to(device)
    decoder1 = DecoderRNN(hidden_size + 10, vocab_size).to(device)

    if os.path.isfile('EncoderRNN.pkl'):
        encoder1.load_state_dict(torch.load('EncoderRNN.pkl'))
    if os.path.isfile('DecoderRNN.pkl'):
        decoder1.load_state_dict(torch.load('DecoderRNN.pkl'))

    print('Training (1) or testing (other number)?')
    if (int(input()) == 1):
        T.trainIters(encoder1, decoder1, n_iters = 50, epoch = 3)
    else:
        print('Test result of tense transform : ')
        test.go_test(encoder1, decoder1)
        print('Testing is done !')
