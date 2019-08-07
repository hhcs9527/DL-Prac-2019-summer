

# import related class and .py file
from autoencoder import EncoderRNN, DecoderRNN
import train as T
import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 256
    vocab_size = 28
    encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size + 10, vocab_size).to(device)
    T.trainIters(encoder1, decoder1, 5000)