import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_token = 1

# change character 2 dic
class Char2Dict:
# C2D
    def __init__(self, hidden_size, cond_embed_size):
        self.char_list = [chr(i) for i in range(97, 123)]
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

        for word in self.char_list:
            self.addWord(word)

        self.hidden_size = hidden_size
        self.cond_embed_size = cond_embed_size
        self.input_size = self.n_words

    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# W2T
    def Word2Tensor(self, word):
        #print(word)
        index = [self.word2index[char] for char in word]
        index.append(EOS_token)
        return torch.tensor(index, dtype = torch.long).view(-1, 1)


# Embedding char to vector / condition to vector
    def embedding(self, char, input_size, output_size):
        embed = nn.Embedding(input_size, output_size)
        return embed(char).view(1, 1, -1)

    def embed_char(self, char, encode):
        if encode == 'encode':
            return self.embedding(char, self.input_size, self.hidden_size).view(1, 1, -1).to(device)
        else:
            return self.embedding(char, self.input_size, self.hidden_size + self.cond_embed_size).view(1, 1, -1).to(device)

    def embed_cond(self, condition):
        cond = torch.tensor(condition, dtype = torch.long)
        return self.embedding(cond, 4, self.cond_embed_size).view(1, 1, -1).to(device)


# Return word
    def return_word(self, char):
        return self.index2word[char.item()]


if __name__ == '__main__':
    
    a = Char2Dict(256, 10)
    word = 'sos'
    ind = a.Word2Tensor(word)
    print(ind)
 
