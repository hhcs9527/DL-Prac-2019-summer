import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_token = 1

# change character 2 dic
class Char2Dict:
# C2D
    def __init__(self, cond_embed_size):
        self.char_list = [chr(i) for i in range(97, 123)]
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

        for word in self.char_list:
            self.addWord(word)
            
        self.cond_embed_size = cond_embed_size

    
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


# Return word
    def return_word(self, char):
        return self.index2word[char.item()]


if __name__ == '__main__':
    a = Char2Dict(256, 10)
    word = 'sos'
    ind = a.Word2Tensor(word)
    print(ind)
 
