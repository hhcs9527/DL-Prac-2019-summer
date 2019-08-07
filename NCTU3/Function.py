#################################################################
# This .py provide the function can be used in this lab
#################################################################

# Standard library
import os

# Class that help reading
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Related function
def readLangs(lang1):
    print("Reading lines...")
    path = './lab3'
    os.chdir(path)
    lines = []
    # Read the file and split into lines
    with open('train.txt', encoding='utf-8') as f:
            for line in iter(f):
                line = line.split( )
                for i in range(len(line)):
                    lines.append(line[i])
    pairs = [chr(i) for i in range(97, 123)]
    input_lang = Lang(lang1)

    return input_lang, lines, pairs


def prepareData(lang1):
    input_lang, lines, pairs = readLangs(lang1)
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair)
    #print("Counted words:")
    return input_lang, lines, pairs