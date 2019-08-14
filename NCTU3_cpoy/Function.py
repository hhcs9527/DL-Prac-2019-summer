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
def readLangs(lang1,file):
    print("Reading lines...")
    path = './lab3'
    os.chdir(path)
    lines = []
    sepline = []
    char = [chr(i) for i in range(97, 123)]
    input_lang = Lang(lang1)
    # Read the file and split into lines
    if file == 'train':
        train_set = []
        with open(file + '.txt', encoding='utf-8') as f:
                for line in iter(f):
                    line = line.split( )
                    for i in range(len(line)):
                        if (line[i] not in sepline or line[i].isdigit()) and i < 4:
                            sepline.append(line[i])
                            lines.append(line[i])
                    for i in range(len(sepline)):
                        for j in range(len(sepline)):
                            pair = [line[i],line[j],j]
                            train_set.append(pair)
                    sepline = []
        return input_lang, char, train_set
    else:
        test_set = []
        with open(file + '.txt', encoding='utf-8') as f:
                for line in iter(f):
                    line = line.split( )
                    test_set.append(line)
        return input_lang, char, test_set


    


def prepareData(lang1,file):
    input_lang, char, data = readLangs(lang1, file)
    print("Counting words...")
    for pair in char:
        input_lang.addSentence(pair)
    #print("Counted words:")
    return input_lang, char, data


#input_lang, char, data = prepareData('train','train')

#print(data)