import torch
from torch.autograd import Variable

SOS_token = 0
EOS_token = 1


class Lang:
    # wctype = WORD or CHAR type language
    def __init__(self, name, wc_type):
        self.name = name
        self.wc_type = wc_type
        self.wc2index = {}
        self.wc2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.max_length = 0

    def getWords(self, sentence):
        if self.wc_type == 'WORD':
            words = sentence.split(' ')
        else:
            words = sentence
        return words

    def addSentence(self, sentence):
        words = self.getWords(sentence)
        max_length = 2
        for word in words:
            self.addWord(word)
            max_length = max_length + 1
        if max_length > self.max_length:
            self.max_length = max_length

    def addWord(self, word):
        if word not in self.wc2index:
            self.wc2index[word] = self.n_words
            self.wc2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.wc2count[word] += 1

    # Lang utility methods
    def indexesFromSentence(self, sentence):
        words = self.getWords(sentence)
        return [self.wc2index[word] for word in words]

    def variableFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        return result
