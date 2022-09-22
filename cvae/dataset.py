from torch.utils.data import Dataset
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
import Const

SOS_TOKEN = ord('z') - ord('a') + 1
EOS_TOKEN = ord('z') - ord('a') + 2


class Mode(Enum):
    TRAIN = 0
    TEST = 1


class WordProcessing():
    def __init__(self):
        self.SOS_TOKEN = '{'
        self.EOS_TOKEN = '|'
        self.UNK_TOKEN = '}'

    def char2int(self, c):
        return ord(c) - ord('a')

    def int2char(self, i):
        return chr(i + ord('a'))

    def word2int(self, word):
        return np.array([self.char2int(c) for c in word])

    def word2tensor(self, word):
        list_word = [self.SOS_TOKEN] + list(word) + [self.EOS_TOKEN]
        list_word = self.word2int(list_word)

        return torch.tensor(list_word).long()

    def tensor2word(self, t):
        s = ''
        for i in t.data:
            c = self.int2char(i)
            if c == '|':
                break

            if c == '{' or c == '|':
                c = ''

            s += c
        return s


class EnglishTenseDataset(Dataset):
    def __init__(self, root='./dataset/', mode=Mode.TRAIN):
        self.mode = mode
        self.root = root
        self.tenses = ['simple-present', 'third-person', 'present-progressive', 'simple-past']
        self.word_processing = WordProcessing()

        if mode == Mode.TRAIN:
            self.data = np.loadtxt(self.root + 'train.txt', dtype=np.str).reshape(-1)
        else:
            self.data = np.loadtxt(self.root + 'test.txt', dtype=np.str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == Mode.TRAIN:
            return self.word_processing.word2tensor(self.data[index]), index % 4
        else:
            target_tenses = np.array([[0, 3], [0, 2], [0, 1], [0, 1], [3, 1], [0, 2], [3, 0], [2, 0], [2, 3], [2, 1]])
            return self.word_processing.word2tensor(self.data[index][0]), target_tenses[index][0], \
                   self.word_processing.word2tensor(self.data[index][1]), target_tenses[index][1]


if __name__ == '__main__':
    dataset = EnglishTenseDataset(mode=Mode.TEST)
    print(dataset[0])
