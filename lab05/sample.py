from __future__ import unicode_literals, print_function, division
from torch.utils.tensorboard import SummaryWriter
from io import open

import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function
4. Gaussian score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, conversion words, Gaussian score, generation words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
========================================================================================"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

# ----------Hyper Parameters----------#
HIDDEN_SIZE = 256
VOCAB_SIZE = 28
LATENT_SIZE = 32
NUM_CONDITION = 4
CONDITION_SIZE = 8
TF_RATIO = 1
EMPTY_INP_RATIO = 0.1
KLD_WEIGHT = 0.0
LR = 0.05

index2char = {0: '<SOS>', 1: '<EOS>'}
char2index = {'<SOS>': 0, '<EOS>': 1}

ground_truth_word = 'accessed'
predicted_word = 'access'


def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""


def Gaussian_score(words):
    words_list = []
    score = 0
    path = 'dataset/train.txt'  # should be your directory of train.txt

    with open(path, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score / len(words)


# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, embedding, con_embedding, hidden_size, condition_size, latent_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size

        self.embedding = embedding
        self.con_embedding = con_embedding
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size + self.condition_size)
        self.mean = nn.Linear(self.hidden_size + self.condition_size, self.latent_size)
        self.logvar = nn.Linear(self.hidden_size + self.condition_size, self.latent_size)

    def reparameterize(self, mean, logvar):
        noise = torch.randn_like(logvar)
        return mean + noise * torch.exp(0.5 * logvar)

    def forward(self, inp_word_tensor, inp_tense_tensor, init_tensor):
        # print("inp_word_tensor", inp_word_tensor.size())
        # print("inp_tense_tensor", inp_tense_tensor.size())

        hidden, cell = init_tensor
        hidden = torch.cat((hidden, inp_tense_tensor), dim=2)

        # print("hidden", hidden.size())

        outputs, (hidden, cell) = self.lstm(inp_word_tensor, (hidden, cell))

        mean = self.mean(hidden)
        logvar = self.logvar(hidden)
        latent = self.reparameterize(mean, logvar)

        # print()

        return latent, mean, logvar


# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embedding, con_embedding, hidden_size, condition_size, latent_size, output_size):
        super(DecoderRNN, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.output_size = output_size

        self.embedding = embedding
        self.con_embedding = con_embedding
        self.lstm = nn.LSTM(self.hidden_size, self.latent_size + self.condition_size)
        self.out = nn.Linear(self.latent_size + self.condition_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, latent, out_tense_tensor, out_word_tensor, out_word, criterion, teacher_forcing):
        # print("latent", latent.size())
        # print("out_word_tensor", out_word_tensor.size())
        # print("out_tense_tensor", out_tense_tensor.size())

        hidden = torch.cat((latent, out_tense_tensor), dim=2)
        cell = torch.randn(1, 1, self.latent_size + self.condition_size, device=device)

        entropy_loss = 0.0
        output_str = ""
        use_teacher_forcing = True if random.random() < teacher_forcing else False
        last_output = 2

        for char_id in range(out_word_tensor.size(0)):
            if use_teacher_forcing:
                x = out_word_tensor[char_id]
            else:
                x = self.embedding(torch.tensor([[last_output]], device=device))

            outputs, (hidden, cell) = self.lstm(x.view(1, 1, -1), (hidden, cell))
            outputs = F.relu(outputs)
            outputs = self.out(outputs)
            outputs = self.softmax(outputs).view(1, -1)

            entropy_loss += criterion(outputs, out_word[char_id])


            value, index = outputs.topk(1)
            last_output = index.squeeze().detach()  # detach from history as input
            last_output = last_output.item()

            if not use_teacher_forcing and last_output == EOS_token:
                break

            output_str += index2char[last_output]

            print(outputs)
            print(last_output)
            print()

        return output_str, entropy_loss


class CVAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_condition, condition_size, latent_size, output_size):
        super(CVAE, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_condition = num_condition
        self.condition_size = condition_size
        self.latent_size = latent_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.con_embedding = nn.Embedding(self.num_condition, self.condition_size)
        self.encoder = EncoderRNN(
            self.embedding, self.con_embedding, self.hidden_size,
            self.condition_size, self.latent_size
        )
        self.decoder = DecoderRNN(
            self.embedding, self.con_embedding, self.hidden_size,
            self.condition_size, self.latent_size, self.output_size
        )

    def forward(self, init_hidden, input_word, output_word, input_tense, output_tense, criterion, teacher_forcing=0.5):
        inp_word_tensor = self.embedding(input_word)
        out_word_tensor = self.embedding(output_word)
        inp_tense_tensor = self.con_embedding(input_tense)
        out_tense_tensor = self.con_embedding(output_tense)

        # print("input_word:", input_word.size())
        # print("output_word:", output_word.size())

        latent, mean, logvar = self.encoder(inp_word_tensor, inp_tense_tensor, init_hidden)
        outputs, entropy_loss = self.decoder(latent, out_tense_tensor, out_word_tensor, output_word, criterion,
                                             teacher_forcing)

        return outputs, entropy_loss, KL_divergence(mean, logvar)

    def init_tensor(self):
        return (
            torch.randn(1, 1, self.hidden_size, device=device),
            torch.randn(1, 1, self.hidden_size + self.condition_size, device=device)
        )


def load_train_data():
    words_list = []
    with open('dataset/pretrain.txt', 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.append(word)

    return words_list


def generate_pair(word_list):
    data = []

    for idx in range(len(word_list)):
        for inp_tense in range(4):
            for out_tense in range(4):
                inp_word, out_word = word_list[idx][inp_tense], word_list[idx][out_tense]
                inp_word_tensor = torch.tensor([char2index[c] for c in inp_word]).view(-1, 1).to(device)
                out_word_tensor = torch.tensor([SOS_token] + [char2index[c] for c in out_word]).view(-1, 1).to(device)
                inp_tense_tensor = torch.tensor([inp_tense]).view(-1, 1).to(device)
                out_tense_tensor = torch.tensor([out_tense]).view(-1, 1).to(device)

                data.append((
                    inp_word_tensor, out_word_tensor, inp_tense_tensor,
                    out_tense_tensor, inp_word, out_word
                ))

    return data


def KL_divergence(mean, logvar):
    return 0.5 * (-torch.sum(logvar + 1) + torch.sum(torch.exp(logvar) + torch.sum(mean ** 2)))


def train_epoch(word_list, model, encoder_optimizer, decoder_optimizer, criterion):
    sum_entropy_loss = 0
    sum_kl_loss = 0

    data = generate_pair(word_list)
    for point in data:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        hidden = model.init_tensor()
        input_word, output_word, input_tense, output_tense, inputs, target = point

        outputs, entropy_loss, kl_loss = model(
            hidden, input_word, output_word, input_tense, output_tense, criterion, TF_RATIO
        )
        loss = entropy_loss + KLD_WEIGHT * kl_loss

        print("loss: ", loss)

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        sum_entropy_loss += entropy_loss.item()
        sum_kl_loss += kl_loss.item()

        print("input", inputs)
        print("target", target)
        print("predict", outputs)
        print()

    return sum_entropy_loss / len(data), sum_kl_loss / len(data)


def training(model, epochs=1000, learning_rate=0.01):
    start = time.time()
    best_bleu_score = 0.0

    encoder_optimizer = optim.SGD(model.encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(model.decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    word_list = load_train_data()

    for it in range(1, epochs + 1):

        model.train()
        avg_entropy, avg_kl = train_epoch(word_list, model, encoder_optimizer, decoder_optimizer, criterion)

        model.eval()

        # bleu_score = evaluating(model)
        bleu_score = 0  # TODO: delete this line

        # if bleu_score > best_bleu_score:
        #     best_bleu_score = bleu_score
        #     torch.save(model.state_dict(), 'best_cvae')
        #     print('New checkpoint saved!')
        #
        # writer.add_scalar('Entropy/train', avg_entropy, it)
        # writer.add_scalar('KL/train', avg_kl, it)
        # writer.add_scalar('BLEU/test', bleu_score, it)

        print(
            '%s (%d %d%%) Entropy: %.4f; KL: %.4f; BLEU-4: %.4f' %
            (timeSince(start, it / epochs), it, it / epochs * 100, avg_entropy, avg_kl, bleu_score)
        )


def load_test_data():
    data = []
    inp_tense = [0, 0, 0, 0, 3, 0, 3, 2, 2, 2]
    out_tense = [3, 2, 1, 1, 1, 2, 0, 0, 3, 1]
    idx = 0
    with open('dataset/test.txt', 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[1] = word[1].strip('\n')

            inp_word_tensor = torch.tensor([char2index[c] for c in word[0]]).view(-1, 1).to(device)
            out_word_tensor = torch.tensor([SOS_token] + [char2index[c] for c in word[1]]).view(-1, 1).to(device)
            inp_tense_tensor = torch.tensor([inp_tense[idx]]).view(-1, 1).to(device)
            out_tense_tensor = torch.tensor([out_tense[idx]]).view(-1, 1).to(device)
            data.append((
                inp_word_tensor, out_word_tensor, inp_tense_tensor,
                out_tense_tensor, word[0], word[1]
            ))

            idx += 1

    return data


def evaluating(model):
    bleu_score = 0.0
    criterion = nn.CrossEntropyLoss()

    data = load_test_data()
    for point in data:
        hidden = model.init_tensor()
        input_word, output_word, input_tense, output_tense, inputs, target = point

        outputs, entropy_loss, kl_loss = model(
            hidden, input_word, output_word, input_tense, output_tense, criterion, TF_RATIO
        )

        bleu_score += compute_bleu(outputs, target)

        print("input", inputs)
        print("target", target)
        print("predict", outputs)
        print()

    return bleu_score / (len(data))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def create_table():
    for c in range(97, 123):
        index2char[c - 95] = chr(c)
        char2index[chr(c)] = c - 95


# -------------------- MAIN ----------------------------
writer = SummaryWriter()
create_table()

cvae = CVAE(VOCAB_SIZE, HIDDEN_SIZE, NUM_CONDITION, CONDITION_SIZE, LATENT_SIZE, VOCAB_SIZE).to(device)
training(cvae, epochs=1000, learning_rate=LR)
