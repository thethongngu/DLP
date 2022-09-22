import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.autograd import Variable
import dataset
from torch.utils.data import DataLoader
import Model
import Const
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import matplotlib.pyplot as plt
import copy

IO_DIM = 30
EMBEDDING_DIM = 512
HIDDEN_DIM = 512
CONDITION_DIM = 8
LATENT_DIM = 32
criterion = nn.CrossEntropyLoss().to(Const.device)

model = Model.EncoderDecoder(IO_DIM, EMBEDDING_DIM, HIDDEN_DIM, CONDITION_DIM, LATENT_DIM).to(Const.device)
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(model, train_loader, optimizer, criterion, kl_weight=0, print_frequency=200):
    model.train()
    word_processing = dataset.WordProcessing()
    epoch_loss = 0
    epoch_kl = 0
    epoch_ce = 0
    total_loss_frequency = 0
    kl_frequency = 0
    bleu_score = 0

    for i, (word, tense) in enumerate(train_loader):
        optimizer.zero_grad()

        input_word = Variable(word).to(Const.device)
        input_tense = Variable(tense).to(Const.device)
        target_word = Variable(word).to(Const.device)
        target_tense = Variable(tense).to(Const.device)

        outputs, mu, sigma = model(input_word, input_tense, target_word, target_tense)
        target_word = target_word[:, 1:].view(-1)
        outputs = outputs[1:].view(-1, outputs.shape[-1])
        prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        targets_str = word_processing.tensor2word(target_word)
        prediction = word_processing.tensor2word(prediction)
        bleu_score += compute_bleu(prediction, targets_str)

        kl_loss = KL_loss(mu, sigma)
        ce_loss = criterion(outputs, target_word)

        total_loss = ce_loss + KLD_weight_annealing(i, 'cyclical') * kl_loss
        total_loss.backward()
        optimizer.step()

        kl_frequency += kl_loss.item()
        total_loss_frequency += total_loss.item()

        epoch_loss += total_loss.item()
        epoch_kl += kl_loss.item()
        epoch_ce += ce_loss.item()

        if (i + 1) % print_frequency == 0:
            print('\t({}/{}) Train Loss: {:.5f} KL Loss: {:.5}'.format(i + 1, len(train_loader),
                                                                       total_loss_frequency / print_frequency,
                                                                       kl_frequency / print_frequency))

            total_loss_frequency = 0
            kl_frequency = 0

    return epoch_loss / len(train_loader), epoch_ce / len(train_loader), epoch_kl / len(train_loader),\
           bleu_score / len(train_loader)


def KLD_weight_annealing(epoch, mode='cyclical'):

    if mode == 'cyclical':
        n_min = 0
        n_max = 0.002
        n_epoch = 2454
        return n_min + 0.5 * (n_max - n_min) * (1 + np.cos(np.pi * ((epoch % n_epoch) + n_epoch) / n_epoch))
    else:
        w = 0.0001 * epoch
        if w > 1.0:
            w = 1.0

        return w


def KL_loss(m, logvar):
    return torch.sum(0.5 * (-logvar + (m ** 2) + torch.exp(logvar) - 1))


def evaluate(model, test_loader):
    word_processing = dataset.WordProcessing()
    total_score = 0
    with torch.no_grad():
        for i, (input_word, input_tense, target_word, target_tense) in enumerate(test_loader):
            input_word = Variable(input_word).to(Const.device)
            input_tense = Variable(input_tense).to(Const.device)
            target_word = Variable(target_word).to(Const.device)
            target_tense = Variable(target_tense).to(Const.device)

            outputs, mu, sigma = model(input_word, input_tense, target_word, target_tense, teacher_forcing_ratio=0)

            outputs = outputs[1:].view(-1, outputs.shape[-1])

            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            inputs_str = word_processing.tensor2word(input_word[0])
            targets_str = word_processing.tensor2word(target_word[0])
            prediction = word_processing.tensor2word(prediction)
            bleu_score = compute_bleu(prediction, targets_str)
            total_score += bleu_score

            print('input:\t\t{}'.format(inputs_str))
            print('target:\t\t{}'.format(targets_str))
            print('prediction:\t{}'.format(prediction))
            print('BLEU score: {}'.format(bleu_score))
            print('')

    return total_score / len(test_loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def testing():
    model = Model.EncoderDecoder(IO_DIM, EMBEDDING_DIM, HIDDEN_DIM, CONDITION_DIM, LATENT_DIM).to(Const.device)
    checkpoint = torch.load('./models_03_acyclic_max0002_epoch10/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    test_loader = DataLoader(dataset=dataset.EnglishTenseDataset(mode=dataset.Mode.TEST), batch_size=1)
    model.eval()
    word_processing = dataset.WordProcessing()
    bleu_scores = []
    with torch.no_grad():
        for (input_word, input_tense, target_word, target_tense) in test_loader:
            input_word = Variable(input_word).to(Const.device)
            input_tense = Variable(input_tense).to(Const.device)
            target_word = Variable(target_word).to(Const.device)
            target_tense = Variable(target_tense).to(Const.device)

            outputs, mu, sigma = model(input_word, input_tense, target_word, target_tense, teacher_forcing_ratio=0)

            outputs = outputs[1:].view(-1, outputs.shape[-1])
            prediction = torch.max(torch.softmax(outputs, dim=1), 1)[1]

            inputs_str = word_processing.tensor2word(input_word[0])
            targets_str = word_processing.tensor2word(target_word[0])
            prediction = word_processing.tensor2word(prediction)
            bleu_score = compute_bleu(prediction, targets_str)

            print('input:\t\t{}'.format(inputs_str))
            print('target:\t\t{}'.format(targets_str))
            print('prediction:\t{}'.format(prediction))
            # print('BLEU score: {}'.format(bleu_score))
            print('')

            bleu_scores.append(bleu_score)

    print('BLEU score avg: {}'.format(sum(bleu_scores) / len(bleu_scores)))


def training():
    N_EPOCHS = 500

    best_valid_bleu = float('-inf')
    train_loader = DataLoader(dataset=dataset.EnglishTenseDataset(), batch_size=1)
    test_loader = DataLoader(dataset=dataset.EnglishTenseDataset(mode=dataset.Mode.TEST), batch_size=1)
    KLD_weight = 0
    total_loss_logger = []
    kl_loss_logger = []
    ce_loss_logger = []
    bleu_score_logger = []

    for epoch in range(N_EPOCHS):
        print('Epoch: {:02}'.format(epoch + 1))

        start_time = time.time()

        total_loss, ce_loss, kl_loss, bleu_score = train(model, train_loader, optimizer, criterion, KLD_weight)
        valid_bleu_score = evaluate(model, test_loader)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        total_loss_logger.append(total_loss)
        kl_loss_logger.append(kl_loss)
        bleu_score_logger.append(valid_bleu_score)
        ce_loss_logger.append(ce_loss)

        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            best_model = copy.deepcopy(model)
            best_model.name = 'best_model'
            save_checkpoint(epoch, best_model, total_loss_logger, ce_loss_logger, kl_loss_logger, bleu_score_logger)

        model.name = 'cvae_' + str(epoch)
        save_checkpoint(epoch, model, total_loss_logger, ce_loss_logger, kl_loss_logger, bleu_score_logger)

        print('Epoch: {:02} | Done | Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
        print('\tTrain Loss: {:.5f} BLEU score: {:.5f} KL Loss: {:5f}'.format(total_loss, bleu_score, kl_loss))
        print('\tValid BLEU score: {:.5f}\n'.format(valid_bleu_score))


def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


def save_checkpoint(epoch, model, loss_logger, ce_logger, kl_logger, bleu_logger):
    state = {'epoch': epoch + 1, 'model': model.state_dict(),
             'loss_logger': loss_logger,
             'bleu_logger': bleu_logger,
             'ce_logger': ce_logger,
             'kl_logger': kl_logger}

    torch.save(state, './models/' + model.name + '.pth')


if __name__ == '__main__':
    # plt.plot([KLD_weight_annealing(i) for i in range(4908)])
    # plt.show()
    # training()
    testing()
