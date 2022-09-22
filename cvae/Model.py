import random
import torch
import torch.nn as nn
import Const
import dataset


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, condition_dim, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim + condition_dim)
        self.mu = nn.Linear(hidden_dim + condition_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim + condition_dim, latent_dim)

    def forward(self, word, condition):
        c_0 = torch.zeros((1, 1, self.hidden_dim + self.condition_dim), device=Const.device)
        h_0 = torch.zeros((1, 1, self.hidden_dim), device=Const.device)
        h_0 = torch.cat((h_0, condition), dim=2)

        embedded = self.embedding(word).view(-1, 1, self.embedding_dim)
        outputs, (hidden, cell) = self.lstm(embedded, (h_0, c_0))

        mu = self.mu(hidden)
        sigma = self.sigma(hidden)

        z = self.sample_z() * torch.exp(sigma / 2) + mu

        return z, mu, sigma

    def sample_z(self):
        return torch.normal(
            torch.FloatTensor([0] * self.latent_dim),
            torch.FloatTensor([1] * self.latent_dim)
        ).to(Const.device)


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, condition_dim, latent_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim

        self.latent2hidden = nn.Linear(latent_dim + condition_dim, hidden_dim + condition_dim)
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim + condition_dim)
        self.out = nn.Linear(hidden_dim + condition_dim, output_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, self.embedding_dim)

        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output).view(-1, self.output_dim)

        return output, hidden

    def initHidden(self, z, condition):
        return (
            self.latent2hidden(torch.cat((z, condition), dim=2)),
            torch.zeros((1, 1, self.hidden_dim + self.condition_dim), device=Const.device)
        )


class EncoderDecoder(nn.Module):
    def __init__(self, io_dim, embedding_dim, hidden_dim, condition_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(io_dim, embedding_dim, hidden_dim, condition_dim, latent_dim)
        self.decoder = Decoder(io_dim, embedding_dim, hidden_dim, condition_dim, latent_dim)
        self.condition_embedding = nn.Embedding(4, condition_dim)
        self.word_processing = dataset.WordProcessing()

    def forward(
            self, input_word, input_tense, target_word, target_tense,
            teacher_forcing_ratio=0.5, word_dropout_ratio=0.1
    ):
        input_condition_embedded = self.condition_embedding(input_tense).view(1, 1, -1)
        target_condition_embedded = self.condition_embedding(target_tense).view(1, 1, -1)

        batch_size = target_word.shape[0]
        max_len = target_word.shape[1]
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, target_vocab_size).to(Const.device)

        z, mu, sigma = self.encoder(input_word[:, 1:-1], input_condition_embedded)

        hidden = self.decoder.initHidden(z, target_condition_embedded)
        input = target_word[:, 0]
        # print(input)
        teacher_force = random.random() < teacher_forcing_ratio

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = torch.argmax(output, dim=1)

            input = (target_word[:, t] if teacher_force else top1)
            word_dropout = random.random() < word_dropout_ratio
            if word_dropout:
                input = torch.tensor(
                    self.word_processing.char2int(self.word_processing.UNK_TOKEN)
                ).to(Const.device)

        return outputs, mu, sigma
