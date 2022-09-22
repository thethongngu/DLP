import torch
import Model, Const, dataset

IO_DIM = 30
EMBEDDING_DIM = 512
HIDDEN_DIM = 512
CONDITION_DIM = 8
LATENT_DIM = 32


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

    print(score)
    return score / len(words)


def decode_inference(decoder, z, target_condition_embedded, max_len=20):
    word_processing = dataset.WordProcessing()
    z = z.view(1, 1, -1)

    hidden = decoder.initHidden(z, target_condition_embedded)
    input = torch.tensor(word_processing.char2int(word_processing.SOS_TOKEN)).to(Const.device)
    target_vocab_size = decoder.output_dim

    outputs = torch.zeros(max_len, 1, target_vocab_size).to(Const.device)

    for t in range(1, max_len):
        output, hidden = decoder(input, hidden)
        outputs[t] = output
        top1 = torch.argmax(output, dim=1)
        input = top1

    return outputs


def word_generation(path):
    checkpoint = torch.load(path, map_location='cpu')
    model = Model.EncoderDecoder(IO_DIM, EMBEDDING_DIM, HIDDEN_DIM, CONDITION_DIM, LATENT_DIM).to(Const.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    word_processing = dataset.WordProcessing()
    english_tense = dataset.EnglishTenseDataset()
    generated = []

    for word_id in range(100):
        noise = model.encoder.sample_z()
        words = []
        for i in range(len(english_tense.tenses)):
            condition = torch.tensor([i]).long().to(Const.device)
            target_condition_embedded = model.condition_embedding(condition).view(1, 1, -1)
            outputs = decode_inference(model.decoder, noise, target_condition_embedded)
            outputs = outputs[1:].view(-1, outputs.shape[-1])
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            prediction = word_processing.tensor2word(prediction)
            # print('{:20s} : {}'.format(english_tense.tenses[i], prediction))
            words.append(prediction)
        print(words)
        generated.append(words)

    print(Gaussian_score(generated))


if __name__ == '__main__':
    word_generation('./models/best_model.pth')
