import matplotlib.pyplot as plt
import torch


def plot_loss(path):
    checkpoint = torch.load(path, map_location='cpu')
    loss_logger = checkpoint['loss_logger']
    ce_logger = checkpoint['ce_logger']
    kl_logger = checkpoint['kl_logger']
    bleu_logger = checkpoint['bleu_logger']

    plt.figure(figsize=(10, 6))
    plt.title('Training Loss')

    plt.plot(kl_logger, label='KLD')

    plt.xlabel('Epoch')
    plt.ylabel('KLD')

    h1, l1 = plt.gca().get_legend_handles_labels()

    ax = plt.gca().twinx()
    ax.plot(loss_logger, label='Loss', c="r")
    ax.plot(ce_logger, '--', label='Cross Entropy', c="m")
    ax.set_ylabel('Loss')

    h2, l2 = ax.get_legend_handles_labels()

    ax.legend(h1 + h2, l1 + l2)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title('Testing BLUE-4 score')
    plt.plot(bleu_logger, label='BLEU-4', c="g")
    plt.xlabel('Epoch')
    plt.ylabel('BLUE-4 score')
    plt.show()


if __name__ == '__main__':
    plot_loss('./models/best_model.pth')