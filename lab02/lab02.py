from dataloader import read_bci_data
from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim

def get_activation_op(activation_type):
    if activation_type == 'ELU':
        return nn.ELU()
    elif activation_type == 'LeakyReLU':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()


class DeepConvNet(nn.Module):
    def __init__(
        self, 
        batch_size=64,
        learning_rate=1e-2,
        epochs=300,
        activation_type='ELU'
    ):

        super(EEGNet, self).__init__()
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1)),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.fourth_layer = nn.Sequential(
            nn.Conv2d(1, 200, kernel_size=(1, 5), stride=(1, 1)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=200, out_features=2)
        )

class EEGNet(nn.Module):
    def __init__(
        self, 
        batch_size=64,
        learning_rate=1e-2,
        epochs=300,
        activation_type='ELU'
    ):

        super(EEGNet, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation_op(activation_type),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation_op(activation_type),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Linear(in_features=736, out_features=2, bias=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, inputs):

        x = self.first_conv(inputs)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.classify(x.view(-1, 736))

        return x

    def train(self, train_data, train_label, test_data, test_label):

        train_log, test_log = [], []
        for epoch in range(self.epochs):

            training_loss = 0.0
            for i in range(0, len(train_data), self.batch_size):
                inputs = train_data[i: i + self.batch_size]
                labels = train_label[i: i + self.batch_size]

                self.optimizer.zero_grad()

                outputs = self.forward(inputs)

                loss = self.criterion(outputs, labels.long())
                loss.backward()
                self.optimizer.step()

                training_loss += loss.item()

            testing_loss = self.test(test_data, test_label)
            train_log.append((epoch, training_loss))
            test_log.append((epoch, testing_loss))
            print('Epoch [%d] training loss: %.10f' % (epoch, training_loss))

        return train_log, test_log

    def test(self, test_data, test_label):
        with torch.no_grad():
            for i in range(0, len(test_data), self.batch_size):
                inputs = train_data[i: i + self.batch_size]
                labels = train_label[i: i + self.batch_size]            


def plot_summary(data):

    plt.figure()
    for j in range(len(data)):
        label, config = data[j][0], data[j][1]
        xx, yy = [], []
        
        for i in range(len(config)):
            xx.append(config[i][0])
            yy.append(config[i][1])
        plt.plot(xx, yy, label=label)

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('summary.png')


if __name__ == '__main__':
    
    train_data, train_label, test_data, test_label = read_bci_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    eeg_net_elu = EEGNet(activation_type='ELU').cuda()
    eeg_net_relu = EEGNet(activation_type='ReLU').cuda()
    eeg_net_lrelu = EEGNet(activation_type='LeakyReLU').cuda()

    train_log, test_log = eeg_net_elu.train(train_data, train_label, test_data, test_label)
    logs.append(('EEG_ELU_train', train_log))
    logs.append(('EEG_ELU_test', test_log))

    # train_log, test_log = eeg_net_relu.train(train_data, train_label, test_data, test_label)
    # logs.append(('EEG_ReLU_train', train_log))
    # logs.append(('EEG_ReLU_test', test_log))

    # train_log, test_log = eeg_net_lrelu.train(train_data, train_label, test_data, test_label)
    # logs.append(('EEG_LeakyReLU_train', train_log))
    # logs.append(('EEG_LeakyReLU_test', test_log))

