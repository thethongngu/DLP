from dataloader import read_bci_data, BCIDataset
from torchsummary import summary
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
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
        learning_rate=1e-2,
        epochs=300,
        activation_type='ELU'
    ):

        super(DeepConvNet, self).__init__()
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.85)
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.85)
        )

        self.fourth_layer = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            get_activation_op(activation_type),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.85)
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=8600, out_features=2)
        )

        self.flatten_size = 8600
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, inputs):

        x = inputs.view(-1, 1, 2, 750)
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = self.fourth_layer(x)
        x = self.dense(x.view(-1, self.flatten_size))

        return x

    def train(self, train_gen, test_gen):

        train_log, test_log = [], []
        for epoch in range(self.epochs):
            for inputs, labels in train_gen:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels.long())
                loss.backward()
                self.optimizer.step()

            train_acc = self.test(train_gen)
            test_acc = self.test(test_gen)
            train_log.append((epoch, train_acc))
            test_log.append((epoch, test_acc))
            
            print('\nEpoch [%d] testing accuracy: %.2f%%' % (epoch, test_acc))
            print('Epoch [%d] training accuracy: %.2f%%' % (epoch, train_acc))

        print('\nFinished training!')
        print('Test accuracy: %.2f%%' % test_log[-1][1])

        return train_log, test_log

    """ Testing accuracy with specific data generator (train or test) """
    def test(self, gen):
        with torch.no_grad():
            corrects = 0
            n_test = 0
            for inputs, labels in gen:
                outputs = self.forward(inputs)            
                corrects += torch.sum(torch.max(outputs, 1)[1] == labels).item()
                n_test += labels.size()[0]
                
        return corrects / n_test * 100

class EEGNet(nn.Module):
    def __init__(
        self, 
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
            nn.Dropout(p=0.5)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation_op(activation_type),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.5)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, inputs):

        x = self.first_conv(inputs)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.classify(x.view(-1, 736))

        return x

    def train(self, train_gen, test_gen):

        train_log, test_log = [], []
        for epoch in range(self.epochs):
            for inputs, labels in train_gen:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels.long())
                loss.backward()
                self.optimizer.step()

            train_acc = self.test(train_gen)
            test_acc = self.test(test_gen)
            print('\nEpoch [%d] testing accuracy: %.2f%%\n' % (epoch, test_acc))
            print('Epoch [%d] training accuracy: %.2f%%' % (epoch, train_acc))
            
            train_log.append((epoch, train_acc))
            test_log.append((epoch, test_acc))

        print('\nFinished training!')
        print('Test accuracy: %.2f%%' % test_log[-1][1])

        return train_log, test_log

    """ Testing accuracy with specific data generator (train or test) """
    def test(self, gen):
        with torch.no_grad():
            corrects = 0
            n_test = 0
            for inputs, labels in gen:
                outputs = self.forward(inputs)            
                corrects += torch.sum(torch.max(outputs, 1)[1] == labels).item()
                n_test += labels.size()[0]
                
        return corrects / n_test * 100

""" Plot accuracy """
def plot_summary(data, name):

    plt.figure()
    for j in range(len(data)):
        label, config = data[j][0], data[j][1]
        xx, yy = [], []
        
        for i in range(len(config)):
            xx.append(config[i][0])
            yy.append(config[i][1])
        plt.plot(xx, yy, label=label)

    plt.xlabel('epochs')
    plt.ylabel('accuracy(%)')
    plt.legend()
    plt.savefig('%s.png' % name)


if __name__ == '__main__':
    
    """ Prepare environment """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    train_data, train_label, test_data, test_label = read_bci_data()
    train_set = BCIDataset(train_data, train_label)
    test_set = BCIDataset(test_data, test_label)

    train_gen = DataLoader(train_set, batch_size=32, shuffle=True)
    test_gen = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    activations = ['ELU', 'ReLU', 'LeakyReLU']
    logs = []

    """ DeepConvNet Experiments """
    for act_type in activations:
        # logs = []
        net = DeepConvNet(epochs=5000, learning_rate=0.0001, activation_type=act_type).cuda()
        train_log, test_log = net.train(train_gen, test_gen)
        logs.append(('Conv_%s_train' % act_type, train_log))
        logs.append(('Conv_%s_test' % act_type, test_log))
    plot_summary(logs, 'summary')
    
    """ EEGNet Experiments """
    for act_type in activations:
        # logs = []
        net = EEGNet(epochs=10000, learning_rate=0.0001, activation_type=act_type).cuda()
        train_log, test_log = net.train(train_gen, test_gen)
        logs.append(('EEG_%s_train' % act_type, train_log))
        logs.append(('EEG_%s_test' % act_type, test_log))
    plot_summary(logs, 'summary')
    

    for model in logs:
        print('%s: %.2f%%' % (model[0], model[1][-1][1]))
        print()

    
    