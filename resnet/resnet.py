from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import plot_confusion_matrix

import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy

""" Freeze parameter """
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def get_optimizer(model):
    update_params = []
    for param in model.parameters():
        if param.requires_grad == True:
            update_params.append(param)

    return optim.SGD(update_params, lr=1e-4, momentum=0.9)

def plot_summary(data, name):

    plt.figure()
    for j in range(len(data)):
        label, config = data[j][0], data[j][1]
        xx, yy = [], []
        
        for i in range(len(config)):
            xx.append(i + 20)
            yy.append(config[i])
        plt.plot(xx, yy, label=label)

    plt.xlabel('epochs')
    plt.ylabel('accuracy(%)')
    plt.legend()
    plt.savefig('%s.png' % name)

""" Train function"""
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_logs, test_logs = [], []

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            print(phase + ': ')
            model.train() if phase == 'train' else model.eval()

            running_loss = running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                """ Init """
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # print(labels)
                    # print(preds)
                    # print(outputs)
                    # print(torch.sum(preds == labels.data).item())

                """ Loss, Accuracy """
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # info
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) * 100
            if phase == 'train':
                train_logs.append((epoch, epoch_acc))
            else:
                test_logs.append((epoch, epoch_acc))

            print('Loss: {:.4f} Acc: {:.4f}%\n'.format(epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

    print('Best testing acc: {:4f}%'.format(best_acc))
    model.load_state_dict(best_model)
    return model, train_logs, test_logs 
        
def test_model(model, testloader):

    model.eval()

    corrects = 0
    label_logs, output_logs = [], []
    for inputs, labels in testloader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        predictions = torch.max(outputs, 1)[1]
        corrects += torch.sum(predictions == labels).item()

        label_logs.extend(labels.cpu().numpy())
        output_logs.extend(predictions.cpu().numpy())

    acc = corrects / len(testloader.dataset) * 100
    print('Testing Acc: {:.4f}%'.format(acc))

    return label_logs, output_logs


if __name__ == "__main__":
    
    """ Get resnet model from torchvision """
    resnet18 = models.resnet18()
    resnet50 = models.resnet50()
    resnet18_pretrained = models.resnet18(pretrained=True)
    resnet50_pretrained = models.resnet50(pretrained=True)

    """ Only update gradient for last layer """
    freeze_model(resnet18_pretrained)
    freeze_model(resnet50_pretrained)

    """ replace the last layer of resnet """
    resnet18.fc = nn.Linear(512, 5)
    resnet50.fc = nn.Linear(2048, 5)
    resnet18_pretrained.fc = nn.Linear(512, 5)
    resnet50_pretrained.fc = nn.Linear(2048, 5)

    """ List of testing model """
    my_models = [resnet18, resnet18_pretrained, resnet50, resnet50_pretrained]
    names = ['resnet18', 'resnet18 pretrained', 'resnet50', 'resnet50 pretrained']

    # my_models = [resnet18]
    # names = ['resnet18']

    """ Loss function """
    criterion = nn.CrossEntropyLoss()

    """ Load data """
    trainset = RetinopathyLoader('data/', 'train', transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(65),
        transforms.ToTensor()
    ]))
    testset = RetinopathyLoader('data/', 'test', transforms.Compose([
        transforms.ToTensor()
    ]))

    class_weights = 1. / torch.tensor([20655, 1955, 4210, 698, 581], dtype=torch.float)
    point_weights = [class_weights[out] for inp, out in trainset]
    sampler = WeightedRandomSampler(
        weights=point_weights, num_samples=len(point_weights), replacement=True
    )

    trainloader = DataLoader(trainset, batch_size=4, sampler=sampler)
    testloader = DataLoader(testset, batch_size=4, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """ Training """
    logs = []

    for idx in range(len(my_models)):
        model = my_models[idx]
        name = names[idx]

        model.load_state_dict(torch.load('%s_best' % name))

        print('\nTraining: %s' % name)

        model = model.to(device)
        optimizer = get_optimizer(model)
        epochs = 30 if name.find('resnet18') != -1 else 30

        dataloaders = {
            'train': trainloader,
            'test': testloader
        }
        best_model, train_log, test_log = train_model(model, dataloaders, criterion, optimizer, epochs)
        label_log, output_log = test_model(model, testloader)

        print(label_log)
        print(output_log)

        plot_confusion_matrix(
            best_model, output_log, label_log, 
            cmap=plt.cm.Blues, normalize='true', labels=[0, 1, 2, 3, 4]
        )
        plt.savefig('confusion_%s.png' % name)

        logs.append(('%s train' % name, train_log))
        logs.append(('%s test' % name, test_log))

        torch.save(best_model.state_dict(), '%s_best' % name)

        if idx == 1 or idx == 3:
            plot_summary(logs, 'resnet18' if name.find('resnet18') != -1 else 'resnet50')
            logs = []

    # ---------------------------------------------------------------------------

    """ Load model """
    for idx in range(4):
        name = names[idx]
        model = my_models[idx].to(device)
        model.load_state_dict(torch.load('%s_best' % name))
        model.eval()

        print(name)
        test_model(model, testloader)
        print()

    # for idx in range(4):
    #     print(my_models[idx])