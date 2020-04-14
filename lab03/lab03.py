from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader

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

    return optim.SGD(update_params, lr=0.001, momentum=0.9)

""" Train function"""
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    logs = []

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = running_corrects = 0
        model.train()

        for inputs, labels in dataloaders:

            """ Init """
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            """ Forward """
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)

            """ Loss, Accuracy """
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # info
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        logs.append((epoch, epoch_acc))

        print('Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

    print('Best training acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model)
    return model, logs 
        
def test_model(model, dataloader, criterion):

    model.eval()

    n_test = 0
    corrects = 0
    for inputs, labels in dataloaders:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        corrects += torch.sum(torch.max(outputs, 1)[1] == labels).item()
        n_test += labels.size()[0]

    acc = corrects / n_test * 100
    print('Testing Acc: {:.4f}'.format(acc))


if __name__ == "__main__":
    
    """ Get resnet model from torchvision """
    resnet18 = models.resnet18(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)

    """ Only update gradient for last layer """
    freeze_model(resnet18)
    freeze_model(resnet50)

    """ replace the last layer of resnet """
    resnet18.fc = nn.Linear(512, 5)
    resnet50.fc = nn.Linear(512, 5)

    """ Load data """
    trainset = RetinopathyLoader('data/', 'train', transforms.Compose([
        transforms.ToTensor()   
    ]))
    testset = RetinopathyLoader('data/', 'test', transforms.Compose([
        transforms.ToTensor()
    ]))
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=4, shuffle=True)

    """ Training """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)
    resnet50 = resnet50.to(device)

    optimizer18 = get_optimizer(resnet18)
    optimizer50 = get_optimizer(resnet50)
    criterion = nn.CrossEntropyLoss()

    best_model, logs = train_model(resnet18, trainloader, criterion, optimizer18, 5)


