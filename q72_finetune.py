import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


batch_size = 32


counter = 0
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

trainset_loader = datasets.ImageFolder(root='../data/oxford-flowers17/train', transform=data_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


testset_loader = datasets.ImageFolder(root='../data/oxford-flowers17/train', transform=data_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(1000, 17)

    def forward(self, x):
        x = x.view(-1, 1000)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class Model(nn.Module):
    def __init__(self, vgg):
        super(Model, self).__init__()
        self.model = vgg
        self.classifier17 = Classifier()
    def forward(self, x):
        x = self.model.forward(x)
        x = self.classifier17.forward(x)
        return x

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


model = models.resnet18(pretrained=True)
print(model)
model = Model(model).to(device)

print(len(trainset_loader))
print(len(testset_loader))

optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.7)

trainLoss = []
trainAcc = []
def train(epoch, log_interval=1):
    model.train()  # set training mode
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            #print(data.size())
            output = model(data)
            #print(output.size(), target.size())
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if iteration % log_interval == 0:
                trainLoss.append(loss.item())
                model.eval()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        test()

def test():
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    acc = correct / len(testset_loader.dataset)
    trainAcc.append(acc)
    trainLoss.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))

def save_checkpoint(model, optimizer, checkpoint_path="q72checkpoint.cp"):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

#load_checkpoint("q72checkpoint.cp", model, optimizer)
train(100)
save_checkpoint(model, optimizer)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.subplot(121)
plt.plot(trainAcc, label="Training Accuracy")
plt.title("Training Accuracy")
plt.subplot(122)
plt.plot(trainLoss, label="Training Loss")
plt.title("Training Loss")
plt.show()
fig.savefig(fname="../report/fig/q721.eps")
