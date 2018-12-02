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
from time import time


batch_size = 32

counter = 0
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

testset = datasets.ImageFolder(root='data/oxford-flowers102/test', transform=data_transform)
testset_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(1000, 102)

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
device = torch.device("cuda:3" if use_cuda else "cpu")
print(device)


model = models.squeezenet1_1(pretrained=False)
model = Model(model).to(device)

print(len(testset_loader))

optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.7)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

#load_checkpoint("checkpoints/squeezenet.npz.checkpoint")

model.eval()  # set evaluation mode

confusion = np.zeros((107, 107))

with torch.no_grad():
    for data, target in testset_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        print(pred.size(), target.size())
