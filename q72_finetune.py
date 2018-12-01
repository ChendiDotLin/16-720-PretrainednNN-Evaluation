import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image


class FLOWER(Dataset):
    """
    A customized data loader for MNIST.
    """

    def __init__(self, train=True):
        self.images = None
        self.labels = None
        self.filenames = []
        if train:
            self.root = '../data/oxford-flowers17/train'
        else:
            self.root = '../data/oxford-flowers17/test'
        self.transform = transforms.ToTensor()

        # read filenames
        for i in range(10):
            filenames = glob.glob(osp.join(self.root, str(i), '*.jpg'))
            for fn in filenames:
                self.filenames.append((fn, i))  # (filename, label) pair

        # if preload dataset into memory
        self._preload()

        self.len = len(self.filenames)

    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:
            # load images
            image = Image.open(image_fn)
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()
            self.labels.append(label)

    def __getitem__(self, index):

        image = self.images[index]
        image = image.resize((224,224))
        label = self.labels[index]
        image = self.transform(image)
        # return image and label
        return image, label

    def __len__(self):
        return self.len

# Create the MNIST dataset.
# transforms.ToTensor() automatically converts PIL images to
# torch tensors with range [0, 1]
trainset = FLOWER(train=True)
# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)

# load the testset
testset = FLOWER(train=False)
# Use the torch dataloader to iterate through the dataset
testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)


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
        self.features = nn.Sequential(
            # stop at conv4
            *list(vgg.features.children())[:],
	        *list(vgg.classifier.children())[:],
            Classifier()
        )
    def forward(self, x):
        x = self.features(x)
        return x

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


model = models.squeezenet1_1(pretrained=True)
model = Model(model).to(device)



print(len(trainset))
print(len(testset))

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
