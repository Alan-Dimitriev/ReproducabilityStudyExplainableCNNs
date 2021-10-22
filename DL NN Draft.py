import torch
import torch.nn as nn
import torch.nn.functional as F

# import statements from pytorch demo that includes learning rate

import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

batch_size = 32
lrs = []
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

dataset_train=datasets.MNIST('./data',train=True,download=True,transform=transform)
dataset_test=datasets.MNIST('./data',train=False,transform=transform)

train_loader=torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset_test,batch_size=batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 500, 5)
        self.con2 = nn.Conv2d(6, 500, 5)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class CNN_Attention(torch.nn.Module):

    def __init__(self, class_num):
        super(CNN_Attention, self).__init__()
        self.class_num = class_num
        self.filter_num = None
        self.Wa = None

    def build(self, input_shape):
        self.filter_num = input_shape[2]

        # self.Wa = (the number of classes, the number of filters)
        self.Wa = self.add_weight(shape=(self.class_num, self.filter_num),
                                  initializer=torch.nn.init.xavier_uniform(), trainable=True)

        super(CNN_Attention, self).build(input_shape)

    def call(self, inputs):
        # inputs_trans = (batch_size, the number of filters, sentence_length)
        inputs_trans = torch.transpose(inputs, [0, 2, 1])

        # at = (batch_size, the number of filters, sentence_length)
        at = torch.matmul(self.Wa, inputs_trans)

        # Softmax
        at = torch.exp(at - torch.max(at, dim=1, keepdim=True))
        at = at / torch.sum(at, dim=-1, keepdim=True)

        # weighted sum
        # v = (batch_size, the number of classes, the number of filters)
        v = torch.bmm(at, inputs)

        return v



def train(model, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # learning rate
        lr = optimizer.param_groups[0]["lr"] = 0.001

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} batch-{}\tLoss: {:.6f} Learning Rate: {}'.format(epoch, batch_idx, loss.item(), lr))
    lrs.append(lr)

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(
        test_loss,correct / len(test_loader.dataset)))
