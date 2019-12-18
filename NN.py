import torch
from torch import max_pool1d
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD

from CustomLoader import CustomDataset


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = Sequential(
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = Dropout()

        # x = torch.randn(3, 256, 256).view(-1, 3, 256, 256)
        # self.convs(x)

        self.fc1 = Linear(262144, 1000)
        self.fc2 = Linear(1000, 4)

    def convs(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        print(out.size())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


trainset = CustomDataset(train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = CustomDataset(train=False)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

classes = ('y1', 'y2', 'y3', 'y4')

model = ConvNet()

# Loss and optimizer
criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

total_step = len(trainloader)
loss_list = []
acc_list = []
num_epochs = 1


# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
#
# print(input)
# print(target)
# output = criterion(input, target)
#
# print(output)


def to_one_hot(y, n_dims=None):
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = 4
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


for epoch in range(num_epochs):
    for i, (labels, images) in enumerate(trainloader):
        # Run the forward pass
        outputs = model(images)
        # print(outputs)
        labels -= 1
        print(outputs, labels)

        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# for i, (lab, img) in enumerate(trainloader):
#     # print(i, lab, img)
#     # print(i, lab.size(), img.size())
#     outputs = model(img)
#     print(outputs)
#     break
