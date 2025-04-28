import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from mytorch import mynn

torch.manual_seed(42)

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
os.makedirs(dataset_path, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root=dataset_path,
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root=dataset_path,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        #TODO 把所有模型的层换成自己的实现
        '''《
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        》'''
        #《
        self.fc1 = mynn.CustomLinear(28 * 28, 128)
        self.fc2 = mynn.CustomLinear(128, 64)
        self.fc3 = mynn.CustomLinear(64, 10)
        self.relu = mynn.CustomReLU()
        #》
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleMLP()
# TODO把损失函数换成自己的实现
'''《
criterion = nn.CrossEntropyLoss()
》'''
#《
criterion = mynn.CustomCrossEntropyLoss()
#》
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')

for epoch in range(1, 3):  
    train(epoch)
    test()

torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'mnist_mlp.pth'))
print("训练完成，模型已保存!")