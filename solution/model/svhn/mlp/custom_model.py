# svhn_mlp_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mytorch import mynn

# 1. 数据预处理和加载
def load_svhn(batch_size=128):
    """
    加载SVHN数据集并返回数据加载器
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
    ])
    
    # 下载并加载训练集
    train_set = datasets.SVHN(
        root='./data', 
        split='train',
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # 下载并加载测试集
    test_set = datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 2. 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size=32 * 32 * 3, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = mynn.MyLinear(input_size, hidden_size)
        self.relu = mynn.MyReLU()
        self.fc2 = mynn.MyLinear(hidden_size, hidden_size//2)
        self.fc3 = mynn.MyLinear(hidden_size//2, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 3. 训练和评估函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    print(f'Train Epoch: {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return accuracy

# 4. 主函数
def main():
    # 超参数设置
    batch_size = 128
    epochs = 20
    learning_rate = 0.001
    hidden_size = 512
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, test_loader = load_svhn(batch_size)
    
    # 初始化模型
    model = MLP(hidden_size=hidden_size).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    test_accuracies = []
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, test_loader, criterion)
        test_accuracies.append(acc)
    
    # 绘制测试准确率曲线
    plt.plot(range(1, epochs+1), test_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('MLP Performance on SVHN')
    plt.savefig('svhn_mlp_accuracy.png')
    plt.show()

if __name__ == '__main__':
    main()