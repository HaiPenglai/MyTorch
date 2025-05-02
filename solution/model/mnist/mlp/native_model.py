from openai import images
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 创建数据集目录
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
os.makedirs(dataset_path, exist_ok=True)

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # TODO:解释这里的数字的意思 【答案：数据集的均值和标准差，通过减去均值，除以标准差的方式进行归一化】
])

# 下载并加载MNIST数据集，如果没有下载过会联网下载(可能需要代理)
train_dataset = datasets.MNIST(
    download=True,
    root=dataset_path,
    train=True,
    transform=transform
)

def show_single_image(dataset):
    img, label = dataset[0]  
    img = img.squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

show_single_image(train_dataset)

# TODO仿照训练数据集，定义测试数据集
'''《
test_dataset = None
》'''
#《
test_dataset = datasets.MNIST(
    download=True,
    root=dataset_path,
    train=False,
    transform=transform
)
#》

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

images, labels = next(iter(train_loader))
print(f"{images[:2].shape=}\n{images[:2]}") # TODO问题：解释images[:2].shape=torch.Size([2, 1, 28, 28]) 【答案：总共2张图像，每张通道数为1，因为黑白色，大小为28*28】
print(f"{labels[:2].shape=}\n{labels[:2]}") # TODO问题：解释labels[:2].shape=torch.Size([2]) 【答案：2张图像的标签，数字范围0~9】
# TODO问题：为什么输出的图片有很多重复数字 【答案：因为图像边缘是黑色的】

# 定义简单的MLP模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        # TODO，填写一个线性层，使模型能正确前向传播
        '''《pass》'''
        #《
        self.fc2 = nn.Linear(128, 64)
        #》
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #TODO 把图像展开为一条向量
        '''《pass》'''
        #《
        x = x.view(-1, 28 * 28)
        #》
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleMLP()
#TODO 选择交叉熵损失
'''《
criterion = None
》'''
#《
criterion = nn.CrossEntropyLoss()
#》
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        #TODO 模型前向传播
        '''《
        output = None
        》'''
        #《
        output = model(data)
        #》
        loss = criterion(output, target)
        #TODO 1.计算梯度并累计到参数的.grad属性，2.梯度下降
        '''《pass》'''
        #《
        loss.backward()
        optimizer.step()
        #》
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            #TODO 获取概率最大的预测， 
            '''《
            pred = None
            》'''
            #《
            pred = output.argmax(dim=1)
            #》
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    #TODO计算准确率，之前算出了预测正确的样本数，除以测试集的样本个数[注意，len(test_loader)表示批次数，len(test_loader.dataset)才是样本总数]， 大约0.95就是正常的
    '''《
    accuracy = None
    》'''
    #《
    accuracy = correct / len(test_loader.dataset)
    #》
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100 * accuracy:.2f}%)\n')

# 训练和测试
for epoch in range(1, 3):  # 训练2个epoch
    train(epoch)
    test()

# 保存模型
model_state_dict = model.state_dict()
#TODO 打印模型的状态字典, 提示model_state_dict.items()可以获得key, param键值对
'''《pass》'''
#《
for key, param in model_state_dict.items():
    print(f"{key}, {param.shape}")
#》
torch.save(model_state_dict, os.path.join(os.path.dirname(__file__), 'mnist_mlp.pth'))
print("训练完成，模型已保存!")