# 🚀 MyTorch - 从零手写pytorch.nn

## 👨💻 作者：海蓬莱 &   

## 📂 目录结构

```
├── exercise/
│   ├── Lab_.ipynb  # 在这里找到TODO位置/回答问题
│   ├── example     # 在这里写TODO理解项目结构
│   ├── mytorch/    # 在这里写TODO实现算子
│   ├── test/       # 在这里写TODO实现测试
|   └── model       # 在这里写TODO用自己的算子训练模型 
│   ...
└── solution/       # 标准答案
    ...
```

## ⬇️ 如何下载

方法一：直接下载ZIP
1. 点击页面中绿色按钮 `Code`
2. 选择 `Download ZIP` 即可

方法二：使用Git克隆
```bash
git clone https://github.com/HaiPenglai/MyTorch.git
```

## 🛠️ 如何安装mytorch

下载完成后

```shell
cd MyTorch\solution
pip install -e mytorch
```

即可安装参考答案实现的mytorch，自己的mytorch需要在`MyTorch\exercise`中实现后同理安装


检查安装正确性:

```python
import mytorch
print(mytorch.__version__)
```

## ❓ 环境装不上怎么办

去看Lab0视频，手把手在新电脑上安装环境

## 🧠 MyTorch介绍

MyTorch是一个Python实现的简化版深度学习框架，支持`pip install`一键安装。我们在不使用`torch.nn`的情况下：

✅ 从零实现了多种神经网络：

✅ 用自己实现的神经网络在多种数据集上训练了模型：

## 🎯 项目介绍

本项目由双人讲解，通过现场手绘示意图、提出问题、回答问题、填写TODO、测试验证的方式进行

学习路径建议：
1. 下载项目代码
2. 观看教学视频
3. 在exercise中实践
4. 在solution验证答案

## 📖 Lab介绍
- Lab0:手把手安装环境
- Lab1:了解pytorch的文件结构，了解单元测试
- Lab2: