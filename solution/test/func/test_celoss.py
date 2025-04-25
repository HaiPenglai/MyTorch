import torch
import torch.nn as nn
from mytorch.mynn import CustomCrossEntropyLoss

def test_cross_entropy():
    # 配置参数
    torch.manual_seed(42)
    
    # 创建测试数据
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    targets = torch.tensor([1, 2])  # 第二个类和第三个类
    
    # 原生交叉熵损失
    criterion = nn.CrossEntropyLoss()
    loss_native = criterion(logits, targets)
    print("nn.CrossEntropyLoss Output:", loss_native.item())
    
    # 自定义交叉熵损失
    custom_criterion = CustomCrossEntropyLoss()
    loss_custom = custom_criterion(logits, targets)
    print("CustomCrossEntropyLoss Output:", loss_custom.item())
    
    # 比较结果
    is_close = torch.allclose(loss_custom, loss_native, atol=1e-6)
    print("\n损失值是否一致:", is_close)
    
    # 测试极端值
    extreme_logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
    extreme_targets = torch.tensor([1])
    
    print("\n测试极端值:")
    print("nn.CrossEntropyLoss 极端值输出:", criterion(extreme_logits, extreme_targets).item())
    print("CustomCrossEntropyLoss 极端值输出:", custom_criterion(extreme_logits, extreme_targets).item())
    
    # 测试reduction参数
    print("\n测试reduction参数:")
    custom_criterion_sum = CustomCrossEntropyLoss(reduction='sum')
    print("sum reduction:", custom_criterion_sum(logits, targets).item())
    
    custom_criterion_none = CustomCrossEntropyLoss(reduction='none')
    print("none reduction:", custom_criterion_none(logits, targets))

if __name__ == '__main__':
    test_cross_entropy()