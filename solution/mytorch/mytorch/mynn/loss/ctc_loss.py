import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean'):
        super(CustomCTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # 使用F.ctc_loss作为底层实现
        loss = F.ctc_loss(
            log_probs, 
            targets, 
            input_lengths, 
            target_lengths,
            blank=self.blank, 
            reduction=self.reduction,
            zero_infinity=False
        )
        
        return loss