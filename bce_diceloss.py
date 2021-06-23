# necessary to import necessary packages, module has own private namespace (like a local function)
import torch
from torch import nn


class BCEDiceLoss_Logits(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean', weight_dice=0.5):
        super().__init__() # changed to super()....
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.bcelogit = nn.BCEWithLogitsLoss() #ADDED BCEWITHLOGITLOSS
        self.weight = weight_dice # Weight of DiceLoss (1- Weight: BCELoss)

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # BCE Loss first
        bceloss = self.bcelogit(predict, target)

        # Diceloss: convert logits to probabilities
        predict = torch.sigmoid(predict) # ADDED TORCH SIGMOID ACCOUNTING FOR LOGITS INPUT OF MODEL
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        diceloss = 1 - num / den

        if self.reduction == 'mean':
            return self.weight*diceloss.mean() + (1-self.weight)*bceloss
        elif self.reduction == 'sum':
            return self.weight*diceloss.sum() + (1-self.weight)*bceloss
        elif self.reduction == 'none':
            return self.weight*diceloss + (1-self.weight)*bceloss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
