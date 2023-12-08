import torch
import torch.nn.functional as F
import torch.nn as nn

def cross_entropy2d(input, target, weight=None, reduction='none'):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(
            ht, wt), mode="bilinear", align_corners=True)

    # https://zhuanlan.zhihu.com/p/76583143
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    # https://www.cnblogs.com/marsggbo/p/10401215.html
    loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )

    return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    @staticmethod
    def make_one_hot(labels, classes):
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[
                                         2], labels.size()[3]).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
        return target

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = self.make_one_hot(
            target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        # print(output_flat.shape)
        # print(target_flat.shape)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss