import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
palette = [[0],[1]]

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    # print("1",shape.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    # print("3",result.shape)
    input = input.to(torch.int64)
    result = result.scatter(1,input.cpu(),1)
    # print(input)
    # print("4",result)
    return result.cuda()


class SegmentationLosses(nn.Module):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=True):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'tversky':
            return self.tverskyLoss
        elif mode == 'dice':
            return self.diceLoss
        elif mode == 'our':
            return self.ourLoss
        elif mode == 'bce':
            return self.BCELoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()
        logpt =criterion(logit, target.long())
        pt =torch.exp(-logpt)
        if alpha is not None:
            logpt *= alpha
        loss = ((1 - pt) ** gamma) * logpt
        return loss

    def diceLoss(self, logit, target):
        ep = 1
        input_logit = logit[:,0]
        input_logit = torch.sigmoid(input_logit)
        intersection = 2 * torch.sum(input_logit * target) + ep
        union = torch.sum(input_logit) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss

    def tverskyLoss(self, logit, target,beta=0.7):
        smooth = 1
        input_logit = logit[:, 0]
        input_logit = torch.sigmoid(input_logit)
        alpha = 1.0 - beta
        tp = torch.sum(input_logit * target)
        fp = torch.sum((1 - target) * input_logit)
        fn = torch.sum(target * (1 - input_logit))
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        loss = 1 - tversky
        return loss


    def ourLoss(self, logit, target):
        log = logit
        tar = target
        loss = SegmentationLosses()
        tversky_loss = 1 -loss.tverskyLoss(log,tar)
        x = torch.pow(tversky_loss, 0.75)
        y = loss.CrossEntropyLoss(log, tar)
        x1 = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0) + y
        return x1 / 2

    def BCELoss(self, logit, target):
        criterion = nn.BCEWithLogitsLoss(weight=self.weight)
        if self.cuda:
            criterion = criterion.cuda()
        input_logit = logit[:, 0]
        loss= criterion(input_logit, target)

        return loss



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 2, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    c = torch.rand(1, 7, 7, 7).cuda()
    d = torch.rand(1, 3, 7, 7).cuda()
    print(loss.ourLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    # print(loss1(a, b))
    # print(loss.tversky_loss(a, d).item())





