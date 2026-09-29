import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
import numpy as np
import torch.nn as nn


# 交叉熵损失
def idh_cross_entropy(input, target, weight):
    return cross_entropy(input, target, weight=weight, ignore_index=-1)


# 关注更难分类的部分
class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=5, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def idh_focal_loss(input, target, weight):
    focalloss = FocalLoss(weight=weight)
    return focalloss(input, target)

def Hybrid_loss(input, target, weight):
    loss = 0.3*idh_focal_loss(input, target, weight) + 0.7*idh_cross_entropy(input, target, weight)
    return loss

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance.
        You can start with a small value and gradually increase it to observe the impact on the model's performance.
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.

        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem.
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies
        the impact of the logits and can be useful when dealing with highly imbalanced datasets.
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class LMFLoss(nn.Module):
    def __init__(self,  weight, cls_num_list=[331, 73], alpha=1, beta=1, gamma=2, max_m=0.5, s=30):
        super().__init__()
        self.focal_loss = FocalLoss(weight, gamma)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss

def idh_LMF_loss(input,target,weight):
    lmf_loss = LMFLoss(weight=weight)
    return lmf_loss(input,target)


if __name__ == "__main__":
    num_list = [565, 180]
    weight = torch.tensor([180, 565], dtype=torch.float).cuda()
    loss = LMFLoss( weight=weight, )

    pred = torch.randn(4, 2)
    pred = F.softmax(pred, dim=-1).cuda()
    truth = torch.tensor([0, 1, 1, 0]).cuda()
    final_loss = loss(pred, truth)
    print(final_loss)


