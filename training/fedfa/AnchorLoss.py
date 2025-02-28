import torch, random
import torch.nn as nn
import torch.nn.functional as F


class AnchorLoss(nn.Module):
    """
    Unmodified AnchorLoss code from FedFA:
        https://ieeexplore.ieee.org/abstract/document/10286887
    """

    def __init__(self, cls_num, feature_num, ablation=0):
        """
        :param cls_num: class number
        :param feature_num: feature dimens
        """
        super().__init__()
        self.cls_num = cls_num
        self.feature_num = feature_num

        # initiate anchors
        if cls_num > feature_num:
            self.anchor = nn.Parameter(
                F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True
            )
        elif ablation == 1:
            self.anchor = nn.Parameter(
                F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True
            )
        elif ablation == 2:
            self.anchor = nn.Parameter(
                F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True
            )
            self.anchor.data = torch.load("utils/converged_anchors_data.pt")
        else:
            I = torch.eye(feature_num, feature_num)
            index = torch.LongTensor(random.sample(range(feature_num), cls_num))
            init = torch.index_select(I, 0, index)
            # for i in range(cls_num):
            #     if i % 2 == 0:
            #         init[i] = -init[i]
            self.anchor = nn.Parameter(init, requires_grad=True)

    def forward(self, feature, _target, Lambda=0.1):
        """
        :param feature: input
        :param _target: label/targets
        :return: anchor loss
        """
        # broadcast feature anchors for all inputs
        # centre = self.anchor.cuda().index_select(dim=0, index=_target.long())
        centre = self.anchor.index_select(dim=0, index=_target.long())
        # compute the number of samples in each class
        counter = torch.histc(_target, bins=self.cls_num, min=0, max=self.cls_num - 1)
        count = counter[_target.long()]
        centre_dis = feature - centre  # compute distance between input and anchors
        pow_ = torch.pow(centre_dis, 2)  # squre
        sum_1 = torch.sum(pow_, dim=1)  # sum all distance
        dis_ = torch.div(sum_1, count.float())  # mean by class
        sum_2 = torch.sum(dis_) / self.cls_num  # mean loss
        res = Lambda * sum_2  # time hyperparameter lambda
        return res
