# Copyright (c) 2022-present, Royal Bank of Canada.
# Copyright (c) 2022-present, Ximeng Sun
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################
# Code is based on the AdaShare (https://arxiv.org/pdf/1911.12423.pdf) implementation 
# from https://github.com/sunxm2357/AdaShare by Ximeng Sun
##################################################################################################



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

affine_par = True
from models.dynamic_conv import (
    Dynamic_conv1d,
)  # https://github.com/kaijieshi7/Dynamic-convolution-Pytorch
from models.TCN import (
    TemporalBlock,
)  # https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
from models.gumbelmodule import GumbelSoftmax

# ################################################
# ############## ResNet Modules ##################
# ################################################


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3,))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=False,
    )


def dy_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def tcn_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    kernel_size = 3
    full_padding = (kernel_size - 1) * dilation

    return TemporalBlock(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
    )


# No projection: identity shortcut
# conv -> bn -> relu -> conv -> bn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.planes = planes

        # 4 level channel usage: 0 -- 0%; 1 -- 25 %; 2 -- 50 %; 3 -- 100%
        self.keep_channels = (planes * np.cumsum([0, 0.25, 0.25, 0.5])).astype("int")
        self.keep_masks = []
        for kc in self.keep_channels:
            mask = np.zeros([1, planes, 1, 1])
            mask[:, :kc] = 1
            self.keep_masks.append(mask)
        self.keep_masks = torch.from_numpy(np.concatenate(self.keep_masks)).float()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes, affine=affine_par)

    def forward(self, x, keep=None):
        # keep: [batch_size], int
        cuda_device = x.get_device()
        out = self.conv1(x)
        out = self.bn1(out)

        # Used for deep elastic
        if keep is not None:
            keep = keep.long()
            bs, h, w = out.shape[0], out.shape[2], out.shape[3]
            # mask: [batch_size, c, 1, 1]
            mask = self.keep_masks[keep].to(cuda_device)
            # mask: [batch_size, c, h, w]
            mask = mask.repeat(1, 1, h, w)
            out = out * mask

        out = self.relu(out)
        out = self.conv2(out)
        y = self.bn2(out)
        return y


class BasicBlock_aig(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, temperature=1):
        super(BasicBlock_aig, self).__init__()
        self.planes = planes
        self.temperature = temperature

        # 4 level channel usage: 0 -- 0%; 1 -- 25 %; 2 -- 50 %; 3 -- 100%
        self.keep_channels = (planes * np.cumsum([0, 0.25, 0.25, 0.5])).astype("int")
        self.keep_masks = []
        for kc in self.keep_channels:
            mask = np.zeros([1, planes, 1, 1])
            mask[:, :kc] = 1
            self.keep_masks.append(mask)
        self.keep_masks = torch.from_numpy(np.concatenate(self.keep_masks)).float()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes, affine=affine_par)

        # https://github.com/andreasveit/convnet-aig
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expansion * planes),
            )
        # Gate layers
        self.fc1 = nn.Conv1d(inplanes, 16, kernel_size=1)
        self.fc1bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Conv1d(16, 2, kernel_size=1)
        self.gs = GumbelSoftmax()
        self.gs.cuda()

    def forward(self, x, keep=None):
        # keep: [batch_size], int
        cuda_device = x.get_device()

        # Compute relevance score - https://github.com/andreasveit/convnet-aig
        w = F.avg_pool1d(x, x.size(2))  # channel wise averaging
        w = self.relu(self.fc1bn(self.fc1(w)))
        w = self.fc2(w)
        # Sample from Gumble Module
        w = self.gs(w, temp=self.temperature, force_hard=True)

        out = self.conv1(x)
        out = self.bn1(out)

        # Used for deep elastic
        if keep is not None:
            keep = keep.long()
            bs, h, w = out.shape[0], out.shape[2], out.shape[3]
            # mask: [batch_size, c, 1, 1]
            mask = self.keep_masks[keep].to(cuda_device)
            # mask: [batch_size, c, h, w]
            mask = mask.repeat(1, 1, h, w)
            out = out * mask

        out = self.relu(out)
        out = self.conv2(out)
        y = self.bn2(out)

        return y, w[:, 1]


class BasicBlock_dynamic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock_dynamic, self).__init__()
        self.planes = planes

        # 4 level channel usage: 0 -- 0%; 1 -- 25 %; 2 -- 50 %; 3 -- 100%
        self.keep_channels = (planes * np.cumsum([0, 0.25, 0.25, 0.5])).astype("int")
        self.keep_masks = []
        for kc in self.keep_channels:
            mask = np.zeros([1, planes, 1, 1])
            mask[:, :kc] = 1
            self.keep_masks.append(mask)
        self.keep_masks = torch.from_numpy(np.concatenate(self.keep_masks)).float()

        self.conv1 = dy_conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = dy_conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes, affine=affine_par)

    def forward(self, x, keep=None):
        # keep: [batch_size], int
        cuda_device = x.get_device()
        out = self.conv1(x)
        out = self.bn1(out)

        # used for deep elastic
        if keep is not None:
            keep = keep.long()
            bs, h, w = out.shape[0], out.shape[2], out.shape[3]
            # mask: [batch_size, c, 1, 1]
            mask = self.keep_masks[keep].to(cuda_device)
            # mask: [batch_size, c, h, w]
            mask = mask.repeat(1, 1, h, w)
            out = out * mask

        out = self.relu(out)
        out = self.conv2(out)
        y = self.bn2(out)
        return y


class BasicBlock_TCN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock_TCN, self).__init__()
        self.planes = planes

        # 4 level channel usage: 0 -- 0%; 1 -- 25 %; 2 -- 50 %; 3 -- 100%
        self.keep_channels = (planes * np.cumsum([0, 0.25, 0.25, 0.5])).astype("int")
        self.keep_masks = []
        for kc in self.keep_channels:
            mask = np.zeros([1, planes, 1, 1])
            mask[:, :kc] = 1
            self.keep_masks.append(mask)
        self.keep_masks = torch.from_numpy(np.concatenate(self.keep_masks)).float()

        self.conv1 = tcn_conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = tcn_conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes, affine=affine_par)

    def forward(self, x, keep=None):
        # keep: [batch_size], int
        cuda_device = x.get_device()
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        y = self.bn2(out)
        return y


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes, affine=affine_par)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)
        out = self.conv2(out1)
        y = self.bn2(out)

        return y, out1


# No projection: identity shortcut and atrous
class Bottleneck(nn.Module):
    expansion = 4

    # |----------------------------------------------------------------|
    # 1x1 conv -> bn -> relu -> 3x3 conv -> bn -> relu -> 1x1 conv -> bn -> relu
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )  # change
        self.bn1 = nn.BatchNorm1d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv1d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(planes * self.expansion, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class Bottleneck2(nn.Module):
    expansion = 4

    # |----------------------------------------------------------------|
    # 1x1 conv -> bn -> relu -> 3x3 conv -> bn -> relu -> 1x1 conv -> bn -> relu
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv1d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(planes * self.expansion, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out = self.conv2(out1)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out, out1


class Residual_Convolution(nn.Module):
    def __init__(self, icol, ocol, num_classes, rate):
        super(Residual_Convolution, self).__init__()
        self.conv1 = nn.Conv1d(
            icol, ocol, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True
        )
        self.conv2 = nn.Conv1d(
            ocol,
            num_classes,
            kernel_size=3,
            stride=1,
            padding=rate,
            dilation=rate,
            bias=True,
        )
        self.conv3 = nn.Conv1d(
            num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True
        )
        self.conv4 = nn.Conv1d(
            ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        dow1 = self.dropout(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        add1 = self.dropout(add1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg


# coarse and fine segmentation
class Residual_Refinement_Module(nn.Module):
    def __init__(self, inplanes, base_channels, num_classes, rate):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Convolution(inplanes, base_channels, num_classes, rate)
        self.RC2 = Residual_Convolution(inplanes, base_channels, num_classes, rate)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1 + seg2]


class Classification_Module(nn.Module):
    def __init__(self, inplanes, num_classes, rate=12):
        super(Classification_Module, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes,
            1024,
            kernel_size=3,
            stride=1,
            padding=rate,
            dilation=rate,
            bias=True,
        )
        self.conv2 = nn.Conv1d(1024, 1024, kernel_size=1)
        self.conv3 = nn.Conv1d(1024, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)

        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class Classification_Module_TCN(nn.Module):
    def __init__(self, inplanes, num_classes, rate=12):
        super(Classification_Module_TCN, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes,
            1024,
            kernel_size=3,
            stride=1,
            padding=2 * rate,
            dilation=rate,
            bias=True,
        )
        self.conv11 = nn.Conv1d(
            inplanes,
            num_classes,
            kernel_size=3,
            stride=1,
            padding=2 * rate,
            dilation=rate,
            bias=True,
        )
        self.chomp1 = Chomp1d(2 * rate)
        self.conv2 = nn.Conv1d(1024, 1024, kernel_size=2)
        self.chomp2 = Chomp1d(rate)
        self.conv3 = nn.Conv1d(1024, num_classes, kernel_size=2)
        self.chomp3 = Chomp1d(rate)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu,
            self.conv2,
            self.chomp2,
            self.relu,
            self.conv3,
            self.chomp3,
            self.relu,
            self.dropout,
        )
        self.net1 = nn.Sequential(self.conv11, self.chomp1, self.relu, self.dropout)

    def forward(self, x):
        out = self.net1(x)
        return out


if __name__ == "__main__":
    block = BasicBlock(64, 256, stride=1, dilation=1)
    block.cuda()
    test_x = torch.ones([8, 64, 128, 128]).cuda()
    keep = torch.randint(0, 3, (8,))
    output = block(test_x, keep)
