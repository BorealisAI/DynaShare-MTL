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

import sys

sys.path.insert(0, "..")
from models.base import *
import torch.nn.functional as F
from torch import nn
from scipy.special import softmax
from models.util import count_params, compute_flops
from models.dynamic_conv import (
    Dynamic_conv1d,
)  # https://github.com/kaijieshi7/Dynamic-convolution-Pytorch
from models.TCN import (
    TemporalBlock,
    TemporalConvNet,
)  # https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

import torch
import tqdm
import time
import copy
import numpy as np


class Deeplab_ResNet_Backbone(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 76
        super(Deeplab_ResNet_Backbone, self).__init__()
        self.conv1 = nn.Conv1d(76, 76, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(76, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=2, padding=0, ceil_mode=True)

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(
            zip(filt_sizes, layers, strides, dilations)
        ):
            blocks, ds = self._make_layer(
                block, filt_size, num_blocks, stride=stride, dilation=dilation
            )
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or dilation == 2
            or dilation == 4
        ):
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion, affine=affine_par),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample

    def forward(self, x, policy=None):
        if policy is None:
            # Forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # Apply the residual skip out of _make_layers_

                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # Do the block dropping
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(
                            -1, 1, 1, 1
                        ) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1 - policy[t])
                    t += 1
        return x


class MTL2(nn.Module):
    def __init__(
        self,
        block,
        layers,
        tasks,
        num_classes_tasks,
        init_method,
        init_neg_logits=None,
        skip_layer=0,
    ):
        super(MTL2, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)
        self.tasks = tasks

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(
                self,
                "task%d_fc1_c0" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=6),
            )
            setattr(
                self,
                "task%d_fc1_c1" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=12),
            )
            setattr(
                self,
                "task%d_fc1_c2" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=18),
            )
            setattr(
                self,
                "task%d_fc1_c3" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=24),
            )

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

        out = np.multiply(
            [num_classes_tasks[t] for t in range(self.num_tasks)], [1, 170, 170, 1]
        )
        # Parametrize inp
        inp = [22 * 25, 22 * 10, 22, 22]

        self.output_list = nn.ModuleList(
            [nn.Linear(inp[i], out[i]) for i in range(self.num_tasks)]
        )

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "logits" in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "backbone" in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "fc" in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ("task" in name and "logits" in name):
                params.append(param)
        return params

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(
                getattr(self, "task%d_logits" % (t_id + 1)),
                temperature,
                hard=hard_sampling,
            )
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, "task%d_logits" % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to(
                        "cuda:%d" % cuda_device
                    )
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                self.policys.append(policy)

        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == "all_chosen":
                assert self.init_neg_logits is not None
                task_logits = self.init_neg_logits * torch.ones(
                    num_layers - self.skip_layer, 2
                )
                task_logits[:, 0] = 0
            elif self.init_method == "random":
                task_logits = 1e-3 * torch.randn(num_layers - self.skip_layer, 2)
            elif self.init_method == "equal":
                task_logits = 0.5 * torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError(
                    "Init Method %s is not implemented" % self.init_method
                )

            self._arch_parameters = []
            self.register_parameter(
                "task%d_logits" % (t_id + 1),
                nn.Parameter(task_logits, requires_grad=True),
            )
            self._arch_parameters.append(getattr(self, "task%d_logits" % (t_id + 1)))

    def forward(
        self,
        img,
        temperature,
        is_policy,
        num_train_layers=None,
        hard_sampling=False,
        mode="train",
    ):

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = img.get_device()
        if is_policy:
            if mode == "train":
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == "eval":
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == "fix_policy":
                for p in self.policys:
                    assert p is not None
            else:
                raise NotImplementedError("mode %s is not implemented" % mode)

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            padding[:, 1] = 0

            padding_policys = []
            feats = []
            for t_id in range(self.num_tasks):
                padding_policy = torch.cat(
                    (padding.float(), self.policys[t_id][-num_train_layers:].float()),
                    dim=0,
                )
                padding_policys.append(padding_policy)

                feats.append(self.backbone(img, padding_policy))
        else:
            feats = [self.backbone(img)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            aux = (
                getattr(self, "task%d_fc1_c0" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c1" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c2" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c3" % (t_id + 1))(feats[t_id])
            )

            if self.tasks[t_id] == "pheno" or self.tasks[t_id] == "ihm":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                output = self.output_list[t_id](aux)
            else:
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                aux = self.output_list[t_id](aux)
                output = aux.reshape(s0, s1, 170)

            outputs.append(output)
        return outputs, self.policys, [None] * self.num_tasks


class Deeplab_ResNet_Backbone_AIG(nn.Module):
    def __init__(self, block, layers, dataset):
        if dataset == "mimic":
            self.inplanes = 76  # For mimic
            super(Deeplab_ResNet_Backbone_AIG, self).__init__()
            self.conv1 = nn.Conv1d(
                76, 76, kernel_size=2, stride=2, padding=0, bias=False
            )
            self.bn1 = nn.BatchNorm1d(76, affine=affine_par)
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
            filt_sizes = [64, 128, 256, 512]
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(
                kernel_size=1, stride=2, padding=0, ceil_mode=True
            )  # change
        elif dataset == "census":
            self.inplanes = 76  # For census
            super(Deeplab_ResNet_Backbone_AIG, self).__init__()
            self.conv1 = nn.Conv1d(
                1, 76, kernel_size=2, stride=2, padding=0, bias=False
            )
            self.bn1 = nn.BatchNorm1d(76, affine=affine_par)
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
            filt_sizes = [64, 128, 256, 512]
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(
                kernel_size=1, stride=2, padding=0, ceil_mode=True
            )
        else:
            self.inplanes = 32  # For pcba
            super(Deeplab_ResNet_Backbone_AIG, self).__init__()
            self.conv1 = nn.Conv1d(
                1, 32, kernel_size=2, stride=2, padding=0, bias=False
            )
            self.bn1 = nn.BatchNorm1d(32, affine=affine_par)
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
            filt_sizes = [64, 128, 256, 512]
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )

            self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(
            zip(filt_sizes, layers, strides, dilations)
        ):
            blocks, ds = self._make_layer(
                block, filt_size, num_blocks, stride=stride, dilation=dilation
            )
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or dilation == 2
            or dilation == 4
        ):
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion, affine=affine_par),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample

    def forward(self, x, policy=None):
        gate_activations = []
        if policy is None:
            # Forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # Apply the residual skip out of _make_layers_

                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )

                    x, a = self.blocks[segment][b](x)
                    gate_activations.append(a)
                    x = F.relu(residual + x)

        else:
            # Do the block dropping
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )
                    x, a = self.blocks[segment][b](x)
                    gate_activations.append(a)
                    fx = F.relu(residual + x)
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(
                            -1, 1, 1, 1
                        ) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1 - policy[t])
                    t += 1

        return x, gate_activations


class MTL2_AIG(nn.Module):
    def __init__(
        self,
        block,
        layers,
        tasks,
        num_classes_tasks,
        init_method,
        dataset,
        init_neg_logits=None,
        skip_layer=0,
    ):
        super(MTL2_AIG, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone_AIG(block, layers, dataset)
        self.num_tasks = len(num_classes_tasks)
        self.tasks = tasks

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(
                self,
                "task%d_fc1_c0" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=6),
            )
            setattr(
                self,
                "task%d_fc1_c1" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=12),
            )
            setattr(
                self,
                "task%d_fc1_c2" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=18),
            )
            setattr(
                self,
                "task%d_fc1_c3" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=24),
            )

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()
        self.convpcba = nn.Conv1d(512, 1, kernel_size=1)

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

        if dataset == "mimic":
            out = np.multiply(
                [num_classes_tasks[t] for t in range(self.num_tasks)], [1, 170, 170, 1]
            )  # For mimic
            inp = [22 * 25, 22 * 10, 22, 22]

        elif dataset == "census":
            out = np.multiply(
                [num_classes_tasks[t] for t in range(self.num_tasks)],
                [1 for t in range(self.num_tasks)],
            )  # for pcba
            inp = [61 for t in range(self.num_tasks)]

        elif dataset == "pcba":
            out = np.multiply(
                [num_classes_tasks[t] for t in range(self.num_tasks)],
                [1 for t in range(self.num_tasks)],
            )  # For pcba
            inp = [128 for t in range(self.num_tasks)]

        self.output_list = nn.ModuleList(
            [nn.Linear(inp[i], out[i]) for i in range(self.num_tasks)]
        )

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "logits" in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "backbone" in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "fc" in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ("task" in name and "logits" in name):
                params.append(param)
        return params

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(
                getattr(self, "task%d_logits" % (t_id + 1)),
                temperature,
                hard=hard_sampling,
            )
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, "task%d_logits" % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to(
                        "cuda:%d" % cuda_device
                    )
                else:
                    policy = torch.from_numpy(np.array(single_policys))

                self.policys.append(policy)

        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == "all_chosen":
                assert self.init_neg_logits is not None
                task_logits = self.init_neg_logits * torch.ones(
                    num_layers - self.skip_layer, 2
                )
                task_logits[:, 0] = 0
            elif self.init_method == "random":
                task_logits = 1e-3 * torch.randn(num_layers - self.skip_layer, 2)
            elif self.init_method == "equal":
                task_logits = 0.5 * torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError(
                    "Init Method %s is not implemented" % self.init_method
                )

            self._arch_parameters = []
            self.register_parameter(
                "task%d_logits" % (t_id + 1),
                nn.Parameter(task_logits, requires_grad=True),
            )
            self._arch_parameters.append(getattr(self, "task%d_logits" % (t_id + 1)))

    def forward(
        self,
        img,
        temperature,
        is_policy,
        num_train_layers=None,
        hard_sampling=False,
        mode="train",
    ):

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = img.get_device()
        if is_policy:
            if mode == "train":
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == "eval":
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == "fix_policy":
                for p in self.policys:
                    assert p is not None
            else:
                raise NotImplementedError("mode %s is not implemented" % mode)

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            padding[:, 1] = 0

            padding_policys = []
            feats = []
            ga = []
            for t_id in range(self.num_tasks):
                padding_policy = torch.cat(
                    (padding.float(), self.policys[t_id][-num_train_layers:].float()),
                    dim=0,
                )
                padding_policys.append(padding_policy)

                feats.append(self.backbone(img, padding_policy)[0])
                ga.append(self.backbone(img, padding_policy)[1])
        else:
            feats = [self.backbone(img)[0]] * self.num_tasks
            ga = [self.backbone(img)[1]] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            if "PCBA" not in self.tasks[0]:
                aux = (
                    getattr(self, "task%d_fc1_c0" % (t_id + 1))(feats[t_id])
                    + getattr(self, "task%d_fc1_c1" % (t_id + 1))(feats[t_id])
                    + getattr(self, "task%d_fc1_c2" % (t_id + 1))(feats[t_id])
                    + getattr(self, "task%d_fc1_c3" % (t_id + 1))(feats[t_id])
                )

            if self.tasks[t_id] == "pheno" or self.tasks[t_id] == "ihm":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                output = self.output_list[t_id](aux)
            elif "PCBA" in self.tasks[0]:
                aux = self.convpcba(feats[t_id])
                output = self.output_list[t_id](aux)
            elif (
                self.tasks[t_id] == "income"
                or self.tasks[t_id] == "education"
                or self.tasks[t_id] == "marital"
            ):
                output = self.output_list[t_id](aux)
            else:
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                aux = self.output_list[t_id](aux)
                output = aux.reshape(s0, s1, 170)

            outputs.append(output)
        return outputs, self.policys, [None] * self.num_tasks, ga


def conv1x1(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
    return Dynamic_conv1d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
    )


class Deeplab_ResNet_Backbone_dynamic(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 76
        super(Deeplab_ResNet_Backbone_dynamic, self).__init__()
        self.conv1 = conv1x1(76, 76, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(76, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=2, padding=0, ceil_mode=True)

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(
            zip(filt_sizes, layers, strides, dilations)
        ):
            blocks, ds = self._make_layer(
                block, filt_size, num_blocks, stride=stride, dilation=dilation
            )
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or dilation == 2
            or dilation == 4
        ):
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion, affine=affine_par),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return layers, downsample

    def forward(self, x, policy=None):
        if policy is None:
            # Forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # Apply the residual skip out of _make_layers_

                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # Do the block dropping
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(
                            -1, 1, 1, 1
                        ) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1 - policy[t])
                    t += 1
        return x


class MTL2_Dynamic(nn.Module):
    def __init__(
        self,
        block,
        layers,
        tasks,
        num_classes_tasks,
        init_method,
        init_neg_logits=None,
        skip_layer=0,
    ):
        super(MTL2_Dynamic, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone_dynamic(block, layers)
        self.num_tasks = len(num_classes_tasks)
        self.tasks = tasks

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(
                self,
                "task%d_fc1_c0" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=6),
            )
            setattr(
                self,
                "task%d_fc1_c1" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=12),
            )
            setattr(
                self,
                "task%d_fc1_c2" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=18),
            )
            setattr(
                self,
                "task%d_fc1_c3" % (t_id + 1),
                Classification_Module(512 * block.expansion, num_class, rate=24),
            )

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

        out = np.multiply(
            [num_classes_tasks[t] for t in range(self.num_tasks)], [1, 170, 170, 1]
        )
        inp = [22 * 25, 22 * 10, 22, 22]
        self.output_list = nn.ModuleList(
            [nn.Linear(inp[i], out[i]) for i in range(self.num_tasks)]
        )

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "logits" in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "backbone" in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "fc" in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ("task" in name and "logits" in name):
                params.append(param)
        return params

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(
                getattr(self, "task%d_logits" % (t_id + 1)),
                temperature,
                hard=hard_sampling,
            )
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, "task%d_logits" % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to(
                        "cuda:%d" % cuda_device
                    )
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                self.policys.append(policy)

        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == "all_chosen":
                assert self.init_neg_logits is not None
                task_logits = self.init_neg_logits * torch.ones(
                    num_layers - self.skip_layer, 2
                )
                task_logits[:, 0] = 0
            elif self.init_method == "random":
                task_logits = 1e-3 * torch.randn(num_layers - self.skip_layer, 2)
            elif self.init_method == "equal":
                task_logits = 0.5 * torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError(
                    "Init Method %s is not implemented" % self.init_method
                )

            self._arch_parameters = []
            self.register_parameter(
                "task%d_logits" % (t_id + 1),
                nn.Parameter(task_logits, requires_grad=True),
            )
            self._arch_parameters.append(getattr(self, "task%d_logits" % (t_id + 1)))

    def forward(
        self,
        img,
        temperature,
        is_policy,
        num_train_layers=None,
        hard_sampling=False,
        mode="train",
    ):

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = copy.deepcopy(img.get_device())

        if is_policy:
            if mode == "train":
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == "eval":
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == "fix_policy":
                for p in self.policys:
                    assert p is not None
            else:
                raise NotImplementedError("mode %s is not implemented" % mode)

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            padding[:, 1] = 0

            padding_policys = []
            feats = []
            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    padding_policy = torch.cat(
                        (
                            padding.float().to(cuda_device),
                            self.policys[t_id][-num_train_layers:]
                            .float()
                            .to(cuda_device),
                        ),
                        dim=0,
                    )
                else:
                    padding_policy = torch.cat(
                        (
                            padding.float(),
                            self.policys[t_id][-num_train_layers:].float(),
                        ),
                        dim=0,
                    )
                padding_policys.append(padding_policy)

                feats.append(self.backbone(img, padding_policy))
        else:
            feats = [self.backbone(img)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            aux = (
                getattr(self, "task%d_fc1_c0" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c1" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c2" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c3" % (t_id + 1))(feats[t_id])
            )

            if self.tasks[t_id] == "pheno" or self.tasks[t_id] == "ihm":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                output = self.output_list[t_id](aux)
            else:
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                aux = self.output_list[t_id](aux)
                output = aux.reshape(s0, s1, 170)

            outputs.append(output)
        return outputs, self.policys, [None] * self.num_tasks


class Deeplab_ResNet_Backbone_TCN(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 76
        super(Deeplab_ResNet_Backbone_TCN, self).__init__()
        self.conv1 = TemporalBlock(
            76, 76, kernel_size=2, stride=1, dilation=1, padding=1
        )
        self.bn1 = nn.BatchNorm1d(76, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=2, padding=0, ceil_mode=True)

        strides = [1, 1, 1, 1]
        dilations = [1, 2, 4, 8]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(
            zip(filt_sizes, layers, strides, dilations)
        ):
            blocks, ds = self._make_layer(
                block,
                filt_size,
                num_blocks,
                stride=stride,
                dilation=dilation,
                padding=(num_blocks - 1) * dilation,
            )
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, padding=0):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or dilation == 2
            or dilation == 4
        ):
            downsample = nn.Sequential(
                TemporalBlock(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                ),
                nn.BatchNorm1d(planes * block.expansion, affine=affine_par),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, padding=padding)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, padding=padding)
            )

        return layers, downsample

    def forward(self, x, policy=None):
        if policy is None:
            # Forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # Apply the residual skip out of _make_layers_
                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # Do the block dropping
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = (
                        self.ds[segment](x)
                        if b == 0 and self.ds[segment] is not None
                        else x
                    )
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(
                            -1, 1, 1, 1
                        ) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1 - policy[t])
                    t += 1
        return x


class MTL2_TCN(nn.Module):
    def __init__(
        self,
        policy_model,
        block,
        layers,
        tasks,
        num_classes_tasks,
        init_method,
        init_neg_logits=None,
        skip_layer=0,
    ):
        super(MTL2_TCN, self).__init__()
        self.backbone = Deeplab_ResNet_Backbone_TCN(block, layers)
        self.num_tasks = len(num_classes_tasks)
        self.tasks = tasks

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(
                self,
                "task%d_fc1_c0" % (t_id + 1),
                Classification_Module_TCN(512 * block.expansion, num_class, rate=6),
            )
            setattr(
                self,
                "task%d_fc1_c1" % (t_id + 1),
                Classification_Module_TCN(512 * block.expansion, num_class, rate=12),
            )
            setattr(
                self,
                "task%d_fc1_c2" % (t_id + 1),
                Classification_Module_TCN(512 * block.expansion, num_class, rate=18),
            )
            setattr(
                self,
                "task%d_fc1_c3" % (t_id + 1),
                Classification_Module_TCN(512 * block.expansion, num_class, rate=24),
            )

        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

        out = np.multiply(
            [num_classes_tasks[t] for t in range(self.num_tasks)], [1, 170, 170, 1]
        )
        if policy_model == "TCN":
            inp = [116 * 25, 116 * 10, 116, 116]
        else:
            inp = [22 * 25, 22 * 10, 22, 22]
        self.output_list = nn.ModuleList(
            [nn.Linear(inp[i], out[i]) for i in range(self.num_tasks)]
        )

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "logits" in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "backbone" in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "fc" in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ("task" in name and "logits" in name):
                params.append(param)
        return params

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(
                getattr(self, "task%d_logits" % (t_id + 1)),
                temperature,
                hard=hard_sampling,
            )
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, "task%d_logits" % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to(
                        "cuda:%d" % cuda_device
                    )
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                self.policys.append(policy)

        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == "all_chosen":
                assert self.init_neg_logits is not None
                task_logits = self.init_neg_logits * torch.ones(
                    num_layers - self.skip_layer, 2
                )
                task_logits[:, 0] = 0
            elif self.init_method == "random":
                task_logits = 1e-3 * torch.randn(num_layers - self.skip_layer, 2)
            elif self.init_method == "equal":
                task_logits = 0.5 * torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError(
                    "Init Method %s is not implemented" % self.init_method
                )

            self._arch_parameters = []
            self.register_parameter(
                "task%d_logits" % (t_id + 1),
                nn.Parameter(task_logits, requires_grad=True),
            )
            self._arch_parameters.append(getattr(self, "task%d_logits" % (t_id + 1)))

    def forward(
        self,
        img,
        temperature,
        is_policy,
        num_train_layers=None,
        hard_sampling=False,
        mode="train",
    ):

        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = img.get_device()
        if is_policy:
            if mode == "train":
                self.policys = self.train_sample_policy(temperature, hard_sampling)
            elif mode == "eval":
                self.policys = self.test_sample_policy(hard_sampling)
            elif mode == "fix_policy":
                for p in self.policys:
                    assert p is not None
            else:
                raise NotImplementedError("mode %s is not implemented" % mode)

            for t_id in range(self.num_tasks):
                if cuda_device != -1:
                    self.policys[t_id] = self.policys[t_id].to(cuda_device)
                else:
                    self.policys[t_id] = self.policys[t_id].cpu()

            skip_layer = sum(self.layers) - num_train_layers
            if cuda_device != -1:
                padding = torch.ones(skip_layer, 2).to(cuda_device)
            else:
                padding = torch.ones(skip_layer, 2)
            padding[:, 1] = 0

            padding_policys = []
            feats = []
            for t_id in range(self.num_tasks):
                padding_policy = torch.cat(
                    (
                        padding.float().to(cuda_device),
                        self.policys[t_id][-num_train_layers:].float().to(cuda_device),
                    ),
                    dim=0,
                )
                padding_policys.append(padding_policy)

                feats.append(self.backbone(img, padding_policy))
        else:
            feats = [self.backbone(img)] * self.num_tasks

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            aux = (
                getattr(self, "task%d_fc1_c0" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c1" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c2" % (t_id + 1))(feats[t_id])
                + getattr(self, "task%d_fc1_c3" % (t_id + 1))(feats[t_id])
            )

            if self.tasks[t_id] == "pheno" or self.tasks[t_id] == "ihm":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                output = self.output_list[t_id](aux)
            else:
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                aux = self.output_list[t_id](aux)
                output = aux.reshape(s0, s1, 170)

            outputs.append(output)
        return outputs, self.policys, [None] * self.num_tasks


class TCN(nn.Module):
    def __init__(
        self,
        input_size,
        tasks,
        num_classes_tasks,
        init_method,
        output_size,
        num_channels,
        kernel_size,
        dropout,
        init_neg_logits=None,
        skip_layer=0,
    ):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(
                self,
                "task%d_fc1_c0" % (t_id + 1),
                nn.Linear(num_channels[-1], output_size),
            )
            setattr(
                self,
                "task%d_fc1_c1" % (t_id + 1),
                nn.Linear(num_channels[-1], output_size),
            )
            setattr(
                self,
                "task%d_fc1_c2" % (t_id + 1),
                nn.Linear(num_channels[-1], output_size),
            )
            setattr(
                self,
                "task%d_fc1_c3" % (t_id + 1),
                nn.Linear(num_channels[-1], output_size),
            )

        self.num_tasks = len(num_classes_tasks)
        self.tasks = tasks
        self.layers = layers
        self.skip_layer = skip_layer
        self.init_method = init_method
        self.init_neg_logits = init_neg_logits
        self.reset_logits()

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "logits" in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "tcn" in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "task" in name and "fc" in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ("task" in name and "logits" in name):
                params.append(param)
        return params

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for t_id in range(self.num_tasks):
            policy = F.gumbel_softmax(
                getattr(self, "task%d_logits" % (t_id + 1)),
                temperature,
                hard=hard_sampling,
            )
            policys.append(policy)
        return policys

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            cuda_device = self.task1_logits.get_device()
            logits1 = self.task1_logits.detach().cpu().numpy()
            policy1 = np.argmax(logits1, axis=1)
            policy1 = np.stack((1 - policy1, policy1), dim=1)
            if cuda_device != -1:
                self.policy1 = torch.from_numpy(np.array(policy1)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy1 = torch.from_numpy(np.array(policy1))
            cuda_device = self.task2_logits.get_device()
            logits2 = self.task2_logits.detach().cpu().numpy()
            policy2 = np.argmax(logits2, axis=1)
            policy2 = np.stack((1 - policy2, policy2), dim=1)
            if cuda_device != -1:
                self.policy2 = torch.from_numpy(np.array(policy2)).to(
                    "cuda:%d" % cuda_device
                )
            else:
                self.policy2 = torch.from_numpy(np.array(policy2))
        else:
            for t_id in range(self.num_tasks):
                task_logits = getattr(self, "task%d_logits" % (t_id + 1))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to(
                        "cuda:%d" % cuda_device
                    )
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                self.policys.append(policy)

        return self.policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for t_id in range(self.num_tasks):
            if self.init_method == "all_chosen":
                assert self.init_neg_logits is not None
                task_logits = self.init_neg_logits * torch.ones(
                    num_layers - self.skip_layer, 2
                )
                task_logits[:, 0] = 0
            elif self.init_method == "random":
                task_logits = 1e-3 * torch.randn(num_layers - self.skip_layer, 2)
            elif self.init_method == "equal":
                task_logits = 0.5 * torch.ones(num_layers - self.skip_layer, 2)
            else:
                raise NotImplementedError(
                    "Init Method %s is not implemented" % self.init_method
                )

            self._arch_parameters = []
            self.register_parameter(
                "task%d_logits" % (t_id + 1),
                nn.Parameter(task_logits, requires_grad=True),
            )
            self._arch_parameters.append(getattr(self, "task%d_logits" % (t_id + 1)))

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # Input should have dimension (N, C, L)

        # Get the output
        outputs = []
        for t_id in range(self.num_tasks):
            aux = (
                getattr(self, "task%d_fc1_c0" % (t_id + 1))(y1[:, :, -1])
                + getattr(self, "task%d_fc1_c1" % (t_id + 1))(y1[:, :, -1])
                + getattr(self, "task%d_fc1_c2" % (t_id + 1))(y1[:, :, -1])
                + getattr(self, "task%d_fc1_c3" % (t_id + 1))(y1[:, :, -1])
            )

            if self.tasks[t_id] == "pheno" or self.tasks[t_id] == "ihm":
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                output = self.output_list[t_id](aux)
            else:
                s0, s1, s2 = aux.shape
                aux = aux.reshape(s0, s1 * s2)
                aux = self.output_list[t_id](aux)
                output = aux.reshape(s0, s1, 170)

            outputs.append(output)
        return outputs, self.policys, [None] * self.num_tasks


if __name__ == "__main__":

    backbone = "ResNet34"
    tasks_num_class = [25, 10, 1, 1]
    if backbone == "ResNet18":
        layers = [2, 2, 2, 2]
        block = BasicBlock
    elif backbone == "ResNet34":
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif backbone == "ResNet101":
        block = Bottleneck
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("backbone %s is invalid" % backbone)

    net = MTL2_Backbone(block, layers, tasks_num_class, "equal")

    img = torch.ones((1, 3, 224, 224))

    count_params(net.backbone)


    if len(tasks_num_class) == 3:
        policy1 = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype("float")
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype("float")
        policy2 = torch.from_numpy(policy2).cuda()

        policy3 = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,])
        policy3 = np.stack([policy3, 1 - policy3], axis=1).astype("float")
        policy3 = torch.from_numpy(policy3).cuda()
        policys = [policy1, policy2, policy3]
    elif len(tasks_num_class) == 2 and backbone == "ResNet34":
        policy1 = np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype("float")
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype("float")
        policy2 = torch.from_numpy(policy2).cuda()
        policys = [policy1, policy2]
    elif len(tasks_num_class) == 2 and backbone == "ResNet18":
        policy1 = np.array([1, 0, 1, 1, 1, 1, 1, 1])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype("float")
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1, 1, 0, 1, 1])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype("float")
        policy2 = torch.from_numpy(policy2).cuda()
        policys = [policy1, policy2]
    elif len(tasks_num_class) == 5:
        policy1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
        policy1 = np.stack([policy1, 1 - policy1], axis=1).astype("float")
        policy1 = torch.from_numpy(policy1).cuda()
        policy2 = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,])
        policy2 = np.stack([policy2, 1 - policy2], axis=1).astype("float")
        policy2 = torch.from_numpy(policy2).cuda()
        policy3 = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
        policy3 = np.stack([policy3, 1 - policy3], axis=1).astype("float")
        policy3 = torch.from_numpy(policy3).cuda()
        policy4 = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
        policy4 = np.stack([policy4, 1 - policy4], axis=1).astype("float")
        policy4 = torch.from_numpy(policy4).cuda()
        policy5 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1,])
        policy5 = np.stack([policy5, 1 - policy5], axis=1).astype("float")
        policy5 = torch.from_numpy(policy5).cuda()
        policys = [policy1, policy2, policy3, policy4, policy5]
    else:
        raise ValueError

    setattr(net, "policys", policys)

    times = []
    input_dict = {"temperature": 5, "is_policy": True, "mode": "fix_policy"}
    net.cuda()
    for _ in tqdm.tqdm(range(1000)):
        start_time = time.time()
        img = torch.rand((1, 3, 224, 224)).cuda()
        net(img, **input_dict)
        times.append(time.time() - start_time)

    print("Average time = ", np.mean(times))

    gflops = compute_flops(
        net, img.cuda(), {"temperature": 5, "is_policy": True, "mode": "fix_policy"}
    )
    print("Number of FLOPs = %.2f G" % (gflops / 1e9 / 2))

