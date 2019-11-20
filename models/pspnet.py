#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import _ConvBatchNormReLU, _ResBlock
from models.resnet_orig import ResNetForPsp, Bottleneck


class _DilatedFCN(nn.Module):
    """ResNet-based Dilated FCN"""

    def __init__(self, n_blocks, use_bn=True):
        super(_DilatedFCN, self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', _ConvBatchNormReLU(3, 64, 3, 2, 1, 1, use_bn=use_bn)),
            ('conv2', _ConvBatchNormReLU(64, 64, 3, 1, 1, 1, use_bn=use_bn)),
            ('conv3', _ConvBatchNormReLU(64, 128, 3, 1, 1, 1, use_bn=use_bn)),
            ('pool', nn.MaxPool2d(3, 2, 1))
        ]))
        self.layer2 = _ResBlock(n_blocks[0], 128, 64, 256, 1, 1, use_bn=use_bn)
        self.layer3 = _ResBlock(
            n_blocks[1], 256, 128, 512, 2, 1, use_bn=use_bn)
        self.layer4 = _ResBlock(
            n_blocks[2], 512, 256, 1024, 1, 2, use_bn=use_bn)
        self.layer5 = _ResBlock(
            n_blocks[3], 1024, 512, 2048, 1, 4, use_bn=use_bn)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h1 = self.layer4(h)
        h2 = self.layer5(h1)
        if self.training:
            return h1, h2
        else:
            return h2


class _PyramidPoolModule(nn.Sequential):
    """Pyramid Pooling Module"""

    def __init__(self, in_channels, pyramids=[6, 3, 2, 1], use_bn=True):
        super(_PyramidPoolModule, self).__init__()
        out_channels = in_channels // len(pyramids)
        self.stages = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(output_size=p)),
                ('conv', _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1, use_bn=use_bn)), ]))
            for p in pyramids
        ])

    def forward(self, x):
        hs = [x]
        height, width = x.size()[2:]
        for stage in self.stages:
            h = stage(x)
            h = F.interpolate(h, (height, width),
                              mode='bilinear', align_corners=True)
            hs.append(h)
        return torch.cat(hs, dim=1)


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network"""
    # DEFAULTS ARE FOR CITYSCAPES

    def __init__(self, n_classes=19, n_blocks=[3, 4, 23, 3], pyramids=[6, 3, 2, 1], input_size=[713, 713], use_bn=True, output_features=False, output_all=False, use_original_base=False):

        super(PSPNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.output_features = output_features
        self.output_all = output_all
        if use_original_base:
            self.fcn = ResNetForPsp(Bottleneck, n_blocks)
        else:
            self.fcn = _DilatedFCN(n_blocks=n_blocks, use_bn=use_bn)
        self.ppm = _PyramidPoolModule(
            in_channels=2048, pyramids=pyramids, use_bn=use_bn)
        self.final = nn.Sequential(OrderedDict([
            ('conv5_4', _ConvBatchNormReLU(4096, 512, 3, 1, 1, 1, use_bn=use_bn)),
            ('drop5_4', nn.Dropout2d(p=0.1)),
        ]))
        self.conv6 = nn.Conv2d(512, n_classes, 1, stride=1, padding=0)
        self.aux = nn.Sequential(OrderedDict([
            ('conv4_aux', _ConvBatchNormReLU(1024, 256, 3, 1, 1, 1, use_bn=use_bn)),
            ('drop4_aux', nn.Dropout2d(p=0.1)),
        ]))
        self.conv6_1 = nn.Conv2d(256, n_classes, 1, stride=1, padding=0)

    def forward(self, x):
        x_size = x.size()
        if self.training:
            aux, h = self.fcn(x)
            aux_feat = self.aux(aux)
        else:
            h = self.fcn(x)

        h = self.ppm(h)
        h_feat = self.final(h)

        if self.training:
            if self.output_all:
                aux_out = self.conv6_1(aux_feat)
                h = self.conv6(h_feat)
                return h_feat, aux_feat, F.interpolate(h, self.input_size, mode='bilinear', align_corners=True), F.interpolate(aux_out, self.input_size, mode='bilinear', align_corners=True)

            elif self.output_features:
                aux_out = self.conv6_1(aux_feat)
                h = self.conv6(h_feat)
                return h_feat, aux_feat

            else:
                aux = self.conv6_1(aux_feat)
                h = self.conv6(h_feat)
                return F.interpolate(h, self.input_size, mode='bilinear', align_corners=True), F.interpolate(aux, self.input_size, mode='bilinear', align_corners=True)
        else:
            if self.output_features:
                return h_feat
            else:
                h = self.conv6(h_feat)
                return F.interpolate(h, self.input_size, mode='bilinear', align_corners=True)

# no upsampling


class PSPNetClustering(nn.Module):
    """Pyramid Scene Parsing Network"""
    # DEFAULTS ARE FOR CITYSCAPES

    def __init__(self, n_classes=19, n_blocks=[3, 4, 23, 3], pyramids=[6, 3, 2, 1], input_size=[713, 713], use_bn=True, output_features=False, output_all=False, use_original_base=False):

        super(PSPNetClustering, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.output_features = output_features
        self.output_for_cluster = False
        self.output_all = output_all
        if use_original_base:
            self.fcn = ResNetForPsp(Bottleneck, n_blocks)
        else:
            self.fcn = _DilatedFCN(n_blocks=n_blocks, use_bn=use_bn)
        self.ppm = _PyramidPoolModule(
            in_channels=2048, pyramids=pyramids, use_bn=use_bn)
        self.final = nn.Sequential(OrderedDict([
            ('conv5_4', _ConvBatchNormReLU(4096, 512,
                                           3, 1, 1, 1, relu=False, use_bn=use_bn)),
            ('drop5_4', nn.Dropout2d(p=0.1)),
        ]))
        self.conv6relu = nn.ReLU()
        self.conv6 = nn.Conv2d(512, n_classes, 1, stride=1, padding=0)
        self.aux = nn.Sequential(OrderedDict([
            ('conv4_aux', _ConvBatchNormReLU(
                1024, 256, 3, 1, 1, 1, relu=False, use_bn=use_bn)),
            ('drop4_aux', nn.Dropout2d(p=0.1)),
        ]))
        self.conv6_1relu = nn.ReLU()
        self.conv6_1 = nn.Conv2d(256, n_classes, 1, stride=1, padding=0)

    def forward(self, x):
        x_size = x.size()
        if self.training:
            aux, h = self.fcn(x)
            aux_feat = self.aux(aux)
        else:
            h = self.fcn(x)

        h = self.ppm(h)
        h_feat = self.final(h)

        if self.training:
            if self.output_all:
                aux = self.conv6_1relu(aux_feat)
                aux = self.conv6_1(aux)
                h = self.conv6relu(h_feat)
                h = self.conv6(h)
                return h_feat, aux_feat, h, aux
            elif self.output_features:
                return h_feat, aux_feat
            else:
                aux_feat = self.conv6_1relu(aux_feat)
                aux = self.conv6_1(aux_feat)
                h_feat = self.conv6relu(h_feat)
                h = self.conv6(h_feat)
                return h, aux
        else:
            if self.output_all:
                h = self.conv6relu(h_feat)
                h = self.conv6(h)
                return h_feat, h

            elif self.output_features:
                return h_feat
            else:
                h_feat = self.conv6relu(h_feat)
                h = self.conv6(h_feat)
                return h


if __name__ == '__main__':
    model = PSPNet(n_classes=19, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1])
    print(list(model.named_children()))
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 713, 713))
    print(model(image).size())
