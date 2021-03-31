import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Conv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride=1, padding=0, bias=False, bn_before=None, bn_after=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn_before = [bn_before]
        self.bn_after = [bn_after]

    def forward(self, x):
        return self.conv(x)


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bn_before=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn_before = bn_before

    def forward(self, x):
        return self.linear(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, images_dims=None, bn_before=None):
        super(BasicBlock, self).__init__()
        self.prunablebn1 = nn.BatchNorm2d(planes)
        self.conv1 = Conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                          bn_after=self.prunablebn1, bn_before=bn_before)

        self.bn2 = nn.BatchNorm2d(planes)
        self.prunablebn3 = nn.BatchNorm2d(planes)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                          bn_before=self.prunablebn1, bn_after=self.prunablebn3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            conv3 = Conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                         bn_before=bn_before, bn_after=self.prunablebn3)
            self.shortcut = nn.Sequential(
                conv3,
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.shortcut_conv = stride != 1 or in_planes != self.expansion * planes
        self.inplanes = in_planes
        self.planes = planes
        self.stride = stride
        self.last_bn = [self.prunablebn3]
        self.pruned_inplanes = in_planes
        self.images_dims = images_dims

    def forward(self, x):
        out = F.relu(self.prunablebn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.prunablebn3(out)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, images_dims=None, bn_before=None):
        super(Bottleneck, self).__init__()

        self.prunablebn1 = nn.BatchNorm2d(planes)
        self.conv1 = Conv(in_planes, planes, kernel_size=1, bias=False, bn_after=self.prunablebn1, bn_before=bn_before)

        self.prunablebn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                          bn_before=self.prunablebn1, bn_after=self.prunablebn2)

        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.prunablebn4 = nn.BatchNorm2d(self.expansion * planes)
        self.conv3 = Conv(planes, self.expansion * planes, kernel_size=1, bias=False,
                          bn_before=self.prunablebn2, bn_after=self.prunablebn4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            conv = Conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                        bn_before=bn_before, bn_after=self.prunablebn4)
            self.shortcut = nn.Sequential(
                conv,
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.shortcut_conv = stride != 1 or in_planes != self.expansion * planes
        self.inplanes = in_planes
        self.planes = planes
        self.stride = stride
        self.last_bn = [self.prunablebn4]
        self.previous_bn = None
        self.images_dims = images_dims

    def forward(self, x):
        out = F.relu(self.prunablebn1(self.conv1(x)))
        out = F.relu(self.prunablebn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.prunablebn4(out)
        out = F.relu(out)
        return out


def get_block_params_count(block, th=0., distributed=False):
    if distributed:
        block = block.module
    if isinstance(block, BasicBlock):
        bn1 = torch.sum(block.prunablebn1.weight.abs() > th)
        bn3 = torch.sum(block.prunablebn3.weight.abs() > th)
        fin = torch.sum(block.previous_bn.abs() > th)
        count = (fin * bn1 * 3 * 3)
        count += (bn1 * 2)
        count += (bn1 * bn3 * 3 * 3)
        count += (2 * bn3 * 2)
        if block.shortcut_conv:
            count += (fin * bn3) + 2 * bn3
        return int(count)
    elif isinstance(block, Bottleneck):
        fin = torch.sum(block.previous_bn.abs() > th)
        f1 = torch.sum(block.prunablebn1.weight.abs() > th)
        f2 = torch.sum(block.prunablebn2.weight.abs() > th)
        fout = torch.sum(block.prunablebn4.weight.abs() > th)

        count = (fin * f1) + (2 * f1)
        count += (f1 * f2 * 3 * 3) + (2 * f2)
        count += (f2 * fout) + (2 * fout)
        if block.shortcut_conv:
            count += (fin * fout) + 2 * fout
        count += 2 * fout
        return int(count)
    else:
        raise TypeError


def get_block_flops_count(block, th=0., distributed=False):
    if distributed:
        block = block.module
    if isinstance(block, BasicBlock):
        bn1 = torch.sum(block.prunablebn1.weight.abs() > th)
        bn3 = torch.sum(block.prunablebn3.weight.abs() > th)
        fin = torch.sum(block.previous_bn.abs() > th)
        dims = block.images_dims[0] * block.images_dims[1]
        count = (fin * bn1 * 3 * 3) * dims
        count += (bn1 * 2) * dims
        count += (bn1 * bn3 * 3 * 3) * dims
        count += (2 * bn3 * 2) * dims
        if block.shortcut_conv:
            count += (fin * bn3) * dims
            count += 2 * bn3 * dims
        return int(count)
    elif isinstance(block, Bottleneck):
        fin = torch.sum(block.previous_bn.abs() > th)
        f1 = torch.sum(block.prunablebn1.weight.abs() > th)
        f2 = torch.sum(block.prunablebn2.weight.abs() > th)
        fout = torch.sum(block.prunablebn4.weight.abs() > th)
        dims = block.images_dims
        in_dims = dims[0] * dims[1]
        out_dims = dims[0] * dims[1]
        if block.stride != 1:
            out_dims = out_dims / 4
        count = (fin * f1) * in_dims
        count += (2 * f1) * in_dims
        count += (f1 * f2 * 3 * 3) * out_dims
        count += (2 * f2) * out_dims
        count += (f2 * fout) * out_dims
        count += (2 * fout) * out_dims
        if block.shortcut_conv:
            count += (fin * fout) * out_dims
            count += 2 * fout * out_dims
        return int(count)
    else:
        raise TypeError


class ResNetModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNetModel, self).__init__()
        self.blocks_list = list()

        self.in_planes = in_planes
        self.is_imagenet = num_classes == 1000

        self.prunablebn1 = nn.BatchNorm2d(in_planes)
        if self.is_imagenet:
            self.conv1 = Conv(3, in_planes, kernel_size=7, stride=2, padding=3,
                              bias=False, bn_after=self.prunablebn1)
        else:  # CIFAR
            self.conv1 = Conv(3, in_planes, kernel_size=3,
                              stride=1, padding=1, bias=False, bn_after=self.prunablebn1)

        size = 224 if self.is_imagenet else 32
        self.post_conv1_dim = (size, size)
        self.layer1, last_bn = self._make_layer(block, in_planes, num_blocks[0], stride=1, size=(size, size),
                                                last_bn=self.prunablebn1)
        self.layer2, last_bn = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2,
                                                size=(size / 2, size / 2), last_bn=last_bn)
        self.layer3, layer3bn = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2,
                                                 size=(size / 4, size / 4), last_bn=last_bn)
        self.layer4 = None
        if len(num_blocks) == 4:
            self.layer4, layer4bn = self._make_layer(block, in_planes * 8, num_blocks[3],
                                                     stride=2, size=(size / 8, size / 8), last_bn=layer3bn)
            self.linear = Linear(in_planes * 8 * block.expansion, num_classes, bn_before=layer4bn)
        else:
            self.linear = Linear(in_planes * 4 * block.expansion, num_classes, bn_before=layer3bn)

    def _make_layer(self, block, planes, num_blocks, stride, size, last_bn=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        last_bn_ = last_bn
        for stride in strides:
            b = block(self.in_planes, planes, stride, size, last_bn_)
            self.blocks_list.append(b)
            layers.append(b)
            self.in_planes = planes * block.expansion
            last_bn_ = b.last_bn[0]
        return nn.Sequential(*layers), layers[-1].last_bn[0]

    def forward(self, x):
        out = F.relu(self.prunablebn1(self.conv1(x)))
        if self.is_imagenet:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4:
            out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.model = ResNetModel(block, num_blocks, num_classes, in_planes)
        self.module = self.model
        self.num_block = num_blocks
        self.num_classes = num_classes
        self.last_bn = None
        self.distributed = False

    def forward(self, x):
        return self.model(x)

    def distribute(self):
        self.model = torch.nn.DataParallel(self.model)
        self.module = self.model.module
        self.distributed = True

    def update_last_bn(self):
        last_bn = self.module.prunablebn1.weight.data
        for b in self.module.blocks_list:
            b.previous_bn = last_bn
            last_bn = b.last_bn[0].weight.data
        self.last_bn = self.module.blocks_list[-1].last_bn[0].weight.data

    def compute_params_count(self, pruning_type='structured', threshold=0):
        if pruning_type == 'unstructured':
            return int(torch.sum(torch.tensor([torch.sum(i.abs() > threshold) for i in self.module.parameters()])))
        elif 'structured' in pruning_type:
            self.update_last_bn()
            count = (3 * torch.sum(self.module.prunablebn1.weight.abs() > threshold)
                     * self.module.conv1.conv.kernel_size[0] * self.module.conv1.conv.kernel_size[1])
            count += torch.sum(self.module.prunablebn1.weight.abs() > threshold) * 2
            for b in self.module.blocks_list:
                count += get_block_params_count(b, threshold)
            count += torch.sum(self.last_bn.abs() > threshold) * self.num_classes + self.num_classes
            return int(count)
        else:
            return int(np.sum([len(i.flatten()) for i in self.module.parameters()]))

    def compute_flops_count(self, threshold=0):
        self.update_last_bn()
        count = 0
        for b in self.module.blocks_list:
            count += get_block_flops_count(b, threshold)
        count += (3 * torch.sum(self.module.prunablebn1.weight.abs() > threshold)
                  * self.module.conv1.conv.kernel_size[0] * self.module.conv1.conv.kernel_size[1]
                  * self.module.post_conv1_dim[0] * self.module.post_conv1_dim[1])
        count += (2 * torch.sum(self.module.prunablebn1.weight.abs() > threshold)
                  * self.module.post_conv1_dim[0] * self.module.post_conv1_dim[1])
        count += torch.sum(self.last_bn.abs() > threshold) * self.num_classes + self.num_classes
        return int(count)


def resnet18(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_planes=in_planes)


def resnet34(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_planes=in_planes)


def resnet50(num_classes=10, in_planes=64):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_planes=in_planes)


def resnet101(num_classes=10, in_planes=64):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_planes=in_planes)


def resnet152(num_classes=10, in_planes=64):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_planes=in_planes)


def resnet20(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, in_planes=in_planes)


def resnet32(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, in_planes=in_planes)


def resnet44(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, in_planes=in_planes)


def resnet56(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, in_planes=in_planes)


def resnet110(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, in_planes=in_planes)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


def resnet_model(model, num_classes=10, in_planes=64):
    if model == 'resnet18':
        return resnet18(num_classes, in_planes)
    elif model == 'resnet34':
        return resnet34(num_classes, in_planes)
    elif model == 'resnet50':
        return resnet50(num_classes, in_planes)
    elif model == 'resnet101':
        return resnet101(num_classes, in_planes)
    elif model == 'resnet152':
        return resnet152(num_classes, in_planes)
    elif model == 'resnet20':
        return resnet20(num_classes, in_planes)
    elif model == 'resnet32':
        return resnet32(num_classes, in_planes)
    elif model == 'resnet44':
        return resnet44(num_classes, in_planes)
    elif model == 'resnet56':
        return resnet56(num_classes, in_planes)
    elif model == 'resnet110':
        return resnet110(num_classes, in_planes)
    else:
        raise ValueError
