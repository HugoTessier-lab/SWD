import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, images_dims=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.prunablebn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prunablebn3 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.shortcut_conv = stride != 1 or in_planes != self.expansion * planes
        self.inplanes = in_planes
        self.planes = planes
        self.stride = stride
        self.last_bn = self.prunablebn3
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

    def __init__(self, in_planes, planes, stride=1, images_dims=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.prunablebn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.prunablebn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.prunablebn4 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.shortcut_conv = stride != 1 or in_planes != self.expansion * planes
        self.inplanes = in_planes
        self.planes = planes
        self.stride = stride
        self.last_bn = self.prunablebn4
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

        if self.is_imagenet:
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:  # CIFAR
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.prunablebn1 = nn.BatchNorm2d(in_planes)
        size = 224 if self.is_imagenet else 32
        self.post_conv1_dim = (size, size)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, size=(size, size))
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, size=(size / 2, size / 2))
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, size=(size / 4, size / 4))
        self.layer4 = None
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, size=(size / 8, size / 8))
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        else:
            self.layer4 = None
            self.linear = nn.Linear(in_planes * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            b = block(self.in_planes, planes, stride, size)
            self.blocks_list.append(b)
            layers.append(b)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

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
            last_bn = b.last_bn.weight.data
        self.last_bn = self.module.blocks_list[-1].last_bn.weight.data

    def compute_params_count(self, pruning_type='structured', threshold=0):
        if pruning_type == 'unstructured':
            return int(torch.sum(torch.tensor([torch.sum(i.abs() > threshold) for i in self.module.parameters()])))
        elif 'structured' in pruning_type:
            self.update_last_bn()
            count = (3 * torch.sum(self.module.prunablebn1.weight.abs() > threshold)
                     * self.module.conv1.kernel_size[0] * self.module.conv1.kernel_size[1])
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
                  * self.module.conv1.kernel_size[0] * self.module.conv1.kernel_size[1]
                  * self.module.post_conv1_dim[0] * self.module.post_conv1_dim[1])
        count += (2 * torch.sum(self.module.prunablebn1.weight.abs() > threshold)
                  * self.module.post_conv1_dim[0] * self.module.post_conv1_dim[1])
        count += torch.sum(self.last_bn.abs() > threshold) * self.num_classes + self.num_classes
        return int(count)


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_planes=64)


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_planes=64)


def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_planes=64)


def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_planes=64)


def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_planes=64)


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, in_planes=64)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, in_planes=64)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, in_planes=64)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, in_planes=64)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, in_planes=64)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


def resnet_model(model, num_classes=10):
    if model == 'resnet18':
        return resnet18(num_classes)
    elif model == 'resnet34':
        return resnet34(num_classes)
    elif model == 'resnet50':
        return resnet50(num_classes)
    elif model == 'resnet101':
        return resnet101(num_classes)
    elif model == 'resnet152':
        return resnet152(num_classes)
    elif model == 'resnet20':
        return resnet20(num_classes)
    elif model == 'resnet32':
        return resnet32(num_classes)
    elif model == 'resnet44':
        return resnet44(num_classes)
    elif model == 'resnet56':
        return resnet56(num_classes)
    elif model == 'resnet110':
        return resnet110(num_classes)
    else:
        raise ValueError
