from torch import nn
import torch
import numpy as np
import math

__all__ = ['MobileNetV2', 'mobilenet_v2']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.prunablebn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)
        self.previous_bn = None
        self.in_planes = in_planes
        self.out_dim = None
        self.last_bn = self.prunablebn

    def forward(self, x):
        out = self.relu(self.prunablebn(self.conv(x)))
        if not self.out_dim:
            self.out_dim = out.shape
        return out

    def get_block_params_count(self, th=0.):
        if self.previous_bn is None:
            fin = self.in_planes
        else:
            fin = torch.sum(self.previous_bn.abs() > th)
        fout = torch.sum(self.prunablebn.weight.abs() > th)
        count = (math.ceil(float(fin * fout) / float(self.conv.groups))
                 * self.conv.kernel_size[0] * self.conv.kernel_size[1])
        count += fout * 2
        return count

    def get_block_flops_count(self, th=0.):
        if self.previous_bn is None:
            fin = self.in_planes
        else:
            fin = torch.sum(self.previous_bn.abs() > th)
        fout = torch.sum(self.prunablebn.weight.abs() > th)
        dims = self.out_dim[2] * self.out_dim[3]
        count = (math.ceil(float(fin * fout) / float(self.conv.groups))
                 * self.conv.kernel_size[0] * self.conv.kernel_size[1]) * dims
        count += fout * 2 * dims
        return count


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.layers = layers
        self.prunablebn = nn.BatchNorm2d(oup) if self.use_res_connect else layers[-1]
        self.conv = nn.Sequential(*layers)
        self.previous_bn = None
        self.out_dim = None
        self.last_bn = self.prunablebn

    def forward(self, x):
        if self.use_res_connect:
            out = self.prunablebn(x + self.conv(x))
        else:
            out = self.conv(x)
        self.out_dim = out.shape
        return out

    def get_block_params_count(self, th=0.):
        self.layers[0].previous_bn = self.previous_bn
        self.layers[1].previous_bn = self.layers[0].prunablebn.weight.data
        fout = torch.sum(self.prunablebn.weight.abs() > th)
        count = self.layers[0].get_block_params_count(th)
        if self.expand_ratio != 1:
            count += self.layers[1].get_block_params_count(th)
        count += torch.sum(self.layers[-3].last_bn.weight.abs() > th) * fout
        count += fout * 2 * (2 if self.use_res_connect else 1)
        return count

    def get_block_flops_count(self, th=0.):
        self.layers[0].previous_bn = self.previous_bn
        self.layers[1].previous_bn = self.layers[0].prunablebn.weight.data
        dims = self.out_dim[2] * self.out_dim[3]
        fout = torch.sum(self.prunablebn.weight.abs() > th)
        count = self.layers[0].get_block_flops_count(th)
        if self.expand_ratio != 1:
            count += self.layers[1].get_block_flops_count(th)
        count += torch.sum(self.layers[-3].last_bn.weight.abs() > th) * fout * dims
        count += fout * 2 * (2 if self.use_res_connect else 1) * dims
        return count


class MobileNetV2Model(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        super(MobileNetV2Model, self).__init__()

        self.block_list = list()

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                b = block(input_channel, output_channel, stride, expand_ratio=t)
                self.block_list.append(b)
                features.append(b)
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        super(MobileNetV2, self).__init__()

        self.model = MobileNetV2Model(num_classes, width_mult, inverted_residual_setting, round_nearest, block)
        self.module = self.model
        self.last_bn = None
        self.num_classes = num_classes

    def forward(self, x):
        out = self.model(x)
        return out

    def distribute(self):
        self.model = torch.nn.DataParallel(self.model)
        self.module = self.model.module

    def update_last_bn(self):
        last_bn = None
        for f in self.module.features:
            if last_bn:
                f.previous_bn = last_bn.weight.data
            last_bn = f.last_bn
        self.last_bn = last_bn.weight.data

    def compute_params_count(self, pruning_type='structured', threshold=0):
        if pruning_type == 'unstructured':
            return int(torch.sum(torch.tensor([torch.sum(i > threshold) for i in self.module.parameters()])))
        elif 'structured' in pruning_type:
            self.update_last_bn()
            count = 0
            for f in self.module.features:
                count += f.get_block_params_count(threshold)
            count += torch.sum(self.last_bn.abs() > threshold) * self.num_classes + self.num_classes
            return count
        else:
            return int(np.sum([len(i.flatten()) for i in self.module.parameters()]))

    def compute_flops_count(self, threshold=0):
        self.update_last_bn()
        count = 0
        for f in self.module.features:
            count += f.get_block_flops_count(threshold)
        count += torch.sum(self.last_bn.abs() > threshold) * self.num_classes + self.num_classes
        return count


def mobilenet_v2(**kwargs):
    model = MobileNetV2(**kwargs)
    return model
