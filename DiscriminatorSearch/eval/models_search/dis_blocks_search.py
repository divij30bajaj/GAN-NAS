# -*- coding: utf-8 -*-
# @Date    : 2019-08-02
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from torch import nn
import torch.nn.functional as F


CONV_TYPE = {0: 'optimized', 1: 'normal'}
NORM_TYPE = {0: None, 1: 'bn', 2: 'in'}
DOWN_TYPE = {0: 'pre', 1: 'post', 2: None}
SHORT_CUT_TYPE = {0: False, 1: True}
SKIP_TYPE = {0: False, 1: True}


def decimal2binary(n):
    return bin(n).replace("0b", "")


class OptimizedDisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad).cuda()
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad).cuda()
        self.bn = nn.BatchNorm2d(out_channels).cuda()
        self.inn = nn.InstanceNorm2d(out_channels).cuda()

        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1).cuda()
            self.c2 = nn.utils.spectral_norm(self.c2).cuda()

    def add_norm(self, x):
        x = x.cuda()
        if self.norm_type:
            if self.norm_type == 'bn':
                h = self.bn(x)
            elif self.norm_type == 'in':
                h = self.inn(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            h = x
        return h.cuda()

    def set_arch(self, down_id, norm_id):
        self.down_type = DOWN_TYPE[down_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        x = x.cuda()
        h = x
        print("Norm: ", x.shape)
        h = self.c1(h)
        h = self.add_norm(h)
        h = self.activation(h)
        h = self.c2(h)
        h = self.add_norm(h)
        if self.down_type is not None:
            h = _downsample(h)
        return h


class DisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(DisBlock, self).__init__()
        self.activation = activation
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad).cuda()
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad).cuda()
        self.bn1 = nn.BatchNorm2d(in_channels).cuda()
        self.inn1 = nn.InstanceNorm2d(in_channels).cuda()

        self.bn2 = nn.BatchNorm2d(hidden_channels).cuda()
        self.inn2 = nn.InstanceNorm2d(hidden_channels).cuda()

        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1).cuda()
            self.c2 = nn.utils.spectral_norm(self.c2).cuda()

    def add_norm(self, x, oneortwo):
        x = x.cuda()
        if self.norm_type:
            if self.norm_type == 'bn':
                if oneortwo == 1:
                    h = self.bn1(x)
                else:
                    h = self.bn2(x)
            elif self.norm_type == 'in':
                if oneortwo == 1:
                    h = self.inn1(x)
                else:
                    h = self.inn2(x)
            else:
                raise NotImplementedError(self.norm_type)
        else:
            h = x
        return h.cuda()

    def set_arch(self, down_id, norm_id):
        self.down_type = DOWN_TYPE[down_id]
        self.norm_type = NORM_TYPE[norm_id]

    def forward(self, x):
        x = x.cuda()
        print("Norm: ", x.shape)
        h = self.add_norm(x, 1)
        h = self.activation(h)
        h = self.c1(h)
        h = self.add_norm(h, 2)
        h = self.activation(h)
        h = self.c2(h)
        if self.down_type is not None:
            h = _downsample(h)
        return h


class DisCell(nn.Module):
    def __init__(self, args, in_channels, out_channels, num_skip_in, ksize=3):
        super(DisCell, self).__init__()

        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.optimized_dis = OptimizedDisBlock(args, in_channels, out_channels, ksize=ksize)
        self.dis = DisBlock(args, in_channels, out_channels, ksize=ksize)
        self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)

        if self.args.d_spectral_norm:
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

        self.num_skip_in = num_skip_in
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def set_arch(self, conv_id, norm_id, down_id, short_cut_id, skip_ins):
        self.optimized_dis.set_arch(down_id, norm_id)
        self.dis.set_arch(down_id, norm_id)

        if self.num_skip_in:
            self.skip_ins = [0 for _ in range(self.num_skip_in)]
            for skip_idx, skip_in in enumerate(decimal2binary(skip_ins)[::-1]):
                self.skip_ins[-(skip_idx + 1)] = int(skip_in)

        self.conv_type = CONV_TYPE[conv_id]
        self.downsample = DOWN_TYPE[down_id]
        self.short_cut = SHORT_CUT_TYPE[short_cut_id]

    def forward(self, x, skip_ft=None):
        residual = x

        if self.conv_type == 'optimized':
            h = self.optimized_dis(residual)
        elif self.conv_type == 'normal':
            h = self.dis(residual)
        else:
            raise NotImplementedError(self.norm_type)

        _, _, ht, wt = h.size()
        h_skip_out = h
        # second conv
        if self.num_skip_in:
            assert len(self.skip_in_ops) == len(self.skip_ins)
            for skip_flag, ft, skip_in_op in zip(self.skip_ins, skip_ft, self.skip_in_ops):
                if skip_flag:
                    h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode="bilinear"))

        # shortcut
        if self.short_cut:
            if self.downsample == 'pre':
                h += self.c_sc(_downsample(x))
            elif self.downsample == 'post':
                h += _downsample(self.c_sc(x))
            else:
                h += self.c_sc(x)

        return h_skip_out, h


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)
