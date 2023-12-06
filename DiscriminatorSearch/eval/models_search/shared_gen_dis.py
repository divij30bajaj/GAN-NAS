# -*- coding: utf-8 -*-
# Original code from AutoGAN: Xinyu Gong (xy_gong@tamu.edu)
# E2GAN modified it for our purpose
# + add progressive state representation

import torch.nn as nn
import torch
from models_search.building_blocks_search import Cell
from models_search.dis_blocks_search import DisCell
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, num_skip_in=0)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, num_skip_in=1)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, num_skip_in=2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )
    def set_stage(self, cur_stage):
        self.cur_stage = cur_stage
    
    def set_arch(self, arch_id, cur_stage):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]
        self.cur_stage = cur_stage
        arch_stage1 = arch_id[:4]
        self.cell1.set_arch(conv_id=arch_stage1[0], norm_id=arch_stage1[1], up_id=arch_stage1[2],
                            short_cut_id=arch_stage1[3], skip_ins=[])
        if cur_stage >= 1:
            arch_stage2 = arch_id[4:9]
            self.cell2.set_arch(conv_id=arch_stage2[0], norm_id=arch_stage2[1], up_id=arch_stage2[2],
                                short_cut_id=arch_stage2[3], skip_ins=arch_stage2[4])

        if cur_stage == 2:
            arch_stage3 = arch_id[9:]
            self.cell3.set_arch(conv_id=arch_stage3[0], norm_id=arch_stage3[1], up_id=arch_stage3[2],
                                short_cut_id=arch_stage3[3], skip_ins=arch_stage3[4])
    
    def forward(self, z, smooth=1., eval=False):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)    
        if self.cur_stage == -1:
            return self.to_rgb(h), F.interpolate(h, size=(1, 1), mode="bilinear").detach()
        h1_skip_out, h1 = self.cell1(h)
        if self.cur_stage == 0:
            if not eval:
                return self.to_rgb(h1)
            else:
                # [z_num, Channel, dsampled_h, dsampled_w] for the state, here we simply adopt 1 to reduce the size of the state.
                return self.to_rgb(h1), F.interpolate(h1, size=(1, 1), mode="bilinear").detach() 
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out,))
        if self.cur_stage == 1:
            _, _, ht, wt = h2.size()            
            if not eval:
                # smooth is disabled in the final submission (= 1.). We leave it here for you to play around. 
                return smooth * self.to_rgb(h2) + (1. - smooth) * self.to_rgb(F.interpolate(h1, size=(ht, wt), mode="bilinear"))
            else:
                return self.to_rgb(h2), F.interpolate(h2, size=(1, 1), mode="bilinear").detach()
        
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        if self.cur_stage == 2:
            _, _, ht, wt = h3.size()          
            if not eval:
                return smooth * self.to_rgb(h3) + (1. - smooth) * self.to_rgb(F.interpolate(h2, size=(ht, wt), mode="bilinear"))
            else:
                return self.to_rgb(h3), F.interpolate(h3, size=(1, 1), mode="bilinear").detach()
            
            

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.args = args
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = DisCell(args, 3, self.ch, num_skip_in=0)
        self.block2 = DisCell(args, self.ch, self.ch, num_skip_in=1)
        self.block3 = DisCell(args, self.ch, self.ch, num_skip_in=2)
        self.l4 = nn.Linear(self.ch, 1, bias=False)
        self.l1 = nn.Linear(self.ch, 1, bias=False)
        self.l2 = nn.Linear(3, self.ch, bias=False)
        if args.d_spectral_norm:
            self.l4 = nn.utils.spectral_norm(self.l4)
            self.l1 = nn.utils.spectral_norm(self.l1)

    def set_stage(self, cur_stage):
        self.cur_stage = cur_stage

    def set_arch(self, arch_id, cur_stage):
        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]
        self.cur_stage = cur_stage
        arch_stage1 = arch_id[:4]
        self.block1.set_arch(conv_id=arch_stage1[0], norm_id=arch_stage1[1], down_id=arch_stage1[2],
                            short_cut_id=arch_stage1[3], skip_ins=[])
        if cur_stage >= 1:
            arch_stage2 = arch_id[4:9]
            self.block2.set_arch(conv_id=arch_stage2[0], norm_id=arch_stage2[1], down_id=arch_stage2[2],
                                short_cut_id=arch_stage2[3], skip_ins=arch_stage2[4])

        if cur_stage == 2:
            arch_stage3 = arch_id[9:]
            self.block3.set_arch(conv_id=arch_stage3[0], norm_id=arch_stage3[1], down_id=arch_stage3[2],
                                short_cut_id=arch_stage3[3], skip_ins=arch_stage3[4])

    def forward(self, x):
        h = x
        if self.cur_stage == -1:
            h = h.permute(0, 2, 3, 1)
            h = self.activation(h)
            h = self.l2(h)
            h = h.permute(0, 3, 1, 2)
            state = F.interpolate(h, size=(1, 1), mode="bilinear").detach()
            h = h.sum(2).sum(2)
            output = self.l1(h)
            return output, state

        h1_skip, h = self.block1(h)
        if self.cur_stage == 0:
            h = self.activation(h)
            state = F.interpolate(h, size=(1, 1), mode="bilinear").detach()
            h = h.sum(2).sum(2)
            output = self.l4(h)
            return output, state

        h2_skip, h = self.block2(h, (h1_skip,))
        if self.cur_stage == 1:
            h = self.activation(h)
            state = F.interpolate(h, size=(1, 1), mode="bilinear").detach()
            h = h.sum(2).sum(2)
            output = self.l4(h)
            return output, state
        h3_skip, h = self.block3(h, (h1_skip, h2_skip))
        if self.cur_stage == 2:
            h = self.activation(h)
            state = F.interpolate(h, size=(1, 1), mode="bilinear").detach()
            h = h.sum(2).sum(2)
            output = self.l4(h)
            return output, state