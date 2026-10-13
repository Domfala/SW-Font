import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin,
                                           register_to_config)


# Spectral normalization utilities and basic layers.

def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x

def power_iteration(W, u_, update=True, eps=1e-12):
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', act='relu', use_sn=False):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]

class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None, ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc: x = self.conv_sc(x)
            if self.downsample:   x = self.downsample(x)
        else:
            if self.downsample:   x = self.downsample(x)
            if self.learnable_sc: x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)


# Squeeze-and-excitation-style channel gating.

class StyleGate(nn.Module):
    """
    Reweights channels of the fused style and width feature map.
    Global average pooling supplies channel descriptors for the gate.
    The gate does not explicitly label texture or width channels.
    """
    def __init__(self, channel, reduction=16):
        super(StyleGate, self).__init__()
        # Pool spatial dimensions into a channel descriptor.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Apply learned channel weights.
        return x * y.expand_as(x)



# Encoder configurations for supported input resolutions.

def style_encoder_textedit_addskip_arch(ch=64, out_channel_multiplier=1, input_nc=4):
    arch = {}
    arch[96] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'resolution': [48, 24, 12, 6, 3]}

    arch[128] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                 'resolution': [64, 32, 16, 8, 4]}

    arch[256] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16]],
                 'resolution': [128, 64, 32, 16, 8, 4]}
    return arch



# Width-aware style encoder.

class SWFontStyleEncoder(ModelMixin, ConfigMixin):
    """
    Encode a reference glyph using image and stroke-width inputs.
    StyleGate reweights the resulting fused feature channels.
    """
    @register_to_config
    def __init__(
            self,
            G_ch=64,
            G_wide=True,
            resolution=128,
            G_kernel_size=3,
            G_attn='64_32_16_8',
            n_classes=1000,
            num_G_SVs=1,
            num_G_SV_itrs=1,
            G_activation=nn.ReLU(inplace=False),
            SN_eps=1e-12,
            output_dim=1,
            G_fp16=False,
            G_init='N02',
            G_param='SN',
            nf_mlp=512,
            nEmbedding=256,
            input_nc=4,
            output_nc=3
    ):
        super(SWFontStyleEncoder, self).__init__()

        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16

        if self.resolution in [96, 128]:
            self.save_featrues = [0, 1, 2, 3, 4]
        elif self.resolution == 256:
            self.save_featrues = [0, 1, 2, 3, 4, 5]

        self.out_channel_nultipiler = 1
        self.arch = style_encoder_textedit_addskip_arch(
            self.ch,
            self.out_channel_nultipiler,
            input_nc
        )[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(
                SNConv2d, kernel_size=3, padding=1,
                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps
            )
            self.which_linear = functools.partial(
                SNLinear, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps
            )

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[DBlock(
                in_channels=self.arch['in_channels'][index],
                out_channels=self.arch['out_channels'][index],
                which_conv=self.which_conv,
                wide=self.G_wide,
                activation=self.activation,
                preactivation=(index > 0),
                downsample=nn.AvgPool2d(2)
            )]]

        self.blocks = nn.ModuleList([
            nn.ModuleList(block) for block in self.blocks
        ])

        # Reweight final backbone features before projection.
        final_channels = self.arch['out_channels'][-1]
        self.style_gate = StyleGate(final_channels, reduction=16)

        self.last_layer = nn.Sequential(
            nn.InstanceNorm2d(final_channels),
            self.activation,
            nn.Conv2d(final_channels, final_channels, kernel_size=1, stride=1)
        )

        self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for initialized parameters: %d' % self.param_count)

    def forward(self, style_rgb, style_width=None):
        """
        Input tensors:
        style_rgb: [B, 3, H, W]
        style_width: [B, 1, H, W], or None
        """
        # Concatenate the style image and the width-map channel.
        if style_width is not None:
            h = torch.cat([style_rgb, style_width], dim=1)
        else:
            # Fill the width channel with zeros when it is not provided.
            dummy_width = torch.zeros_like(style_rgb[:, :1, :, :])
            h = torch.cat([style_rgb, dummy_width], dim=1)

        residual_features = []
        residual_features.append(h)

        # Extract multi-scale features.
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
            if index in self.save_featrues[:-1]:
                residual_features.append(h)

        # Apply channel-wise gating.
        h = self.style_gate(h)

        # Project the gated features into the final style representation.
        style_emd = self.last_layer(h)

        return style_emd, None, residual_features
