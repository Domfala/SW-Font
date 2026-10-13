import torch
from torch import nn
import torch.nn.functional as F

from .attention import SpatialTransformer, ChannelAttnBlock
from .resnet import Downsample2D, ResnetBlock2D, Upsample2D


# Skeleton conditioning and spatial affine modulation.

class ConditionAdapter(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

class GeometryInjectorAffine(nn.Module):
    def __init__(self, in_channels, cond_channels=64):
        super().__init__()
        self.to_scale = nn.Conv2d(cond_channels, in_channels, kernel_size=1)
        self.to_shift = nn.Conv2d(cond_channels, in_channels, kernel_size=1)

        # Zero-initialize the affine predictors for identity modulation at initialization.
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, hidden_states, cond_feat):
        # Perform interpolation in float32 for bfloat16 conditioning features.
        original_dtype = cond_feat.dtype
        if original_dtype == torch.bfloat16:
            cond_feat = cond_feat.to(torch.float32)

        # Resize the conditioning features to the spatial size of the hidden states.
        res_cond = F.interpolate(cond_feat, size=hidden_states.shape[-2:], mode='bilinear', align_corners=False)

        # Restore the original bfloat16 dtype after interpolation.
        if original_dtype == torch.bfloat16:
            res_cond = res_cond.to(original_dtype)

        gamma = self.to_scale(res_cond)
        beta = self.to_shift(res_cond)
        return hidden_states * (1.0 + gamma) + beta


def get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels,
                   add_downsample, resnet_eps, resnet_act_fn, attn_num_head_channels,
                   resnet_groups=None, cross_attention_dim=None, downsample_padding=None,
                   content_channel=64, channel_attn=False, reduction=32, **kwargs):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels,
            add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, downsample_padding=downsample_padding)
    elif down_block_type == "MCADownBlock2D":

        return SWFontDownBlock2D(
            num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels,
            add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, downsample_padding=downsample_padding, cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels, content_channel=content_channel)
    else:
        raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(up_block_type, num_layers, in_channels, out_channels, prev_output_channel,
                 temb_channels, add_upsample, resnet_eps, resnet_act_fn, attn_num_head_channels,
                 upblock_index, resnet_groups=None, cross_attention_dim=None, content_channel=64, **kwargs):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers, in_channels=in_channels, prev_output_channel=prev_output_channel, out_channels=out_channels,
            temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups)
    elif up_block_type == "StyleRSIUpBlock2D":

        return SWFontUpBlock2D(
            num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel,
            temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, cross_attention_dim=cross_attention_dim, attn_num_head_channels=attn_num_head_channels,
            content_channel=content_channel)
    else:
        raise ValueError(f"{up_block_type} does not exist.")


# Conditioned downsampling, middle, and upsampling blocks.

class SWFontDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0, num_layers=1,
                 resnet_eps=1e-6, resnet_act_fn="swish", resnet_groups=32, resnet_pre_norm=True,
                 attn_num_head_channels=1, cross_attention_dim=1280, output_scale_factor=1.0,
                 downsample_padding=1, add_downsample=True, content_channel=64, channel_attn=False, reduction=32):
        super().__init__()
        resnets = []
        attentions = []
        content_attentions = []
        geo_injectors = []

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels

            # Fuse content features through channel attention.
            content_attentions.append(
                ChannelAttnBlock(in_channels=in_ch + content_channel, out_channels=in_ch, groups=resnet_groups,
                                 non_linearity=resnet_act_fn, channel_attn=channel_attn, reduction=reduction))

            # Apply skeleton-conditioned affine modulation.
            geo_injectors.append(GeometryInjectorAffine(in_channels=in_ch, cond_channels=64))


            resnets.append(ResnetBlock2D(in_channels=in_ch, out_channels=out_channels, temb_channels=temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, dropout=dropout,
                                         non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor,
                                         pre_norm=resnet_pre_norm))

            # Apply cross-attention to style features.
            attentions.append(
                SpatialTransformer(out_channels, attn_num_head_channels, out_channels // attn_num_head_channels,
                                   depth=1, context_dim=cross_attention_dim, num_groups=resnet_groups))

        self.content_attentions = nn.ModuleList(content_attentions)
        self.geo_injectors = nn.ModuleList(geo_injectors)
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, index, temb=None, encoder_hidden_states=None, skeleton_feat=None, **kwargs):
        output_states = ()
        for resnet, attn, content_attn, geo_inj in zip(self.resnets, self.attentions, self.content_attentions,
                                                       self.geo_injectors):

            # Fuse multi-scale content features.
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)

            # Apply skeleton-guided affine modulation.
            if skeleton_feat is not None:
                hidden_states = geo_inj(hidden_states, skeleton_feat)

            hidden_states = resnet(hidden_states, temb)

            # Apply style cross-attention.
            current_style_feature = encoder_hidden_states[0]
            b, c, h, w = current_style_feature.shape
            context = current_style_feature.permute(0, 2, 3, 1).reshape(b, h * w, c)
            hidden_states = attn(hidden_states, context=context)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock2D(nn.Module):
    def __init__(self, in_channels, temb_channels, dropout=0.0, num_layers=1, resnet_eps=1e-6,
                 resnet_act_fn="swish", resnet_groups=32, resnet_pre_norm=True, attn_num_head_channels=1,
                 cross_attention_dim=1280, channel_attn=False, content_channel=64, reduction=32, **kwargs):
        super().__init__()
        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels,
                                 eps=resnet_eps, groups=resnet_groups, dropout=dropout, non_linearity=resnet_act_fn,
                                 pre_norm=resnet_pre_norm)]
        attentions = []
        content_attentions = []
        geo_injectors = []

        for _ in range(num_layers):
            content_attentions.append(
                ChannelAttnBlock(in_channels=in_channels + content_channel, out_channels=in_channels,
                                 groups=resnet_groups, non_linearity=resnet_act_fn, channel_attn=channel_attn,
                                 reduction=reduction))
            geo_injectors.append(GeometryInjectorAffine(in_channels=in_channels, cond_channels=64))
            attentions.append(
                SpatialTransformer(in_channels, attn_num_head_channels, in_channels // attn_num_head_channels, depth=1,
                                   context_dim=cross_attention_dim, num_groups=resnet_groups))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, dropout=dropout,
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.content_attentions = nn.ModuleList(content_attentions)
        self.geo_injectors = nn.ModuleList(geo_injectors)

    def forward(self, hidden_states, index, temb=None, encoder_hidden_states=None, skeleton_feat=None, **kwargs):
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet, attn, content_attn, geo_inj in zip(self.resnets[1:], self.attentions, self.content_attentions,
                                                       self.geo_injectors):
            current_content_feature = encoder_hidden_states[1][index]
            hidden_states = content_attn(hidden_states, current_content_feature)

            if skeleton_feat is not None:
                hidden_states = geo_inj(hidden_states, skeleton_feat)

            current_style_feature = encoder_hidden_states[0]
            b, c, h, w = current_style_feature.shape
            context = current_style_feature.permute(0, 2, 3, 1).reshape(b, h * w, c)
            hidden_states = attn(hidden_states, context=context)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states

class SWFontUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, temb_channels, dropout=0.0, num_layers=1,
                 resnet_eps=1e-6, resnet_act_fn="swish", resnet_groups=32, resnet_pre_norm=True,
                 attn_num_head_channels=1, cross_attention_dim=1280, add_upsample=True, **kwargs):
        super().__init__()
        resnets = []
        attentions = []
        geo_injectors = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # Upsampling combines skip connections, affine modulation, and style attention.
            geo_injectors.append(GeometryInjectorAffine(in_channels=resnet_in_channels + res_skip_channels, cond_channels=64))
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm))
            attentions.append(SpatialTransformer(out_channels, attn_num_head_channels, out_channels // attn_num_head_channels, depth=1, context_dim=cross_attention_dim, num_groups=resnet_groups))

        self.geo_injectors = nn.ModuleList(geo_injectors)
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, skeleton_feat=None, upsample_size=None, **kwargs):
        style_feat = encoder_hidden_states[0] if isinstance(encoder_hidden_states, (list, tuple)) else encoder_hidden_states

        for resnet, attn, geo_inj in zip(self.resnets, self.attentions, self.geo_injectors):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if skeleton_feat is not None:
                hidden_states = geo_inj(hidden_states, skeleton_feat)

            hidden_states = resnet(hidden_states, temb)

            if style_feat is not None:
                b, c, h, w = style_feat.shape
                context = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
                hidden_states = attn(hidden_states, context=context)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states




class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0, num_layers=1, resnet_eps=1e-6, resnet_time_scale_shift="default", resnet_act_fn="swish", resnet_groups=32, resnet_pre_norm=True, output_scale_factor=1.0, add_downsample=True, downsample_padding=1):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_ch, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")])
        else: self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers: hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states

class UpBlock2D(nn.Module):
    def __init__(self, in_channels, prev_output_channel, out_channels, temb_channels, dropout=0.0, num_layers=1, resnet_eps=1e-6, resnet_time_scale_shift="default", resnet_act_fn="swish", resnet_groups=32, resnet_pre_norm=True, output_scale_factor=1.0, add_upsample=True):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels + res_skip_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, non_linearity=resnet_act_fn, output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm))
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else: self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers: hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states