from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, register_to_config)
from diffusers.utils import BaseOutput, logging

from .embeddings import TimestepEmbedding, Timesteps


from .unet_blocks import (DownBlock2D,
                             UNetMidBlock2D,
                             UpBlock2D,
                             get_down_block,
                             get_up_block,
                             ConditionAdapter)

logger = logging.get_logger(__name__)


@dataclass
class UNetOutput(BaseOutput):
    sample: torch.FloatTensor



class SWFontUNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = None,
            up_block_types: Tuple[str] = None,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 1,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: int = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            attention_head_dim: int = 8,
            channel_attn: bool = False,
            reduction: int = 32,
            content_start_channel: int = 64,
            content_encoder_downsample_size: int = 3,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # ============================================================
        # Initialize the skeleton condition adapter.
        # ============================================================
        # Map the single-channel skeleton to 64 feature channels.
        self.cond_channels = 64
        self.condition_adapter = ConditionAdapter(in_channels=1, out_channels=self.cond_channels)

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # --------------------------------------------------------
        # Down Blocks
        # --------------------------------------------------------
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # Determine the content channels at each downsampling stage.
            if i != 0:
                content_channel = content_start_channel * (2 ** (i - 1))
            else:
                content_channel = 0

            down_block = get_down_block(
                down_block_type, num_layers=layers_per_block, in_channels=input_channel,
                out_channels=output_channel, temb_channels=time_embed_dim, add_downsample=not is_final_block,
                resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim, attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                content_channel=content_channel,
                reduction=reduction, channel_attn=channel_attn,
            )
            self.down_blocks.append(down_block)

        # --------------------------------------------------------
        # Mid Block
        # --------------------------------------------------------

        # Determine the content channels for the middle block.
        mid_content_channel = content_start_channel * (2 ** (content_encoder_downsample_size - 1))

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=0.0,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            channel_attn=channel_attn,
            content_channel=mid_content_channel,
            reduction=reduction,
        )

        self.num_upsamplers = 0

        # --------------------------------------------------------
        # Up Blocks
        # --------------------------------------------------------
        reversed_block_out_channels = list(reversed(block_out_channels))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                upblock_index=i,
                content_channel=self.cond_channels,
                reduction=reduction,
                channel_attn=channel_attn,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (DownBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            skeleton_map: torch.Tensor = None,
            content_encoder_downsample_size: int = 3,
            return_dict: bool = False,
            **kwargs
    ) -> Union[UNetOutput, Tuple]:

        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # ============================================================

        # ============================================================
        # Encode the skeleton condition when provided.
        skeleton_feat = None
        if skeleton_map is not None:
            skeleton_feat = self.condition_adapter(skeleton_map)

        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for index, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                # Pass the stage index and skeleton features to the conditioned down block.
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    index=index,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    skeleton_feat=skeleton_feat,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                temb=emb,
                index=content_encoder_downsample_size,
                encoder_hidden_states=encoder_hidden_states,
                skeleton_feat=skeleton_feat
            )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    skeleton_feat=skeleton_feat,
                    upsample_size=upsample_size
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNetOutput(sample=sample)