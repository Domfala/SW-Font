import os
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ==================================================================
# Import SW-Font model components.
# ==================================================================
from src.modules.content_encoder import ContentEncoder  # Content encoder
from src.modules.style_encoder import SWFontStyleEncoder  # Four-channel style encoder with StyleGate
from src.modules.unet import SWFontUNet  # U-Net with skeleton-guided affine modulation

# Import the WES-Net architecture.
# WES-Net is implemented under src/width_estimator.

def build_unet(args):
    # Construct the SW-Font U-Net.
    # Preserve the architecture used by existing checkpoints.
    unet = SWFontUNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        # Keep the existing block names for compatibility.
        down_block_types=('DownBlock2D',
                          'MCADownBlock2D',
                          'MCADownBlock2D',
                          'DownBlock2D'),
        up_block_types=('UpBlock2D',
                        'StyleRSIUpBlock2D',
                        'StyleRSIUpBlock2D',
                        'UpBlock2D'),
        block_out_channels=args.unet_channels,
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        reduction=32,
        content_start_channel = args.content_start_channel,
        content_encoder_downsample_size = args.content_encoder_downsample_size
    )
    print("✅ Built SWFontUNet (with Skeleton Affine Injection)!")
    return unet


def build_style_encoder(args):
    style_image_encoder = SWFontStyleEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size[0],
        input_nc=4  # Four input channels: RGB (3) and stroke width (1)
    )
    print("✅ Built SWFontStyleEncoder (Width-Aware with StyleGate)!")
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size[0]
    )
    print("✅ Built Content Encoder!")
    return content_image_encoder


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="scaled_linear",
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True
    )
    return ddpm_scheduler

def build_wes_net(args):
    """Build WES-Net and load its pretrained weights."""
    from src.width_estimator.Estimator_model.MS_CBAM import EnhancedUNet as MSCBAMUNet

    wes_model_path = getattr(args, "wes_net_path", None)
    if not wes_model_path:
        raise ValueError("--wes_net_path must be provided.")

    wes_model_path = os.path.abspath(
        os.path.expanduser(wes_model_path)
    )

    if not os.path.isfile(wes_model_path):
        raise FileNotFoundError(
            f"WES-Net checkpoint not found: {wes_model_path}"
        )

    wes_net = MSCBAMUNet(in_channels=3, out_channels=1)

    print(f"Loading WES-Net weights from {wes_model_path}...")
    state_dict = torch.load(wes_model_path, map_location="cpu")

    # Support checkpoints saved with DistributedDataParallel.
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    wes_net.load_state_dict(new_state_dict, strict=True)
    print("WES-Net weights loaded successfully.")

    return wes_net
