"""SW-Font training and inference arguments."""

import argparse


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (True or False).")


def parse_channels(value):
    parts = value.strip().strip("()").split(",")
    try:
        channels = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated channel sizes.") from exc
    if len(channels) != 4 or any(channel <= 0 for channel in channels):
        raise argparse.ArgumentTypeError("Expected four positive channel sizes.")
    return channels


def get_parser():
    parser = argparse.ArgumentParser(
        description="SW-Font training and inference configuration."
    )

    # General
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--experience_name", type=str, default="swfont_training")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")

    # Model
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--unet_channels", type=parse_channels, default=(64, 128, 256, 512))
    parser.add_argument("--style_image_size", type=int, default=128)
    parser.add_argument("--content_image_size", type=int, default=128)
    parser.add_argument("--content_encoder_downsample_size", type=int, default=3)
    parser.add_argument("--channel_attn", type=parse_bool, default=True)
    parser.add_argument("--content_start_channel", type=int, default=64)
    parser.add_argument("--style_start_channel", type=int, default=64)

    # Training
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--perceptual_coefficient", type=float, default=0.01)
    parser.add_argument("--max_train_steps", type=int, default=200000)
    parser.add_argument("--stop_at_step", type=int, default=None)
    parser.add_argument("--ckpt_interval", type=int, default=40000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        help="Learning-rate scheduler (includes SW-Font partial_linear).")
    parser.add_argument("--lr_warmup_steps", type=int, default=10000)
    parser.add_argument("--drop_prob", type=float, default=0.1)

    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])

    # Sampling
    parser.add_argument("--algorithm_type", type=str, default="dpmsolver++")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--local_rank", type=int, default=-1)

    # WES-Net
    parser.add_argument("--wes_net_path", type=str,
                        default="src/width_estimator/pretrained/wes_net.pth")
    parser.add_argument("--wes_loss_coeff", type=float, default=1.0)

    # Checkpoint and dataset lists
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    parser.add_argument("--val_list_path", type=str, default="./val_list.txt")
    parser.add_argument("--test_list_path", type=str, default="./data/test_list.txt")

    return parser
