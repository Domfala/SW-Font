import os
import math
import time
import logging
import cv2
from tqdm.auto import tqdm
from PIL import Image
from diffusers import DDIMScheduler
import numpy as np
import torch.nn as nn

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.swfont import get_parser

# Import the SW-Font model.
from src.model import SWFontModel
# Import model construction functions.
from src.build import build_ddpm_scheduler
from src.build import (build_unet,
                          build_style_encoder,
                          build_content_encoder,
                          build_wes_net)

from utils import (save_args_to_yaml,
                   x0_from_epsilon,
                   reNormalize_img,
                   normalize_mean_std,
                   save_single_image)
from diffusers.training_utils import EMAModel
from src.criterion import ContentPerceptualLoss

logger = get_logger(__name__)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filters for horizontal and vertical gradients.
        filter_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        filter_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('filter_x', filter_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))
        self.register_buffer('filter_y', filter_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        # Compute horizontal and vertical image gradients.
        pred_dx = F.conv2d(pred, self.filter_x, padding=1, groups=3)
        pred_dy = F.conv2d(pred, self.filter_y, padding=1, groups=3)
        target_dx = F.conv2d(target, self.filter_x, padding=1, groups=3)
        target_dy = F.conv2d(target, self.filter_y, padding=1, groups=3)

        # Apply smooth L1 loss to gradient magnitudes.
        pred_mag = torch.sqrt(pred_dx ** 2 + pred_dy ** 2 + 1e-6)
        target_mag = torch.sqrt(target_dx ** 2 + target_dy ** 2 + 1e-6)
        return F.smooth_l1_loss(pred_mag, target_mag)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    return args


def make_image_grid(imgs, rows, cols):
    if not imgs or len(imgs) == 0: return None
    w, h = imgs[0].size
    grid = Image.new('RGB', (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(imgs):
        if i >= rows * cols: break
        grid.paste(img, (i % cols * w, i // cols * h))
    return grid


def main():
    args = get_args()
    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.basicConfig(
        filename=f"{args.output_dir}/swfont_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Build the SW-Font model components.
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)

    # ===================================================================
    # Memory optimization options.
    # ===================================================================

    # Use attention slicing to reduce memory usage.
    try:
        unet.set_attention_slice(1)
        print("✅ Enabled Attention Slicing (Saves VRAM)")
    except Exception as e:
        print(f"⚠️ Attention Slicing failed: {e}")

    # Enable memory-efficient attention when available.
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("✅ Enabled xformers memory efficient attention!")
    except Exception as e:
        print("⚠️ xformers not available or failed to enable.")

    # Enable gradient checkpointing when supported.
    try:
        unet.enable_gradient_checkpointing()
        print("✅ Enabled Gradient Checkpointing!")
    except Exception as e:
        print("⚠️ Gradient Checkpointing failed.")
    # ===================================================================

    if args.resume_ckpt_path:
        if not os.path.isdir(args.resume_ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {args.resume_ckpt_path}"
            )

        for filename in (
            "unet.pth",
            "style_encoder.pth",
            "content_encoder.pth",
        ):
            checkpoint_file = os.path.join(
                args.resume_ckpt_path, filename
            )
            if not os.path.isfile(checkpoint_file):
                raise FileNotFoundError(
                    f"Missing checkpoint weight: {checkpoint_file}"
                )
        print(f"Loading model weights from {args.resume_ckpt_path}...")
        unet.load_state_dict(torch.load(f"{args.resume_ckpt_path}/unet.pth", map_location='cpu'), strict=False)
        style_encoder.load_state_dict(torch.load(f"{args.resume_ckpt_path}/style_encoder.pth", map_location='cpu'),
                                      strict=False)
        if 'content_encoder' in locals() and content_encoder is not None:
            content_encoder.load_state_dict(
                torch.load(f"{args.resume_ckpt_path}/content_encoder.pth", map_location='cpu'))
        print("Model weights loaded successfully.")

    model = SWFontModel(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)

    ema_model = EMAModel(
        model.parameters(),
        decay=0.9999,
        update_after_step=100,
        model_cls=SWFontModel,
        model_config=model.config
    )
    ema_model.to(accelerator.device)

    # Build and freeze the pretrained WES-Net estimator.
    wes_net = build_wes_net(args)
    if wes_net:
        wes_net.eval()
        for param in wes_net.parameters():
            param.requires_grad = False
        wes_net = wes_net.to(accelerator.device)

    perceptual_loss = ContentPerceptualLoss()
    criterion_ssim = SSIM(window_size=11).to(accelerator.device)

    edge_criterion = EdgeLoss().to(accelerator.device)

    content_transforms = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_transforms = transforms.Compose([
        transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    target_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    width_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_font_dataset = FontDataset(
        args=args, phase='train',
        transforms=[content_transforms, style_transforms, target_transforms, width_transforms]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size,
        collate_fn=CollateFN(), num_workers=4, pin_memory=True, persistent_workers=True
    )

    val_font_dataset = FontDataset(
        args=args, phase='val',
        transforms=[content_transforms, style_transforms, target_transforms, width_transforms]
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_font_dataset, shuffle=False, batch_size=args.train_batch_size,
        collate_fn=CollateFN(), num_workers=4, pin_memory=True
    )

    if args.scale_lr:
        args.learning_rate = (
                    args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
    )


    if args.lr_scheduler == "partial_linear":
        warmup_steps = args.lr_warmup_steps * args.gradient_accumulation_steps
        total_steps = args.max_train_steps * args.gradient_accumulation_steps
        end_factor = 0.7906976744186046

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 1.0 - (1.0 - end_factor) * min(progress, 1.0)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )


    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    global_step = 0
    if args.resume_ckpt_path:
        try:
            dir_name = os.path.basename(os.path.normpath(args.resume_ckpt_path))
            if "global_step_" in dir_name:
                global_step = int(dir_name.split("_")[-1])
        except Exception:
            global_step = 0

    stop_at_step = (
        args.stop_at_step
        if args.stop_at_step is not None
        else args.max_train_steps
    )

    if not 0 < stop_at_step <= args.max_train_steps:
        raise ValueError("Invalid stop_at_step.")

    if global_step >= stop_at_step:
        raise ValueError("Checkpoint has already reached the stopping step.")
  
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, initial=global_step)
    progress_bar.set_description("Steps")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    fixed_sample_batch = next(iter(train_dataloader))
    fixed_sample_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in fixed_sample_batch.items()}
    num_samples = min(9, len(fixed_sample_batch["content_image"]))

    fixed_sample = {
        "content_image": fixed_sample_batch["content_image"][:num_samples],
        "style_image": fixed_sample_batch["style_image"][:num_samples],
        "style_width_image": fixed_sample_batch["style_width_image"][:num_samples],
        "content_skeleton_image": fixed_sample_batch["content_skeleton_image"][:num_samples],
        "width_image_gt": fixed_sample_batch["width_image"][:num_samples]  # Cache target width maps for validation.
    }

    # Limit WES-Net supervision to early diffusion timesteps.
    structure_guidance_start_step = 300

    unwrapped_model_ref = accelerator.unwrap_model(model)
    inference_wrapper = SWFontModel(
        unet=unwrapped_model_ref.unet,
        style_encoder=unwrapped_model_ref.style_encoder,
        content_encoder=unwrapped_model_ref.content_encoder,
    ).to(accelerator.device)
    inference_wrapper.eval()

    for epoch in range(num_train_epochs):
        if global_step >= stop_at_step:
            break
        for step, samples in enumerate(train_dataloader):
            model.train()

            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]

            # Load width and skeleton conditioning tensors.
            style_width_images = samples["style_width_image"]
            content_skeleton_images = samples["content_skeleton_image"]
            target_width_gt = samples["width_image"]  # GT Width for WES-Net Loss

            with accelerator.accumulate(model):
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,),
                                          device=target_images.device).long()
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                if args.drop_prob > 0:
                    context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                    for i, mask_value in enumerate(context_mask):
                        if mask_value == 1:
                            content_images[i, :, :, :] = 1
                            style_images[i, :, :, :] = 1
                            # Zero-valued width and skeleton maps are -1 after normalization.
                            style_width_images[i, :, :, :] = -1
                            content_skeleton_images[i, :, :, :] = -1

                # Predict noise using style, content, width, and skeleton inputs.
                noise_pred = model(
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                    style_images=style_images,
                    content_images=content_images,
                    style_width_images=style_width_images,
                    content_skeleton_images=content_skeleton_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size
                )

                # Base denoising loss.
                loss = F.smooth_l1_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Recover the estimated clean sample for auxiliary supervision.
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps
                )


                # Perceptual loss.
                if args.perceptual_coefficient > 0:
                    pred_original_sample = reNormalize_img(pred_original_sample_norm)
                    norm_pred_ori = normalize_mean_std(pred_original_sample)
                    norm_target_ori = normalize_mean_std(nonorm_target_images)
                    loss_percep = perceptual_loss.calculate_loss(
                        generated_images=norm_pred_ori, target_images=norm_target_ori, device=target_images.device
                    )
                    loss += args.perceptual_coefficient * loss_percep

                # WES-Net stroke-width supervision.
                if wes_net is not None:
                    # Select samples according to their diffusion timesteps.
                    valid_mask = (timesteps < structure_guidance_start_step)

                    if valid_mask.any():
                        # Select predicted images before WES-Net inference.
                        # The selected batch contains sum(valid_mask) images.
                        wes_input = pred_original_sample_norm[valid_mask].float().clamp(-1, 1)

                        if wes_input.shape[-1] != 128:
                            wes_input = F.interpolate(wes_input, size=(128, 128), mode='bilinear', align_corners=False)

                        # Invert image polarity for WES-Net input.
                        wes_input = -1.0 * wes_input

                        W_pred = torch.tanh(wes_net(wes_input))

                        # Select matching target width maps.
                        W_GT = target_width_gt.to(accelerator.device)[valid_mask]

                        # B. MSE Loss
                        loss_mse = F.mse_loss(W_pred.float(), W_GT.float())

                        # SSIM loss on the selected width maps.
                        W_pred_norm = (W_pred.float() + 1.0) / 2.0
                        W_GT_norm = (W_GT.float() + 1.0) / 2.0

                        ssim_val = criterion_ssim(W_pred_norm.clamp(0, 1), W_GT_norm.clamp(0, 1))
                        loss_ssim = 1.0 - ssim_val

                        loss_swm = 1.0 * loss_mse + 0.5 * loss_ssim
                        loss += args.wes_loss_coeff * loss_swm
                        accelerator.log({"L_SWM": loss_swm.item()}, step=global_step)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    ema_model.step(accelerator.unwrap_model(model).parameters())

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                avg_loss = accelerator.gather(loss).mean().detach().item()

                accelerator.log({"train_loss": avg_loss}, step=global_step)

                if global_step % args.ckpt_interval == 0:
                    if accelerator.is_main_process:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)

                        ema_model.store(unwrapped_model.parameters())
                        ema_model.copy_to(unwrapped_model.parameters())

                        torch.save(unwrapped_model.unet.state_dict(), f"{save_dir}/unet.pth")
                        torch.save(unwrapped_model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                        torch.save(unwrapped_model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                        torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pth")
                        torch.save(lr_scheduler.state_dict(), f"{save_dir}/scheduler.pth")
                        torch.save(unwrapped_model, f"{save_dir}/total_model.pth")
                        inference_wrapper.eval()

                        logging.info(f"Running validation on {len(val_font_dataset)} samples...")
                        total_ssim, total_mse, val_count = 0.0, 0.0, 0

                        for val_batch in tqdm(val_dataloader, desc="Validating", leave=False):
                            v_c = val_batch["content_image"].to(accelerator.device)
                            v_s = val_batch["style_image"].to(accelerator.device)
                            v_s_w = val_batch["style_width_image"].to(accelerator.device)
                            v_c_skel = val_batch["content_skeleton_image"].to(accelerator.device)
                            v_t = val_batch["target_image"].to(accelerator.device)

                            with torch.no_grad():
                                latents = torch.randn((len(v_c), 3, args.resolution, args.resolution),
                                                      device=accelerator.device)
                                val_scheduler = DDIMScheduler(
                                    num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                                    beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
                                    steps_offset=1
                                )
                                val_scheduler.set_timesteps(20)

                                for t in val_scheduler.timesteps:
                                    ts = torch.full((len(v_c),), t, device=accelerator.device, dtype=torch.long)

                                    # Prepare validation conditioning tensors.
                                    cond_input = [v_s, v_c, v_s_w, v_c_skel]
                                    noise_pred = inference_wrapper(
                                        x_t=latents,
                                        timesteps=ts,
                                        style_images=v_s,
                                        content_images=v_c,
                                        style_width_images=v_s_w,
                                        content_skeleton_images=v_c_skel,
                                        content_encoder_downsample_size=args.content_encoder_downsample_size
                                    )
                                    latents = val_scheduler.step(noise_pred, t, latents).prev_sample

                            pred_norm = (latents.clamp(-1, 1) + 1) / 2
                            target_norm = (v_t + 1) / 2
                            total_ssim += criterion_ssim(pred_norm, target_norm).item()
                            total_mse += F.mse_loss(pred_norm, target_norm).item()
                            val_count += 1

                        avg_ssim = total_ssim / val_count
                        avg_mse = total_mse / val_count
                        accelerator.log({"Val/SSIM": avg_ssim, "Val/MSE": avg_mse}, step=global_step)
                        logging.info(f"Step {global_step} Validation Results: SSIM={avg_ssim:.4f}, MSE={avg_mse:.4f}")
                        torch.cuda.empty_cache()

                        # ============================================================
                        # Visualize synthesized glyphs and stroke-width maps.
                        # ============================================================
                        logging.info(f"Generating comparison grid at step {global_step}...")
                        f_c = fixed_sample["content_image"].to(accelerator.device)
                        f_s = fixed_sample["style_image"].to(accelerator.device)
                        f_s_w = fixed_sample["style_width_image"].to(accelerator.device)
                        f_c_skel = fixed_sample["content_skeleton_image"].to(accelerator.device)

                        # Read cached target width maps.
                        f_w_gt = fixed_sample["width_image_gt"].to(accelerator.device)  # Target width maps for comparison.

                        scheduler = DDIMScheduler(
                            num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                            beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1
                        )
                        scheduler.set_timesteps(20)
                        latents = torch.randn((len(f_c), 3, args.resolution, args.resolution),
                                              device=accelerator.device)

                        with torch.no_grad():
                            for t in scheduler.timesteps:
                                ts = torch.full((len(f_c),), t, device=accelerator.device, dtype=torch.long)
                                # Call the model using named arguments.
                                noise_pred = inference_wrapper(
                                    x_t=latents,
                                    timesteps=ts,
                                    style_images=f_s,
                                    content_images=f_c,
                                    style_width_images=f_s_w,
                                    content_skeleton_images=f_c_skel,
                                    content_encoder_downsample_size=args.content_encoder_downsample_size
                                )
                                latents = scheduler.step(noise_pred, t, latents).prev_sample

                            # Convert generated images to [0, 1].
                            f_x0 = (latents.clamp(-1, 1) / 2 + 0.5)

                            # Predict stroke-width maps.
                            if wes_net is not None:
                                wes_val_input = latents.detach().clamp(-1, 1)
                                if wes_val_input.shape[-1] != 128:
                                    wes_val_input = F.interpolate(wes_val_input, size=(128, 128), mode='bilinear')

                                # Invert image polarity for WES-Net inference.
                                wes_val_input = -1.0 * wes_val_input

                                # Bound predicted stroke widths using tanh.
                                f_w_pred_raw = torch.tanh(wes_net(wes_val_input))
                                f_w_pred = (f_w_pred_raw / 2 + 0.5).repeat(1, 3, 1, 1)  # Replicate the width map across three channels.
                            else:
                                f_w_pred = torch.zeros_like(f_x0)

                            # Replicate target width maps across three channels.
                            f_w_gt_vis = (f_w_gt / 2 + 0.5).repeat(1, 3, 1, 1)

                        # Concatenate glyphs and width maps for visualization.
                        # Concatenate along the image width dimension.
                        comparison_tensors = torch.cat([f_x0, f_w_pred, f_w_gt_vis], dim=3)  # Concatenate along the image width dimension.
                        save_image(comparison_tensors, f"{save_dir}/val_comparison_{global_step}.png", nrow=1,
                                   normalize=False)

                        logging.info(f"Comparison grid saved: val_comparison_{global_step}.png")

                        torch.cuda.empty_cache()

                        # Restore training parameters after EMA validation.
                        ema_model.restore(unwrapped_model.parameters())
                        model.train()

                    accelerator.wait_for_everyone()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(f"Step {global_step} Loss: {logs['step_loss']}")
            progress_bar.set_postfix(**logs)

            if global_step >= stop_at_step:
                break
    accelerator.end_training()


if __name__ == "__main__":
    main()
