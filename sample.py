import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from src.model import SWFontModel
from src.build import build_unet, build_style_encoder, build_content_encoder
from configs.swfont import get_parser
from diffusers import DPMSolverMultistepScheduler

def get_paired_path(orig_path, target_folder, is_content=False):
    """
    Replace /C/ in content image paths with the selected directory.
    Replace /F/ in style image paths with the selected directory.
    """
    if is_content:
        # Use the configured directory for content skeletons.
        return orig_path.replace('/C/', f'/{target_folder}/')
    else:
        # Use the configured directory for style width maps.
        return orig_path.replace('/F/', f'/{target_folder}/')

def run_batch_from_list(args, model, scheduler, transform_rgb, transform_gray, accelerator):
    if not os.path.exists(args.test_list_path):
        print(f"Error: List file not found: {args.test_list_path}")
        return

    with open(args.test_list_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    total_tasks = len(lines)

    if accelerator.state.num_processes > 1:
        lines = lines[accelerator.process_index:: accelerator.num_processes]

    # Display a progress bar on the main process only.
    if accelerator.is_main_process:
        print(f"Total tasks: {total_tasks}")
        print(f"Running on {accelerator.num_processes} GPUs.")
        iterator = tqdm(lines, desc="Sampling SW-Font")
    else:
        iterator = lines

    guidance_scale = getattr(args, 'guidance_scale', 7.5)
    rescale_phi = getattr(args, 'rescale_phi', 0.7)

    for line in iterator:
        try:
            parts = line.split("|")
            if len(parts) < 4: continue
            c_path, s_path, rel_dir, filename = parts[0], parts[1], parts[2], parts[3]
        except ValueError:
            continue

        final_dir = os.path.join(args.save_image_dir, rel_dir)
        os.makedirs(final_dir, exist_ok=True)
        final_save_path = os.path.join(final_dir, filename)

        if os.path.exists(final_save_path):
            continue

        s_w_path = get_paired_path(s_path, args.style_width_dir, is_content=False)
        c_skel_path = get_paired_path(c_path, args.content_skeleton_dir, is_content=True)

        missing = []
        if not os.path.isfile(c_path):
            missing.append(f"Content: {c_path}")
        if not os.path.isfile(s_path):
            missing.append(f"Style: {s_path}")
        if not os.path.isfile(s_w_path):
            missing.append(f"Style width: {s_w_path}")
        if not os.path.isfile(c_skel_path):
            missing.append(f"Content skeleton: {c_skel_path}")

        if missing:
            if accelerator.is_main_process:
                print(f"Warning: [{filename}] Missing input images: {missing}")
            continue

        try:
            c_img = transform_rgb(Image.open(c_path).convert("RGB")).unsqueeze(0).to(accelerator.device)
            s_img = transform_rgb(Image.open(s_path).convert("RGB")).unsqueeze(0).to(accelerator.device)
            s_w_img = transform_gray(Image.open(s_w_path).convert("L")).unsqueeze(0).to(accelerator.device)
            c_skel_img = transform_gray(Image.open(c_skel_path).convert("L")).unsqueeze(0).to(accelerator.device)


            uc_c = torch.ones_like(c_img)
            uc_s = torch.ones_like(s_img)
            uc_s_w = torch.ones_like(s_w_img) * -1.0
            uc_c_skel = torch.ones_like(c_skel_img) * -1.0

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        cond_input = [s_img, c_img, s_w_img, c_skel_img]
        uncond_input = [uc_s, uc_c, uc_s_w, uc_c_skel]

        # Initialize latents using the content image size.
        img_size = args.content_image_size[0] if isinstance(args.content_image_size, tuple) else args.content_image_size
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(args.seed + accelerator.process_index)
        latents = torch.randn(
            (1, 3, img_size, img_size),
            generator=generator,
            device=accelerator.device
        )

        # Reset the scheduler timesteps before each sample.
        scheduler.set_timesteps(getattr(args, 'num_inference_steps', 20))

        with torch.no_grad():
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                t_in = torch.cat([t.unsqueeze(0)] * 2).to(accelerator.device)

                s_in = torch.cat([uncond_input[0], cond_input[0]])
                c_in = torch.cat([uncond_input[1], cond_input[1]])
                s_w_in = torch.cat([uncond_input[2], cond_input[2]])
                c_skel_in = torch.cat([uncond_input[3], cond_input[3]])

                # Predict noise for conditional and unconditional inputs.
                noise_pred_chunk = model(
                    x_t=latent_model_input,
                    timesteps=t_in,
                    style_images=s_in,
                    content_images=c_in,
                    style_width_images=s_w_in,
                    content_skeleton_images=c_skel_in,
                    content_encoder_downsample_size=getattr(args, 'content_encoder_downsample_size', 3),
                )

                if isinstance(noise_pred_chunk, (list, tuple)):
                    noise_pred_chunk = noise_pred_chunk[0]

                noise_pred_uncond, noise_pred_text = noise_pred_chunk.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if rescale_phi > 0.0:
                    std_cond = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
                    std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
                    factor = std_cond / (std_cfg + 1e-5)
                    noise_pred_rescaled = noise_pred * factor
                    noise_pred = rescale_phi * noise_pred_rescaled + (1.0 - rescale_phi) * noise_pred

                latents = scheduler.step(noise_pred, t, latents).prev_sample

        final_img = (latents / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
        Image.fromarray((final_img * 255).astype(np.uint8)).save(final_save_path)

    accelerator.wait_for_everyone()

def main():
    accelerator = Accelerator()
    device = accelerator.device

    parser = get_parser()
    existing_args = [action.dest for action in parser._actions]

    if "save_image_dir" not in existing_args:
        parser.add_argument("--save_image_dir", type=str, required=True)

    if "ckpt_dir" not in existing_args:
        parser.add_argument("--ckpt_dir", type=str, required=True)

    if "test_list_path" not in existing_args:
        parser.add_argument("--test_list_path", type=str, required=True)

    if "style_width_dir" not in existing_args:
        parser.add_argument("--style_width_dir", type=str, default="style_width")

    if "content_skeleton_dir" not in existing_args:
        parser.add_argument("--content_skeleton_dir", type=str, default="content_skeleton")

    if "rescale_phi" not in existing_args:
        parser.add_argument("--rescale_phi", type=float, default=0.7)

    args = parser.parse_args()

    if isinstance(args.style_image_size, int):
        args.style_image_size = (args.style_image_size, args.style_image_size)
    if isinstance(args.content_image_size, int):
        args.content_image_size = (args.content_image_size, args.content_image_size)

    set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        print(f"Loading SW-Font model from {args.ckpt_dir}...")

    unet = build_unet(args)
    style_encoder = build_style_encoder(args)
    content_encoder = build_content_encoder(args)

    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location="cpu"), strict=True)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location="cpu"), strict=True)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu"), strict=True)

    model = SWFontModel(unet, style_encoder, content_encoder).to(device)
    model.eval()

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="scaled_linear",
        solver_order=2,
        prediction_type="epsilon",
        thresholding=False,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        lower_order_final=True,
    )

    transform_rgb = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    transform_gray = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    run_batch_from_list(args, model, scheduler, transform_rgb, transform_gray, accelerator)

if __name__ == "__main__":
    main()
