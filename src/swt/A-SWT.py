import os
import argparse
from typing import TypeVar, NamedTuple, List, Optional
import cv2
import numpy as np
from tqdm import tqdm

# Type aliases
Image = np.ndarray
GradientImage = np.ndarray
Position = NamedTuple('Position', [('x', int), ('y', int)])
Ray = List[Position]
ImageOrValue = TypeVar('ImageOrValue', float, Image)
Gradients = NamedTuple('Gradients', [('x', GradientImage), ('y', GradientImage)])


# ==========================================
# Image preprocessing and stroke width transform
# ==========================================

def gamma(x: ImageOrValue, coeff: float = 2.2) -> ImageOrValue:
    return x ** (1. / coeff)


def gleam(im: Image, gamma_coeff: float = 2.2) -> Image:
    im = gamma(im, gamma_coeff)
    im = np.mean(im, axis=2)
    return np.expand_dims(im, axis=2)


def preprocess_image(im_bgr: Image) -> Image:
    im = im_bgr.astype(np.float32) / 255.
    return gleam(im)


def get_edges(im: Image, lo: float = 50, hi: float = 150, window: int = 3) -> Image:
    im_uint8 = (im * 255.).clip(0, 255).astype(np.uint8)
    edges = cv2.Canny(im_uint8, lo, hi, apertureSize=window)
    return edges.astype(np.float32) / 255.


def get_gradients(im: Image) -> Gradients:
    grad_x = cv2.Scharr(im, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(im, cv2.CV_64F, 0, 1)
    return Gradients(x=grad_x, y=grad_y)


def apply_swt(im: Image, edges: Image, gradients: Gradients, dark_on_bright: bool = True,
              step_size: float = 0.5, angle_tolerance: float = -0.5) -> Image:
    swt = np.squeeze(np.ones_like(im)) * np.inf
    norms = np.sqrt(gradients.x ** 2 + gradients.y ** 2)
    norms[norms == 0] = 1
    inv_norms = 1. / norms
    directions = Gradients(x=gradients.x * inv_norms, y=gradients.y * inv_norms)
    rays = []
    height, width = im.shape[0:2]

    for y in range(height):
        for x in range(width):
            if edges[y, x] < .5:
                continue
            ray = swt_process_pixel(Position(x=x, y=y), edges, directions, out=swt,
                                    dark_on_bright=dark_on_bright, step_size=step_size,
                                    angle_tolerance_cosine=angle_tolerance)
            if ray:
                rays.append(ray)

    for ray in rays:
        median = np.median([swt[p.y, p.x] for p in ray])
        for p in ray:
            swt[p.y, p.x] = min(median, swt[p.y, p.x])

    swt[swt == np.inf] = 0
    return swt


def swt_process_pixel(
        pos: Position, edges: Image, directions: Gradients, out: Image,
        dark_on_bright: bool = True, step_size: float = 0.5, max_steps: int = 200, angle_tolerance_cosine: float = -0.5
) -> Optional[Ray]:
    height, width = edges.shape[:2]
    gradient_direction = -1 if dark_on_bright else 1
    dir_x = directions.x[pos.y, pos.x]
    dir_y = directions.y[pos.y, pos.x]

    if np.isnan(dir_x) or np.isnan(dir_y): return None

    x, y = pos.x + 0.5, pos.y + 0.5
    ray = [pos]
    visited = {(pos.x, pos.y)}

    for step in range(1, max_steps + 1):
        x += gradient_direction * dir_x * step_size
        y += gradient_direction * dir_y * step_size
        cur_x, cur_y = int(round(x)), int(round(y))

        if not (0 <= cur_x < width and 0 <= cur_y < height):
            return None

        cur_pos = Position(cur_x, cur_y)
        if (cur_x, cur_y) in visited:
            continue

        visited.add((cur_x, cur_y))
        ray.append(cur_pos)

        if edges[cur_y, cur_x] < 0.5:
            continue

        cur_dir_x = directions.x[cur_y, cur_x]
        cur_dir_y = directions.y[cur_y, cur_x]
        if (dir_x * cur_dir_x + dir_y * cur_dir_y) >= angle_tolerance_cosine:
            return None

        dx, dy = cur_x - pos.x, cur_y - pos.y
        stroke_width = np.sqrt(dx * dx + dy * dy)

        for p in ray:
            out[p.y, p.x] = min(stroke_width, out[p.y, p.x])
        return ray
    return None


def template_fill_swt(swt: np.ndarray, template_img: np.ndarray, neighborhood: int = 5,
                      method: str = 'inpaint') -> np.ndarray:
    if swt.shape != template_img.shape[:2]:
        template_img = cv2.resize(template_img, (swt.shape[1], swt.shape[0]), interpolation=cv2.INTER_NEAREST)

    template_bin = (template_img < 128).astype(np.uint8)
    missing_mask = (template_bin == 1) & (swt == 0)

    if method == 'inpaint':
        mask = missing_mask.astype(np.uint8) * 255
        swt_norm = np.uint8(np.clip(swt / (swt.max() + 1e-5), 0, 1) * 255)
        inpainted = cv2.inpaint(swt_norm, mask, inpaintRadius=int(neighborhood), flags=cv2.INPAINT_TELEA)
        restored = (inpainted.astype(np.float32) / 255.) * swt.max()
        return np.where(swt > 0, swt, restored)
    else:
        swt_filled = swt.copy()
        h, w = swt.shape
        offset = neighborhood // 2
        for y, x in zip(*np.where(missing_mask)):
            region = swt[max(y - offset, 0):min(y + offset + 1, h), max(x - offset, 0):min(x + offset + 1, w)]
            region_nonzero = region[region > 0]
            if len(region_nonzero) > 0:
                swt_filled[y, x] = np.mean(region_nonzero)
        return swt_filled


# ==========================================
# Denoising and normalization
# ==========================================

def ultimate_denoise(image: np.ndarray, spot_kernel_size: int = 7) -> np.ndarray:
    'Morphological opening followed by median filtering.'
    if spot_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spot_kernel_size, spot_kernel_size))
        processed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        processed = image
    return cv2.medianBlur(processed, 3)


def normalize_swt(swt_final: np.ndarray, norm_method: str, hard_limit: float) -> np.ndarray:
    'Normalize stroke widths using logarithmic or linear scaling.'
    if norm_method == 'log':
        swt_clipped = np.clip(swt_final, 0, hard_limit)
        swt_norm = np.log1p(swt_clipped) / np.log1p(hard_limit)
    else:  # linear
        swt_norm = np.clip(swt_final / hard_limit, 0, 1)

    return (swt_norm * 255).astype(np.uint8)


# ==========================================
# Batch processing
# ==========================================

def process_single_image(input_path, output_path, args):
    try:
        orig_img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if orig_img_bgr is None:
            return False

        h_orig, w_orig = orig_img_bgr.shape[:2]
        scale = args.upscale_factor

        # Upsample to retain thin strokes
        if scale > 1.0:
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            img_bgr = cv2.resize(orig_img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_bgr = orig_img_bgr

        # Preprocessing and stroke width transform
        im = preprocess_image(img_bgr)
        edges = get_edges(im, lo=args.canny_lo, hi=args.canny_hi)
        gradients = get_gradients(im)
        swt = apply_swt(im, edges, gradients, not args.bright_on_dark,
                        step_size=args.step_size, angle_tolerance=args.angle_tol)

        # Fill missing stroke widths
        template_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        swt_filled = template_fill_swt(swt, template_gray,
                                       neighborhood=max(3, int(args.neighborhood_size * scale)),
                                       method=args.fill_method)

        # Resize to the original image dimensions
        if scale > 1.0:
            swt_final = cv2.resize(swt_filled, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
            swt_final = swt_final / scale  # Restore stroke widths in original pixel units
        else:
            swt_final = swt_filled

        # Normalize stroke widths
        swt_vis = normalize_swt(swt_final, args.norm_method, args.max_width_limit)

        # Optional morphological spot removal
        if args.clean_spot_size > 0:
            swt_vis = ultimate_denoise(swt_vis, spot_kernel_size=args.clean_spot_size)

        # Save the resulting width map
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, swt_vis)
        return True

    except Exception as e:
        print(f"\n[Error] Failed on {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='A-SWT stroke-width map generation.')

    # Input and output arguments
    parser.add_argument('--input', type=str, required=True, help='Input image directory.')
    parser.add_argument('--output', type=str, required=True, help='Output image directory.')
    parser.add_argument('--flat', action='store_true', help='Process only files in the input directory; do not recurse.')

    # SWT algorithm parameters
    parser.add_argument('--upscale_factor', type=float, default=2.0,
                        help='Image enlargement factor; 1.0 disables upscaling.')
    parser.add_argument('--canny_lo', type=int, default=50, help='Lower Canny edge threshold.')
    parser.add_argument('--canny_hi', type=int, default=150, help='Upper Canny edge threshold.')
    parser.add_argument('--step_size', type=float, default=0.5, help='Ray tracing step size in pixels.')
    parser.add_argument('--angle_tol', type=float, default=-0.5, help='Cosine threshold for opposing edge gradients.')

    # Filling and normalization parameters
    parser.add_argument('--fill_method', choices=['average', 'inpaint'], default='inpaint', help='Method for filling missing stroke widths.')
    parser.add_argument('--neighborhood_size', type=int, default=7, help='Base neighborhood size for width filling.')
    parser.add_argument('--norm_method', choices=['log', 'linear'], default='log',
                        help='Width normalization method: log or linear.')
    parser.add_argument('--max_width_limit', type=float, default=25.0,
                        help='Maximum physical width (pixels) for normalization.')

    # Denoising parameters
    parser.add_argument('--clean_spot_size', type=int, default=7,
                        help='Morphological opening kernel size (odd); 0 disables opening.')

    parser.add_argument('--bright_on_dark', action='store_true', help='Detect light strokes on a dark background.')

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect image paths
    tasks = []
    if not args.flat:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    tasks.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                tasks.append(os.path.join(input_dir, file))

    print("Starting A-SWT processing...")
    print(
        f"Upscale factor: {args.upscale_factor}x | normalization: {args.norm_method} | width limit: {args.max_width_limit} | spot removal kernel: {args.clean_spot_size}")
    print(f"Found {len(tasks)} images to process\n")

    success_count = 0
    for input_path in tqdm(tasks, desc="Processing"):
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)

        if process_single_image(input_path, output_path, args):
            success_count += 1

    print(f"\nProcessing completed: {success_count}/{len(tasks)} successful")


if __name__ == '__main__':
    main()
