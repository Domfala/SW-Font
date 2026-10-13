import os
import random
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_nonorm_transform(resolution):
    nonorm_transform = transforms.Compose(
        [transforms.Resize((resolution, resolution),
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor()])
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation
    Supports both:
    1. Folder Scanning (Training/Validation default)
    2. File List Loading (Validation/Test specific via txt list)
    """

    def __init__(self, args, phase, transforms=None):
        super().__init__()
        self.args = args
        self.root = args.data_root
        self.phase = phase

        # Count warnings for missing auxiliary inputs.
        self.missing_warning_count = 0
        self.max_warnings = 10

        # Initialize dataset indices.
        self.target_images = []
        self.data_list = []
        self.style_to_images = {}


        self.get_path()

        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)


    def get_path(self):
        """
        Load paired validation entries when configured.
        If a validation list is available, use its specified image paths.
        Otherwise, scan the dataset folders.
        """

        # --------------------------------------------------------
        # Prefer explicit validation pairs when a list is available.
        # --------------------------------------------------------
        use_list_mode = False

        if self.phase == 'val' and hasattr(self.args, 'val_list_path') and self.args.val_list_path:
            list_path = self.args.val_list_path
            if os.path.exists(list_path):
                print(f"📄 [Dataset] Loading validation list from: {list_path}")
                with open(list_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line: continue

                    # Support pipe-separated and whitespace-separated sample entries.
                    if '|' in line:
                        parts = line.split('|')
                    else:
                        parts = line.split()


                    if len(parts) >= 3:
                        # Four columns: content path, style path, label directory, target filename.
                        # Three columns: content path, style path, target path.



                        if len(parts) == 4:
                            # Reconstruct the target path from its directory and filename.











                            c_path = parts[0]
                            s_path = parts[1]
                            label_dir = parts[2]
                            t_name = parts[3]

                            # Resolve the target image path.
                            t_path = os.path.join(self.root, self.phase, label_dir, t_name)

                            # Store fixed pairs for deterministic sample selection.
                            self.data_list.append({
                                'content': c_path,
                                'style': s_path,
                                'target': t_path
                            })
                            self.target_images.append(t_path)

                        else:
                            # Three-column format: content, style, target.
                            self.data_list.append({
                                'content': parts[0],
                                'style': parts[1],
                                'target': parts[2]
                            })
                            self.target_images.append(parts[2])

                use_list_mode = True
                print(f"✅ Loaded {len(self.data_list)} samples from list.")

            else:
                print(f"⚠️ Warning: val_list_path provided but not found: {list_path}. Fallback to scan mode.")

        # --------------------------------------------------------
        # Fall back to directory scanning.
        # --------------------------------------------------------
        if not use_list_mode:

            target_image_dir = f"{self.root}/{self.phase}/TargetImage"
            if not os.path.exists(target_image_dir):
                raise FileNotFoundError(f"❌ Error: Dataset path not found: {target_image_dir}")

            # Select a reference-style directory when available.
            style_ref_dir_F = f"{self.root}/{self.phase}/F"
            style_ref_dir_Std = f"{self.root}/{self.phase}/StyleImage"

            if os.path.exists(style_ref_dir_F):
                print(f"ℹ️ [Validation] Found Reference Style Folder: {style_ref_dir_F}")
                ref_source_dir = style_ref_dir_F
            elif os.path.exists(style_ref_dir_Std):
                ref_source_dir = style_ref_dir_Std
            else:
                ref_source_dir = target_image_dir

            # Collect target glyph paths.
            for style in os.listdir(target_image_dir):
                style_dir = os.path.join(target_image_dir, style)
                if not os.path.isdir(style_dir): continue
                for img in os.listdir(style_dir):
                    if not img.endswith('.png'): continue
                    img_path = os.path.join(style_dir, img)
                    self.target_images.append(img_path)

            # Group reference glyphs by style.
            if os.path.exists(ref_source_dir):
                for style in os.listdir(ref_source_dir):
                    style_dir = os.path.join(ref_source_dir, style)
                    if not os.path.isdir(style_dir): continue
                    images = []
                    for img in os.listdir(style_dir):
                        if img.endswith('.png'):
                            images.append(os.path.join(style_dir, img))
                    if len(images) > 0:
                        self.style_to_images[style] = images

            print(f"✅ [Scan Mode] Loaded {len(self.target_images)} target images.")

    def __getitem__(self, index):
        # Resolve paths for this sample.

        if self.data_list and index < len(self.data_list):
            item = self.data_list[index]
            target_image_path = item['target']
            fixed_content_path = item['content']
            fixed_style_path = item['style']


            target_image_name = os.path.basename(target_image_path)

            # Infer the style identifier and character from the filename.

            try:
                name_no_ext = os.path.splitext(target_image_name)[0]
                parts = name_no_ext.split('+')
                if len(parts) >= 2:
                    style = parts[0]
                    content = parts[-1]
                else:
                    style = "unknown"
                    content = "unknown"
            except:
                style = "unknown"
                content = "unknown"

        else:
            # Infer auxiliary paths for samples discovered by directory scanning.
            target_image_path = self.target_images[index]
            target_image_name = os.path.basename(target_image_path)
            fixed_content_path = None
            fixed_style_path = None


            try:
                name_no_ext = os.path.splitext(target_image_name)[0]
                parts = name_no_ext.split('+')
                if len(parts) >= 2:
                    style = parts[0]
                    content = parts[-1]
                else:
                    raise ValueError(f"Filename format error: {target_image_name}")
            except Exception as e:
                print(f"⚠️ Error parsing filename: {target_image_path}, {e}")
                raise e

        # --------------------------------------------------------
        # Load the content glyph and associated width prior.
        # --------------------------------------------------------
        # Use the explicit content path when it exists.
        if fixed_content_path and os.path.exists(fixed_content_path):
            content_image_path = fixed_content_path
        else:

            content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.png"
            if not os.path.exists(content_image_path):
                content_image_path = f"{self.root}/ContentImage/{content}.png"

        if os.path.exists(content_image_path):
            content_image = Image.open(content_image_path).convert('RGB')
        else:

            raise FileNotFoundError(f"❌ Missing Content Image: {content_image_path}")

        # Locate the width map paired with the content glyph.


        if "ContentImage" in content_image_path:
            content_width_path = content_image_path.replace("ContentImage", "ContentWidth")
        else:
            # Fallback
            content_width_path = f"{self.root}/{self.phase}/ContentWidth/{content}.png"

        if os.path.exists(content_width_path):
            content_width_image = Image.open(content_width_path).convert("L")
        else:
            if self.missing_warning_count < self.max_warnings:
                print(f"⚠️ [Warning] Missing Content Width: {content_width_path}")
                self.missing_warning_count += 1
            content_width_image = Image.new('L', content_image.size, 0)

        # --------------------------------------------------------
        # Load the style reference image.
        # --------------------------------------------------------
        if fixed_style_path and os.path.exists(fixed_style_path):
            # Use the fixed reference image in list mode.
            style_image_path = fixed_style_path
        else:
            # Randomly choose a reference glyph of the same style in scan mode.
            images_related_style = self.style_to_images.get(style, [])
            if len(images_related_style) > 0:
                temp_list = images_related_style.copy()
                if target_image_path in temp_list:
                    temp_list.remove(target_image_path)

                if len(temp_list) > 0:
                    style_image_path = random.choice(temp_list)
                else:
                    style_image_path = target_image_path
            else:
                style_image_path = target_image_path

        style_image = Image.open(style_image_path).convert("RGB")

        # --------------------------------------------------------
        # Load the target glyph and its precomputed width supervision map.
        # --------------------------------------------------------
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        # Resolve the precomputed target width map.
        # Map TargetImage paths to PredWidthImage paths.
        width_image_path = target_image_path.replace("TargetImage", "PredWidthImage")

        if os.path.exists(width_image_path):
            target_width_image = Image.open(width_image_path).convert("L")
        else:

            width_image_path_alt = f"{self.root}/{self.phase}/PredWidthImage/{style}/{target_image_name}"
            if os.path.exists(width_image_path_alt):
                target_width_image = Image.open(width_image_path_alt).convert("L")
            else:
                if self.missing_warning_count < self.max_warnings:
                    print(f"⚠️ [Warning] Missing Target Width: {width_image_path}")
                    self.missing_warning_count += 1
                target_width_image = Image.new('L', target_image.size, 0)

        # ========================================================
        # Load the width map paired with the style reference.
        # ========================================================
        style_width_path = style_image_path.replace("TargetImage", "PredWidthImage").replace("StyleImage", "PredWidthImage").replace("/F/",
                                                                                            "/PredWidthImage/")
        if os.path.exists(style_width_path):
            style_width_image = Image.open(style_width_path).convert("L")
        else:
            # Try an alternative width-map directory layout.
            style_width_path_alt = f"{self.root}/{self.phase}/PredWidthImage/{os.path.basename(os.path.dirname(style_image_path))}/{os.path.basename(style_image_path)}"
            if os.path.exists(style_width_path_alt):
                style_width_image = Image.open(style_width_path_alt).convert("L")
            else:
                if self.missing_warning_count < self.max_warnings:
                    print(f"[Warning] Missing Style Width: {style_width_path}")
                    self.missing_warning_count += 1
                style_width_image = Image.new('L', style_image.size, 0)

        # ========================================================
        # Load the content skeleton map.
        # ========================================================
        content_skeleton_path = content_image_path.replace("ContentImage", "ContentSkeleton")
        if os.path.exists(content_skeleton_path):
            content_skeleton_image = Image.open(content_skeleton_path).convert("L")
        else:
            if self.missing_warning_count < self.max_warnings:
                print(f"[Warning] Missing Content Skeleton: {content_skeleton_path}")
                self.missing_warning_count += 1
            content_skeleton_image = Image.new('L', content_image.size, 0)

        # ========================================================
        # Apply width-condition dropout during training.
        # ========================================================
        if self.phase == 'train':
            # With probability 0.3, zero the style and content width maps only.
            if random.random() < 0.3:
                # For transforms normalized to [-1, 1], zero-valued maps become -1.
                style_width_image = Image.new('L', style_image.size, 0)
                content_width_image = Image.new('L', content_image.size, 0)

        # --------------------------------------------------------
        # 7. Transforms
        # --------------------------------------------------------
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

            # Transform grayscale width and skeleton maps.
            # The optional fifth transform is used for target-width supervision.
            # This separate transform can use a different target resolution.
            if len(self.transforms) > 4:
                target_width_image = self.transforms[4](target_width_image)
            else:
                target_width_image = self.transforms[3](target_width_image)

            # Transform the other conditioning maps using the configured resolution.
            content_width_image = self.transforms[3](content_width_image)


            style_width_image = self.transforms[3](style_width_image)
            content_skeleton_image = self.transforms[3](content_skeleton_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            "content_width_image": content_width_image,
            "width_image": target_width_image,

            # Include auxiliary conditioning tensors in the returned sample.
            "style_width_image": style_width_image,
            "content_skeleton_image": content_skeleton_image,
        }

        return sample

    def __len__(self):
        return len(self.target_images)
