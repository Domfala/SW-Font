# SW-Font: Few-Shot Font Generation via Stroke-Width Spatial Distribution Modeling

**Official PyTorch implementation of SW-Font**, a few-shot font generation framework that incorporates spatial stroke-width priors and topological skeleton guidance into conditional diffusion.

[Overview](#overview) · [Generation Results](#generation-results) · [Installation](#installation) · [Data Preparation](#data-preparation) · [Inference](#inference) · [Training](#training) · [TODO](#todo)

## Overview

SW-Font synthesizes Chinese glyphs from content and style references. It combines three main components:

- **Adapted Stroke Width Transform (A-SWT):** constructs spatial stroke-width maps from glyph images.
- **Geometry-Guided Cross-Attention (G-GCA):** integrates structural guidance and width-aware style features into the diffusion network.
- **Width Estimation Network (WES-Net):** provides auxiliary stroke-width supervision during training.

## Generation Results

![Font generation results](assets/swfont_qualitative_results_top_only.png)

*Examples of Chinese font generation using SW-Font.*

## Installation

```bash
git clone https://github.com/Domfala/SW-Font.git
cd SW-Font
pip install -r requirements.txt
```

A CUDA-compatible PyTorch environment is recommended. Ensure that PyTorch, TorchVision, and xFormers versions are compatible with your CUDA installation.

## Data Preparation

SW-Font uses **rendered glyph images** for training. You can prepare your own data from appropriately licensed TTF/OTF fonts by rendering the required characters to `128 × 128` PNG images.

For font dataset preparation, see [FFG-benchmarks](https://github.com/clovaai/fewshot-font-generation) and [MSD-Font](https://github.com/fubinfb/MSD-Font). Font resources referenced in related projects include [FounderType](https://www.foundertype.com/index.php/FindFont/index); please check the applicable font licenses before use.

Organize the training data as follows (and use the same layout under `data/val/`):

```text
data/train/
├── ContentImage/        # Content glyphs
├── ContentSkeleton/     # Content skeletons
├── ContentWidth/        # Content width maps
├── TargetImage/         # Style folders, e.g. 001/001+汉.png
└── PredWidthImage/      # Matching style-width maps
```

Width maps can be prepared using A-SWT:

```bash
python src/swt/A-SWT.py --input data/train/TargetImage --output data/train/PredWidthImage
python src/swt/A-SWT.py --input data/train/ContentImage --output data/train/ContentWidth
```

The A-SWT implementation and its default preprocessing settings are available in [A-SWT.py](src/swt/A-SWT.py).

Generate aligned content skeletons separately and place them under `ContentSkeleton/`. Ensure that all glyphs and auxiliary maps have matching filenames and directory structures. The evaluation splits are **SFUC** (seen fonts, unseen characters), **UFSC** (unseen fonts, seen characters), and **UFUC** (unseen fonts, unseen characters).

For dataset split definitions and character complexity categories, see [Dataset Splits](docs/dataset_splits.md).

## Pretrained Models

For inference, place the main-network checkpoint files in:

```text
checkpoints/swfont/
├── unet.pth
├── style_encoder.pth
└── content_encoder.pth
```

The auxiliary WES-Net checkpoint is used during training, not inference. Specify its path using `--wes_net_path` in the training scripts.

## Inference

SW-Font generates `128 × 128` glyphs using DPM-Solver++ (20 sampling steps by default).

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/sample.sh data/test_list.txt outputs/inference
```

Each line of `data/test_list.txt` contains:

```text
content_image_path|style_image_path|output_subdirectory|output_filename
```

The inference script locates the corresponding content skeleton and style-width map by replacing the `/C/` path component with `/C_Ske/` and `/F/` with `/F_wid/`. Keep these paired directories consistent when preparing test inputs.

## Training

WES-Net is pretrained separately with A-SWT stroke-width maps and kept frozen during main-network training.

The SW-Font main network is trained at `128 × 128` resolution for **200K steps** in two consecutive phases:

| Phase | Training steps | Learning rate | Stroke-width loss weight |
| --- | --- | --- | --- |
| Phase 1 | 0–100K | `1e-4` | `0.5` |
| Phase 2 | 100K–200K | `2e-5` | `0.1` |

Run the phases in sequence:

```bash
# Phase 1: 0–100K
bash scripts/train_stage1.sh

# Phase 2: 100K–200K (continues from Phase 1)
bash scripts/train_stage2.sh
```

Phase 2 loads the 100K-step model weights. The training scripts require the prepared dataset and the pretrained WES-Net checkpoint.

## TODO

- [x] Release the SW-Font source code and training/inference scripts.
- [x] Add generation results and data preparation instructions.
- [ ] Release the pretrained SW-Font main-network weights.
- [ ] Provide sample inputs and preprocessing examples.
- [ ] Publish evaluation scripts and benchmark instructions.

## License

See [LICENSE](LICENSE) for the repository license. Third-party datasets and fonts remain subject to their respective licenses.

## Citation

If you use this project in your research, please cite *SW-Font: Few-Shot Font Generation via Stroke-Width Spatial Distribution Modeling*.

## Acknowledgments

We thank the authors of [FontDiffuser](https://github.com/yeungchenwa/fontdiffuser), [FFG-benchmarks](https://github.com/clovaai/fewshot-font-generation), [MSD-Font](https://github.com/fubinfb/MSD-Font), and other open-source projects for making their work publicly available.
