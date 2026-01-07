# SW-Font
Official implementation of "Structure-Aware Few-shot Font Generation" using Geometry-Guided Content Adaptation (G-GCA) and Width Estimation Network (WES-Net).
# Structure-Aware Font Diffusion with G-GCA and WES-Net
(Coming soon)

## 📖 Abstract
This repository contains the official implementation of our research on Few-shot Font Generation. 
We propose a novel **Geometry-Guided Content Adaptation (G-GCA)** module and a **Width Estimation Network (WES-Net)** to solve the problem of structural inconsistency and stroke width instability in font generation.

## 🚀 News
- **[2025-01-06]** Repository created. The code and pretrained models will be released soon!

## 🛠️ Architecture
Our method consists of two key components:
1. **G-GCA Module**: Injects adaptive gate maps to maintain structural integrity.
2. **WES-Net**: Supervises stroke width to ensure style consistency.

*(More details will be updated upon code release)*

## 📂 Data Preparation
(Coming soon)

## 🚀 Training & Inference
(The training scripts and usage instructions will be updated here.)

## 📊 Results

| Method | SSIM ↑ | RMSE ↓ | FID ↓ |
| :--- | :---: | :---: | :---: |
| FontDiffuser (Baseline) | 0.xxx | 0.xxx | xx.x |
| **Ours (G-GCA + WES)** | **0.xxx** | **0.xxx** | **xx.x** |

## 📝 Citation
If you find our work useful, please consider citing:
```bibtex
@article{yourname2025structure,
  title={Structure-Aware Font Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
