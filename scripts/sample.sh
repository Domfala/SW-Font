#!/usr/bin/env bash

TEST_LIST="${1:-data/test_list.txt}"
OUTPUT_DIR="${2:-outputs/inference}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python sample.py \
    --ckpt_dir="checkpoints/swfont" \
    --test_list_path="$TEST_LIST" \
    --save_image_dir="$OUTPUT_DIR" \
    --style_width_dir="F_wid" \
    --content_skeleton_dir="C_Ske" \
    --resolution=128 \
    --content_image_size=128 \
    --style_image_size=128 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --num_inference_steps=20 \
    --guidance_scale=4 \
    --rescale_phi=0.7 \
    --seed=123
