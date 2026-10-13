
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port=29501 \
    train.py \
    --seed=123 \
    --experience_name="swfont_stage1" \
    --data_root="data" \
    --output_dir="outputs/swfont_stage1" \
    --report_to="tensorboard" \
    --resolution=128 \
    --style_image_size=128 \
    --content_image_size=128 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=4 \
    --perceptual_coefficient=0.01 \
    --wes_loss_coeff=0.5 \
    --max_train_steps=100000 \
    --stop_at_step=100000 \
    --ckpt_interval=10000 \
    --gradient_accumulation_steps=4 \
    --log_interval=50 \
    --learning_rate=1e-4 \
    --lr_scheduler="partial_linear" \
    --lr_warmup_steps=10000 \
    --drop_prob=0.1 \
    --mixed_precision="bf16" \
    --wes_net_path="src/width_estimator/pretrained/wes_net.pth"
