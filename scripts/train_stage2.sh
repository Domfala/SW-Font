
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port=29501 \
    train.py \
    --seed=123 \
    --experience_name="swfont_stage2" \
    --data_root="data" \
    --output_dir="outputs/swfont_stage2" \
    --resume_ckpt_path="outputs/swfont_stage1/global_step_100000" \
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
    --wes_loss_coeff=0.1 \
    --max_train_steps=200000 \
    --stop_at_step=200000 \
    --ckpt_interval=10000 \
    --gradient_accumulation_steps=4 \
    --log_interval=50 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --drop_prob=0.1 \
    --mixed_precision="bf16" \
    --wes_net_path="src/width_estimator/pretrained/wes_net.pth"
