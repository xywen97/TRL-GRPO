#!/bin/bash
# 首次使用前，配置 accelerate：
# accelerate config
# 按提示选择多卡单机等配置。
export NCCL_P2P_LEVEL=NVL
export ACCELERATE_USE_FSDP=False

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file "deepspeed_configs/zero3.yaml" main.py \
  --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
  --dataset_id "AI-MO/NuminaMath-TIR" \
  --output_dir "ckpts/Qwen2.5-0.5B-GRPO-test-$(date +%Y%m%d_%H%M%S)" \
  --learning_rate 1e-5 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_completion_length 512 \
  --max_prompt_length 256 \
  --num_generations 4 \
  --warmup_ratio 0.05 \
  --logging_steps 16 \
  --save_steps 16 \
  --save_strategy "epoch"