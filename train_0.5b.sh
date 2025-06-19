#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python main.py \
  --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
  --dataset_id "AI-MO/NuminaMath-TIR" \
  --output_dir "Qwen2.5-0.5B-GRPO-test" \
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