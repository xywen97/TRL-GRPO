#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python eval.py \
  --model_dir "Qwen2.5-3B-GRPO-test" \
  --dataset_id "AI-MO/NuminaMath-TIR" \
  --split "test[:5%]" \
  --batch_size 2 \
  --max_completion_length 512 \
  --max_prompt_length 256 \
  --output_file "results/eval_results.json"
