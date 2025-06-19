#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python eval.py \
  --model_dir "Qwen2.5-3B-GRPO-test" \
  --dataset_id "AI-MO/NuminaMath-TIR" \
  --split "test[:50%]" \
  --batch_size 4 \
  --max_completion_length 512 \
  --max_prompt_length 256 \
  --output_file "results/eval_results_$(date +%Y%m%d_%H%M%S).json"
