import argparse
from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument('--dataset_id', type=str, default="AI-MO/NuminaMath-TIR")
    parser.add_argument('--output_dir', type=str, default="Qwen2-0.5B-GRPO-test")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_completion_length', type=int, default=64)
    parser.add_argument('--max_prompt_length', type=int, default=128)
    parser.add_argument('--num_generations', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument('--save_steps', type=int, default=20)
    parser.add_argument('--save_strategy', type=str, default="epoch")
    # 可根据需要添加更多参数
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)