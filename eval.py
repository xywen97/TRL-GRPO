import argparse
from data_utils import make_conversation, SYSTEM_PROMPT
from reward import accuracy_reward
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="模型目录")
    parser.add_argument('--dataset_id', type=str, required=True, help="评测数据集")
    parser.add_argument('--split', type=str, default="test[:5%]", help="数据集分割")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_completion_length', type=int, default=64)
    parser.add_argument('--max_prompt_length', type=int, default=128)
    parser.add_argument('--output_file', type=str, default="eval_results.json")
    return parser.parse_args()

def main():
    args = parse_args()
    # 加载数据
    dataset = load_dataset(args.dataset_id, split=args.split)
    dataset = dataset.map(make_conversation)
    # 加载分词器（用原始预训练模型的分词器）
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto", device_map="auto")
    model.eval()
    results = []
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i:i+args.batch_size]  # batch 是 dict of lists
        batch_size_actual = len(batch["prompt"])
        batch_list = [{k: v[j] for k, v in batch.items()} for j in range(batch_size_actual)]
        prompts = [x["prompt"] for x in batch_list]
        # 拼接 prompt
        input_texts = [
            SYSTEM_PROMPT + "\n" + x[1]["content"] for x in prompts
        ]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_prompt_length).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_length,
                do_sample=False
            )
        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 评测
        solutions = [x["solution"] if "solution" in x else "" for x in batch_list]
        completions_for_reward = [[{"content": c}] for c in completions]
        rewards = accuracy_reward(completions_for_reward, solution=solutions)
        for j, (prompt, completion, solution, reward) in enumerate(zip(input_texts, completions, solutions, rewards)):
            results.append({
                "prompt": prompt,
                "completion": completion,
                "solution": solution,
                "reward": reward
            })
    # 保存结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # 打印整体准确率
    acc = sum([r["reward"] for r in results]) / len(results)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main() 