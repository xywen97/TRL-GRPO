# trl_GRPO

## 项目简介
本项目基于 Huggingface Transformers 和 TRL（Transformer Reinforcement Learning）库，结合 LoRA 微调方法，实现了对数学推理类数据集的指令微调与奖励驱动训练，并支持自动化评测。

## 主要功能
- 支持自定义 reward 函数的 RL 微调（GRPO）
- 支持数学推理数据集的加载与格式化
- 支持 LoRA 低秩适配微调
- 支持训练与评测分离，命令行参数灵活配置
- 自动保存模型、评测结果

## 目录结构
```
trl_GRPO/
├── main.py           # 训练主入口，参数解析
├── train.py          # 训练流程实现
├── data_utils.py     # 数据加载与预处理
├── reward.py         # 奖励函数实现
├── eval.py           # 评测脚本
├── eval.sh           # 评测启动脚本
├── .gitignore        # Git忽略文件
├── README.md         # 项目说明
├── outputs/          # 推理/训练输出（建议忽略）
├── logs/             # 日志文件（建议忽略）
├── checkpoints/      # 检查点（建议忽略）
├── results/          # 评测结果输出（建议忽略）
└── ...
```

## 环境依赖
- Python 3.8+
- torch
- transformers
- trl
- peft
- datasets
- wandb（可选，用于日志追踪）
- tqdm

建议使用 conda 或 venv 虚拟环境，并通过 `pip install -r requirements.txt` 安装依赖。

## 训练用法

```bash
python main.py \
  --model_id "Qwen/Qwen2-0.5B-Instruct" \
  --dataset_id "AI-MO/NuminaMath-TIR" \
  --output_dir "Qwen2-0.5B-GRPO-test" \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --batch_size 16 \
  --max_completion_length 64 \
  --max_prompt_length 128 \
  --num_generations 4
```

## 评测用法

```bash
bash eval.sh
```
或手动：
```bash
python eval.py \
  --model_dir "Qwen2-0.5B-GRPO-test" \
  --dataset_id "AI-MO/NuminaMath-TIR" \
  --split "test[:5%]" \
  --batch_size 16 \
  --max_completion_length 512 \
  --max_prompt_length 256 \
  --output_file "results/eval_results.json"
```

## 结果输出
- 训练模型权重保存在 `output_dir` 目录下
- 评测结果为 json 格式，包含 prompt、completion、solution、reward 字段，并在终端输出整体准确率

## 备注
- 推荐使用 GPU 环境运行，评测时可通过 `CUDA_VISIBLE_DEVICES` 指定显卡
- 评测脚本默认使用原始预训练模型的分词器，模型权重用本地微调结果
- 详细参数可通过 `python main.py --help` 或 `python eval.py --help` 查看

---
如有问题欢迎反馈！ 