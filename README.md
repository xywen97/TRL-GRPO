# trl_GRPO

## 项目简介
本项目基于 Huggingface Transformers 和 TRL（Transformer Reinforcement Learning）库，结合 LoRA 微调方法，实现了对数学推理类数据集的指令微调与奖励驱动训练，并支持自动化评测与 reward 规则单元测试。

## 主要功能
- 支持自定义 reward 函数的 RL 微调（GRPO）
- 支持数学推理数据集的加载与格式化
- 支持 LoRA 低秩适配微调
- 支持训练、评测、reward 规则单元测试分离
- 自动保存模型、评测结果
- 明确的输出格式规范，reward 规则可直接本地测试

## 目录结构
```
trl_GRPO/
├── main.py           # 训练主入口，参数解析
├── train.py          # 训练流程实现
├── data_utils.py     # 数据加载与预处理
├── reward.py         # 奖励函数实现及单元测试
├── requirements.txt  # 环境文件
├── eval.py           # 评测脚本
├── eval.sh           # 评测启动脚本
├── .gitignore        # Git忽略文件
├── README.md         # 项目说明
├── ckpts/            # 模型训练输出文件夹
├── results/          # 评测结果输出
└── ...
```

## 环境依赖 (参考requirements.txt文件)
- Python 3.10+
- torch
- transformers
- trl
- peft
- datasets
- wandb（可选，用于日志追踪）或者使用tensorboard
- tqdm

建议使用 conda 或 venv 虚拟环境，并通过 `pip install -r requirements.txt` 安装依赖。

## 输出格式规范
- 推荐模型输出格式：
  ```
  <think>推理过程</think><answer>\boxed{最终答案}</answer>
  ```
- reward 规则：
  - `<think>` 和 `<answer>` 必须成对出现，否则 reward=0
  - `<think>` 和 `<answer>` 成对出现，且 `<answer>` 内有 `\boxed{}`，reward=1.0
  - `<think>` 和 `<answer>` 成对出现，但 `<answer>` 内无 `\boxed{}`，reward=0.5

## 训练用法

```bash
python main.py \
  --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
  --dataset_id "AI-MO/NuminaMath-TIR" \
  --output_dir "ckpts/Qwen2.5-0.5B-GRPO-test-$(date +%Y%m%d_%H%M%S)" \
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

## reward.py 单元测试用法

直接运行即可测试 reward 规则和答案提取效果：
```bash
python reward.py
```
- 会输出 extract_boxed、format_reward、accuracy_reward 的多组测试用例结果，便于调试和验证。

## 结果输出
- 训练模型权重保存在 `output_dir` 目录下
- 评测结果为 json 格式，包含 prompt、completion、solution、reward 字段，并在终端输出整体准确率

## 备注
- 推荐使用 GPU 环境运行，评测时可通过 `CUDA_VISIBLE_DEVICES` 指定显卡
- 评测脚本默认使用原始预训练模型的分词器，模型权重用本地微调结果
- 详细参数可通过 `python main.py --help` 或 `python eval.py --help` 查看

---
如有问题欢迎反馈！ 