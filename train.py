from data_utils import load_and_process_data
from reward import format_reward, accuracy_reward
from trl import GRPOConfig, GRPOTrainer
import wandb
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def train(args):
    train_dataset, test_dataset = load_and_process_data(args.dataset_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = GRPOConfig(
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        report_to="wandb",
        logging_steps=args.logging_steps,
        push_to_hub=False,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    trainer.save_model(args.output_dir) 