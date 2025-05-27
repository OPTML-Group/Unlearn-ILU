import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.append("src")
import argparse
import random

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling

from dataset.HorryPotter import HP


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--cache_dir", type=str, default=".cache", help="Cache directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of warmup steps"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--save_dir", type=str, default="files/models/hp", help="Save dir"
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = args_parser()
    set_seed(args.seed)
    dataset = HP("HP")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, use_fast=False
    )
    dataset = dataset.build_pretrain_dataset(tokenizer)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        save_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        output_dir=args.save_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
