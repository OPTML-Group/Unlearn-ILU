import os
import sys

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("src")
import torch
from peft import  get_peft_model, LoraConfig, AdaLoraConfig, TaskType
from pruner.utils import WrappedGPT, find_layers
from dataset import get_dataset
from metrics import (
    eval_copyright,
    eval_few_shots,
    eval_PII,
    eval_ppl,
    eval_tofu,
    eval_toxic,
    eval_wmdp,
)
from optim import create_sophia_optimizer
from unlearn import GenerateMask, get_unlearn_method


class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.unlearn_method = kwargs["unlearn_method"]
        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.alpha = kwargs.get("alpha", None)
        self.gamma = kwargs.get("gamma", None)
        self.mask_path = kwargs.get("mask_path", None)
        self.task_name = kwargs.get("task_name", None)
        self.k = kwargs.get("k", 100)
        self.sophia = kwargs.get("sophia", False)
        self.betas_low = kwargs.get("betas_low", 0.9)
        self.betas_high = kwargs.get("betas_high", 0.95)
        self.betas = (self.betas_low, self.betas_high)
        self.rho = kwargs.get("rho", 0.03)
        self.p = kwargs.get("p", 0.0)
        self.q = kwargs.get("q", 0.0)
        self.if_llama = "llama" in self.model_name
        self.resume_path = kwargs.get("resume_path", None)
        self.max_steps = kwargs.get("max_steps", -1)
        self.use_lora = kwargs.get("use_lora", False)
        self.if_wanda = False
        self.mu = kwargs.get("mu", 1e-3)
    def init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        if self.use_lora:
            peft_config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=["q_proj","v_proj"], 
                lora_dropout=0.05,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())

        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except:
            self.device = torch.device("cuda:0")

    def init_dataset(self):
        unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_dataset = test_dataset
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        if self.max_steps == -1:
            self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
            self.steps_per_epoch = len(unlearn_dataset) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
        else:
            self.steps_per_epoch = self.max_steps // self.num_epochs

    def init_unlearner(self, logger):
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        if self.unlearn_method == "origin":
            self.unlearner = None
            return
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            output_dir=unlearn_checkpoint,
            optim="adamw_torch",
            save_steps=self.max_steps,
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            save_total_limit=1,
            report_to=[], 
        )
        if self.optimizer is not None:
            self.unlearner = get_unlearn_method(
                name=self.unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.mask,
                optimizers=(self.optimizer, None),
            )
        else:
            self.unlearner = get_unlearn_method(
                name=self.unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.mask,
                if_wanda=self.if_wanda,
            )

    def init_mask(self, logger):
        if self.mask_path is None:
            self.mask = None
            return

        elif os.path.exists(self.mask_path):
            self.mask = torch.load(self.mask_path)
            parts = self.mask_path.split("/")
            score_type = parts[-2]
            if score_type == "wanda":
                self.if_wanda = True
            else:
                self.if_wanda = False
            if not self.if_wanda:
                for key, tensor in self.model.named_parameters():
                    self.mask[key] = self.mask[key].to(tensor.device)
            else:
                try:
                    layers = self.model.model.layers
                except:
                    layers = self.model.model.decoder.layers
                cnt = 0
                with torch.no_grad():
                    for layer in layers:
                        subset = find_layers(layer)
                        for name in subset:
                            print(subset[name].weight.device)
                            self.mask[cnt] = self.mask[cnt].to(subset[name].weight.device)
                            cnt+=1 
            return
        else:
            parts = self.mask_path.split("/")
            score_type = parts[-2]
            if score_type == "wanda":
                self.if_wanda = True
            else:
                self.if_wanda = False

            ratio = float(parts[-1].split("_")[-1].split(".p")[0])
            root = logger.get_root()
            mask_dir = self.mask_path.replace(f"with_{ratio}.pt", "")
            if mask_dir == self.mask_path:
                mask_dir = self.mask_path.replace(f"with_{self.p}_{self.q}.pt", "")
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            mask_args = transformers.TrainingArguments(
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_steps=max(1, self.max_steps // 10),
                max_steps=self.max_steps,
                learning_rate=self.lr,
                bf16=True,
                bf16_full_eval=False,
                logging_steps=max(1, self.max_steps // 20),
                logging_dir=f"{root}/logs",
                optim="adamw_torch",
                save_steps=self.steps_per_epoch,
                weight_decay=self.weight_decay,
                remove_unused_columns=False,
                save_total_limit=3,
                output_dir=mask_dir,
                report_to=[],
            )
            if score_type == "wanda":
                unlearn_dataset,_,_,_ = get_dataset(
                    self.dataset_names,
                    self.tokenizer,
                    self.dataset_seed,
                    128,
                    self.self_retain,
                    self.if_llama,
                )
                self.mask = GenerateMask(
                    score_type=score_type,
                    ratios=[ratio],
                    mask_dir=mask_dir,
                    model=self.model,
                    data_collator=self.unlearn_collator,
                    tokenizer=self.tokenizer,
                    train_dataset=unlearn_dataset,
                    eval_dataset=None,
                    compute_metrics=None,
                    args=mask_args,
                    p=self.p,
                    q=self.q,
                    mu=self.mu,
                ).get_mask()
            else:
                GenerateMask(
                    score_type=score_type,
                    ratios=[ratio],
                    mask_dir=mask_dir,
                    model=self.model,
                    data_collator=self.unlearn_collator,
                    tokenizer=self.tokenizer,
                    train_dataset=self.unlearn_dataset,
                    eval_dataset=None,
                    compute_metrics=None,
                    args=mask_args,
                    p=self.p,
                    q=self.q,
                    mu=self.mu,
                ).get_mask()
            if score_type == "snip_forget_reinit":
                self.mask = None
                os.system(f"rm -rf {self.mask_path}")
                return
            self.mask = torch.load(self.mask_path)
            if not self.if_wanda:
                for key, tensor in self.model.named_parameters():
                    self.mask[key] = self.mask[key].to(tensor.device)    
            else:
                try:
                    layers = self.model.model.layers
                except:
                    layers = self.model.model.decoder.layers
                cnt = 0
                with torch.no_grad():
                    for layer in layers:
                        subset = find_layers(layer)
                        for name in subset:
                            print(subset[name].weight.device)
                            self.mask[cnt] = self.mask[cnt].to(subset[name].weight.device)
                            cnt+=1
    def init_optimizer(self):
        if self.sophia:
            self.optimizer = create_sophia_optimizer(
                self.model,
                lr=self.lr,
                betas=self.betas,
                rho=self.rho,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = None

    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        if self.resume_path is not None:
            model_name = self.resume_path
        else:
            model_name = os.path.join(root, "checkpoints")
        if self.task_name != "tofu":
            eval_ppl(model_name=model_name, output_path=f"{root}/ppl.json")
            torch.cuda.empty_cache()
            if self.task_name == "wmdp":
                eval_few_shots(model_name=model_name,  task_list=["mmlu"],output_path=f"{root}/mmlu.json")
            else:
                eval_few_shots(model_name=model_name, output_path=f"{root}/few_shots.json")
        torch.cuda.empty_cache()
        if self.task_name == "toxic":
            eval_toxic(
                model_name=model_name, output_dir=root, dataset=self.unlearn_dataset
            )
        elif self.task_name == "copyright":
            eval_copyright(model_name=model_name, output_dir=root,batch_size=16,if_llama=self.if_llama)
            torch.cuda.empty_cache()
        elif self.task_name == "tofu":

            forget_subset = self.dataset_names["forget"].split("_")[1]
            retain_subset = self.dataset_names["retain"].split("_")[1]
            if forget_subset == "full":
                forget_subset = "forget10"
            elif "retain" in forget_subset:
                forget_ratio = 100-int(forget_subset.split("retain")[-1])
                forget_subset = f"forget{forget_ratio}"
            if retain_subset == "full":
                retain_subset = "retain90"
            eval_tofu(
                model_name=model_name,
                output_dir=root,
                forget_subset=forget_subset,
                retain_subset=retain_subset,
                if_llama=self.if_llama,
            )
        elif self.task_name == "wmdp":
            eval_few_shots(model_name=model_name, task_list=["wmdp"],output_path=f"{root}/wmdp.json")

    def save(self, logger):
        logger.save_ckpt("model", self.model, self.use_lora)
        logger.save_ckpt("tokenizer", self.tokenizer, self.use_lora)

    def run(self, logger):
        if self.resume_path is None:
            self.init_model()
            self.init_optimizer()
            self.init_dataset()
            self.init_mask(logger)
            self.init_unlearner(logger)
            if self.unlearner:
                self.unlearner.train()
            self.save(logger)
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            self.eval(logger)
        else:
            self.init_model()
            self.init_dataset()
            self.eval(logger)


def get(**kwargs):
    return Unlearn(**kwargs)
