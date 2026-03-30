"""
Train LoRA / LoRA+ / AdaLoRA / Bayesian-LoRA on classification benchmarks.
Supports: ARC-Challenge, OBQA (OpenBookQA), Winogrande.
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
)

# Add parent directory to path for local peft (use abspath to avoid CWD ambiguity)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from peft import LoraConfig, AdaLoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from peft.tuners.lora.bayesian_lora_layer import BayesianLoRALayer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_file_logger(log_dir: str):
    """Add a FileHandler to logger pointing to log_dir/train.log."""
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


# ───────────────────── Logging Callback ─────────────────────

class LossLoggerCallback(TrainerCallback):
    """Forward Trainer's built-in log dict to Python logger (→ FileHandler)."""

    def on_log(self, _args, state, _control, logs=None, **_kwargs):
        if logs is None:
            return
        step = state.global_step
        parts = [f"step={step}"]
        for k, v in logs.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        logger.info("  ".join(parts))


# ───────────────────── Custom Trainer for Bayesian-LoRA ─────────────────────

class BayesianTrainer(Trainer):
    """Trainer that adds KL divergence to the task loss for ELBO training."""

    def __init__(self, *args, kl_weight=1.0, dataset_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.dataset_size = dataset_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss

        # Collect KL from all BayesianLinear2 layers
        kl = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
        for module in model.modules():
            if hasattr(module, "kl_divergence"):
                kl = kl + module.kl_divergence().to(task_loss.dtype)

        loss = task_loss + self.kl_weight * kl / self.dataset_size

        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"step={self.state.global_step} "
                f"task_loss={task_loss.item():.4f} "
                f"kl={kl.item():.4f} "
                f"total_loss={loss.item():.4f}"
            )

        return (loss, outputs) if return_outputs else loss


# ───────────────────── Dataset Loading ─────────────────────

DATASET_CONFIG = {
    "ARC-Easy": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "train_split": "train",
        "test_split": "test",
    },
    "ARC-Challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "train_split": "train",
        "test_split": "test",
    },
    "OBQA": {
        "path": "allenai/openbookqa",
        "name": "main",
        "train_split": "train",
        "test_split": "test",
    },
    "Winogrande": {
        "path": "allenai/winogrande",
        "name": "winogrande_m",
        "train_split": "train",
        "test_split": "validation",
    },
}


def format_example(example, dataset_name):
    """Format a dataset example into prompt + answer."""
    if dataset_name in ("ARC-Challenge", "ARC-Easy"):
        question = example["question"]
        choices = example["choices"]
        labels = choices["label"]
        texts = choices["text"]
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        answer = example["answerKey"]
        prompt = f"Question: {question}\n{options}\nAnswer:"
        return prompt, f" {answer}"

    elif dataset_name == "OBQA":
        question = example["question_stem"]
        choices = example["choices"]
        labels = choices["label"]
        texts = choices["text"]
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        answer = example["answerKey"]
        prompt = f"Question: {question}\n{options}\nAnswer:"
        return prompt, f" {answer}"

    elif dataset_name == "Winogrande":
        sentence = example["sentence"]
        opt1 = example["option1"]
        opt2 = example["option2"]
        answer = example["answer"]  # "1" or "2"
        label = "A" if answer == "1" else "B"
        prompt = f"Sentence: {sentence}\nA. {opt1}\nB. {opt2}\nAnswer:"
        return prompt, f" {label}"


def build_dataset(tokenizer, dataset_name, split, max_length=512):
    """Load and tokenize classification dataset for causal LM training."""
    cfg = DATASET_CONFIG[dataset_name]
    split_name = cfg["train_split"] if split == "train" else cfg["test_split"]
    ds = load_dataset(cfg["path"], cfg["name"], split=split_name, trust_remote_code=True)

    # Filter out Winogrande samples with no answer
    if dataset_name == "Winogrande":
        ds = ds.filter(lambda x: x["answer"] in ("1", "2"))

    tokenized = []
    for example in ds:
        prompt, answer = format_example(example, dataset_name)
        full_text = prompt + answer
        enc = tokenizer(full_text, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
        prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length)
        prompt_len = len(prompt_enc["input_ids"])

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt tokens
        labels[attention_mask == 0] = -100  # mask padding

        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return tokenized


# ───────────────────── Model Setup ─────────────────────

def setup_model(args, total_steps=None, tokenizer=None):
    """Load base model and attach adapter based on method."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    target_modules = [m.strip() for m in args.target_modules.split(",")]

    if args.method == "LoRA":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,  # no dropout for fair comparison
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    elif args.method == "LoRA_plus":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # LoRA+ optimizer created later in training

    elif args.method == "AdaLoRA":
        config = AdaLoraConfig(
            init_r=args.lora_r * 2,
            target_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            total_step=total_steps,
        )
        model = get_peft_model(model, config)

    elif args.method == "Bayesian_LoRA":
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        config._register_custom_module({nn.Linear: BayesianLoRALayer})
        model = get_peft_model(model, config)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    model.print_trainable_parameters()
    return model, tokenizer


# ───────────────────── Training ─────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["LoRA", "LoRA_plus", "AdaLoRA", "Bayesian_LoRA"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ARC-Easy", "ARC-Challenge", "OBQA", "Winogrande"])
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,lm_head")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--loraplus_lr_ratio", type=float, default=16.0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to write train.log (in addition to stdout)")
    parser.add_argument("--inducing_rows", type=int, default=None,
                        help="Inducing matrix rows (r). Overrides inducing.json if set.")
    parser.add_argument("--inducing_cols", type=int, default=None,
                        help="Inducing matrix cols (c). Overrides inducing.json if set.")
    args = parser.parse_args()

    if args.log_dir:
        setup_file_logger(args.log_dir)

    # Override inducing.json if r/c specified
    if args.inducing_rows is not None or args.inducing_cols is not None:
        import json
        inducing_path = os.path.join(_REPO_ROOT, "inducing.json")
        with open(inducing_path, "r") as f:
            ind_cfg = json.load(f)
        if args.inducing_rows is not None:
            ind_cfg["inducing_rows"] = args.inducing_rows
        if args.inducing_cols is not None:
            ind_cfg["inducing_cols"] = args.inducing_cols
        with open(inducing_path, "w") as f:
            json.dump(ind_cfg, f, indent=2)
        logger.info(f"inducing.json updated: rows={ind_cfg['inducing_rows']}, cols={ind_cfg['inducing_cols']}")

    import math
    torch.manual_seed(args.seed)

    # Load tokenizer first to build dataset, so we can compute total_steps for AdaLoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Building {args.dataset} training set...")
    train_dataset = build_dataset(tokenizer, args.dataset, "train", args.max_length)
    logger.info(f"Training samples: {len(train_dataset)}")

    steps_per_epoch = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.num_epochs
    logger.info(f"Total optimizer steps: {total_steps}")

    model, tokenizer = setup_model(args, total_steps=total_steps, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        save_safetensors=False,
        remove_unused_columns=False,
    )

    # LoRA+ uses custom optimizer; Bayesian-LoRA uses custom Trainer with KL loss
    loss_logger = LossLoggerCallback()

    if args.method == "LoRA_plus":
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=args.learning_rate,
            loraplus_lr_ratio=args.loraplus_lr_ratio,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            tokenizer=tokenizer,
            optimizers=(optimizer, None),
            callbacks=[loss_logger],
        )
    elif args.method == "Bayesian_LoRA":
        trainer = BayesianTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            tokenizer=tokenizer,
            kl_weight=1.0,
            dataset_size=len(train_dataset),
            callbacks=[loss_logger],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            tokenizer=tokenizer,
            callbacks=[loss_logger],
        )

    logger.info(f"Starting {args.method} training on {args.dataset}...")
    trainer.train()

    save_dir = os.path.join(args.output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Adapter saved to {save_dir}")


if __name__ == "__main__":
    main()
