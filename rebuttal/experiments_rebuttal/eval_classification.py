"""
Evaluate LoRA / LoRA+ / AdaLoRA / Bayesian-LoRA on classification benchmarks.
Computes: ACC, ECE, NLL.  Supports MC sampling (n_mc > 1) for Bayesian-LoRA.
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from peft import PeftModel, LoraConfig
from peft.tuners.lora.bayesian_lora_layer import BayesianLoRALayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_file_logger(log_dir: str):
    """Add a FileHandler to logger pointing to log_dir/eval.log."""
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "eval.log"), mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


# ───────────────────── Dataset Loading ─────────────────────

DATASET_CONFIG = {
    "ARC-Easy": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Easy",
        "test_split": "test",
    },
    "ARC-Challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "test_split": "test",
    },
    "OBQA": {
        "path": "allenai/openbookqa",
        "name": "main",
        "test_split": "test",
    },
    "Winogrande": {
        "path": "allenai/winogrande",
        "name": "winogrande_m",
        "test_split": "validation",
    },
}


def load_test_data(dataset_name):
    """Load test data and return list of (prompt, answer_label, choice_labels)."""
    cfg = DATASET_CONFIG[dataset_name]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["test_split"], trust_remote_code=True)

    examples = []
    for ex in ds:
        if dataset_name in ("ARC-Challenge", "ARC-Easy"):
            question = ex["question"]
            choices = ex["choices"]
            labels = choices["label"]
            texts = choices["text"]
            options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            prompt = f"Question: {question}\n{options}\nAnswer:"
            answer = ex["answerKey"]
            examples.append({"prompt": prompt, "answer": answer, "labels": labels})

        elif dataset_name == "OBQA":
            question = ex["question_stem"]
            choices = ex["choices"]
            labels = choices["label"]
            texts = choices["text"]
            options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            prompt = f"Question: {question}\n{options}\nAnswer:"
            answer = ex["answerKey"]
            examples.append({"prompt": prompt, "answer": answer, "labels": labels})

        elif dataset_name == "Winogrande":
            if ex["answer"] not in ("1", "2"):
                continue
            sentence = ex["sentence"]
            opt1, opt2 = ex["option1"], ex["option2"]
            prompt = f"Sentence: {sentence}\nA. {opt1}\nB. {opt2}\nAnswer:"
            answer = "A" if ex["answer"] == "1" else "B"
            examples.append({"prompt": prompt, "answer": answer, "labels": ["A", "B"]})

    return examples


# ───────────────────── Evaluation Logic ─────────────────────

def get_choice_logprobs(model, tokenizer, prompt, choice_labels, device):
    """
    Compute log-probabilities for each choice label token given the prompt.
    Returns a tensor of shape (num_choices,) with log-probs.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)
    # logits at the last prompt token predict the next token
    last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
    log_probs = F.log_softmax(last_logits.float(), dim=-1)

    choice_logprobs = []
    for label in choice_labels:
        token_id = tokenizer.encode(f" {label}", add_special_tokens=False)
        if len(token_id) == 0:
            token_id = tokenizer.encode(label, add_special_tokens=False)
        # Use the first token if label tokenizes to multiple tokens
        tid = token_id[0]
        choice_logprobs.append(log_probs[tid].item())

    return torch.tensor(choice_logprobs)


def mc_get_choice_logprobs(model, tokenizer, prompt, choice_labels, device, n_mc=4):
    """
    MC sampling: run n_mc forward passes and average the softmax probabilities.
    Returns log of averaged probabilities: (num_choices,).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    all_probs = []
    for _ in range(n_mc):
        with torch.no_grad():
            outputs = model(**enc)
        last_logits = outputs.logits[0, -1, :]
        probs = F.softmax(last_logits.float(), dim=-1)

        choice_probs = []
        for label in choice_labels:
            token_id = tokenizer.encode(f" {label}", add_special_tokens=False)
            if len(token_id) == 0:
                token_id = tokenizer.encode(label, add_special_tokens=False)
            tid = token_id[0]
            choice_probs.append(probs[tid].item())
        all_probs.append(choice_probs)

    # Average probabilities across MC samples
    avg_probs = np.mean(all_probs, axis=0)
    avg_probs = avg_probs / avg_probs.sum()  # renormalize
    return torch.tensor(np.log(avg_probs + 1e-12))


def compute_ece(confidences, accuracies, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # include right boundary for last bin
            mask = (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (count / total) * abs(avg_acc - avg_conf)
    return ece


def evaluate_model(model, tokenizer, examples, device, n_mc=1):
    """Run evaluation and return ACC, ECE, NLL."""
    use_mc = n_mc > 1

    all_correct = []
    all_confidence = []
    all_nll = []

    for ex in tqdm(examples, desc="Evaluating"):
        prompt = ex["prompt"]
        answer = ex["answer"]
        labels = ex["labels"]

        if use_mc:
            log_probs = mc_get_choice_logprobs(
                model, tokenizer, prompt, labels, device, n_mc
            )
        else:
            log_probs = get_choice_logprobs(
                model, tokenizer, prompt, labels, device
            )

        probs = F.softmax(log_probs.float(), dim=-1)
        pred_idx = probs.argmax().item()
        pred_label = labels[pred_idx]

        correct = int(pred_label == answer)
        confidence = probs[pred_idx].item()

        # NLL of the correct answer
        answer_idx = labels.index(answer)
        nll = -log_probs[answer_idx].item()

        all_correct.append(correct)
        all_confidence.append(confidence)
        all_nll.append(nll)

    all_correct = np.array(all_correct)
    all_confidence = np.array(all_confidence)
    all_nll = np.array(all_nll)

    acc = all_correct.mean() * 100
    ece = compute_ece(all_confidence, all_correct)
    nll = all_nll.mean()

    return acc, ece, nll


# ───────────────────── Main ─────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None,
                        choices=["LoRA", "LoRA_plus", "AdaLoRA", "Bayesian_LoRA"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ARC-Easy", "ARC-Challenge", "OBQA", "Winogrande"])
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to saved LoRA adapter (omit for base model eval)")
    parser.add_argument("--n_mc", type=int, default=4,
                        help="Number of MC forward passes (>1 for Bayesian)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to write eval.log (in addition to stdout)")
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

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    logger.info(f"Loading base model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Load adapter (skip for base model eval)
    if args.lora_path is not None:
        logger.info(f"Loading adapter from: {args.lora_path}")
        if args.method == "Bayesian_LoRA":
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(args.lora_path)
            peft_config._register_custom_module({nn.Linear: BayesianLoRALayer})
            model = PeftModel.from_pretrained(model, args.lora_path, config=peft_config)
            n_mc = args.n_mc
        else:
            model = PeftModel.from_pretrained(model, args.lora_path)
            n_mc = 1
    else:
        logger.info("No adapter provided — evaluating base model")
        n_mc = 1

    model = model.to(device)
    model.eval()

    # Load test data
    logger.info(f"Loading {args.dataset} test set...")
    examples = load_test_data(args.dataset)
    logger.info(f"Test samples: {len(examples)}")

    # Evaluate
    acc, ece, nll = evaluate_model(model, tokenizer, examples, device, n_mc)

    results = {
        "method": args.method,
        "dataset": args.dataset,
        "n_mc": n_mc,
        "seed": args.seed,
        "ACC": round(acc, 2),
        "ECE": round(ece, 4),
        "NLL": round(nll, 4),
        "num_examples": len(examples),
    }
    logger.info(f"Results: ACC={acc:.2f}%, ECE={ece:.4f}, NLL={nll:.4f}")

    # Save results
    if args.output_file is None:
        args.output_file = os.path.join(
            os.path.dirname(args.lora_path), "eval_results.json"
        )
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
