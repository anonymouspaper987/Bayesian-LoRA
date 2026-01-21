# -*- coding: utf-8 -*-
from typing import Tuple, List
import os
import sys
import math
import logging
import importlib
import hydra
import torch as t
import torch.nn.functional as F
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torchmetrics import Accuracy
from transformers import get_cosine_schedule_with_warmup
from transformers.modeling_outputs import ModelOutput
from torchmetrics.classification import MulticlassCalibrationError
from bnn.bayesianize import bayesianize_
from utils.efficiency_calculation import count_trainable_params
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
sys.path.insert(0, parent_dir)

from utils import dsets
from utils.loggers import setup_logging
from utils.setup_llm import setup_llm


import torch

@torch.no_grad()
def simple_ece(pred_probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:

    if targets.ndim != 1:
        targets = targets.view(-1)
  

    conf, pred = pred_probs.max(dim=1)                    # [N], [N]
    correct = (pred == targets).to(pred_probs.dtype)      # [N], float

    N = pred_probs.size(0)
    if n_bins < 1:
        return 0.0
    if n_bins == 1:
        return float((conf - correct).abs().mean())

    boundaries = torch.linspace(
        0.0, 1.0, steps=n_bins + 1, device=pred_probs.device, dtype=pred_probs.dtype
    )[1:-1]  

    bin_ids = torch.bucketize(conf, boundaries, right=False)

    counts = torch.bincount(bin_ids, minlength=n_bins).to(pred_probs.dtype)      # [B]
    sum_conf = torch.zeros(n_bins, dtype=pred_probs.dtype, device=pred_probs.device)
    sum_acc  = torch.zeros_like(sum_conf)
    sum_conf.scatter_add_(0, bin_ids, conf)
    sum_acc.scatter_add_(0, bin_ids, correct)

    nonzero = counts > 0
    avg_conf = torch.zeros_like(sum_conf)
    avg_acc  = torch.zeros_like(sum_acc)
    avg_conf[nonzero] = sum_conf[nonzero] / counts[nonzero]
    avg_acc[nonzero]  = sum_acc[nonzero]  / counts[nonzero]

    ece = ((counts / N) * (avg_conf - avg_acc).abs()).sum()
    return float(ece)
@torch.no_grad()
def eval_step(
    ep: int,
    model,
    tokenizer,
    target_ids: torch.Tensor,
    val_loader,
    n_classes: int,
    device: torch.device,
    optimizer,
    logger,
    step_count: int,
    cfg: DictConfig,
) -> Tuple[float, float, float]:
    model.eval()

    n_mc = int(getattr(getattr(cfg, "eval", {}), "n_mc", 4))
    mc_use_train_mode = True

    acc_metric = Accuracy(task="multiclass", num_classes=n_classes).to(device)
    total_nll, total_batches = 0.0, 0
    probs_all, labels_all = [], []
    
    def mc_predict_proba(model, batch_inputs, target_ids, n_samples=10, use_train_mode=True):
        probs_sum = None
  
        with t.no_grad():
            for _ in range(n_samples):
     
              
                if use_train_mode:
                    model.train() 
                else:
                    model.eval()
                (model.module if hasattr(model, "module") else model).apply(
                    lambda m: m.reset_cache() if hasattr(m, "reset_cache") else None
                )
                outputs: ModelOutput = model(**batch_inputs)
                logits = outputs.logits[:, -1, target_ids]        # [B, C]
                probs = F.softmax(logits, dim=-1)                 # [B, C]
                probs_sum = probs if probs_sum is None else (probs_sum + probs)
        model.eval()
        return probs_sum / float(n_samples)

    pbar_val = tqdm(val_loader, desc=f"Eval ep{ep} (MC={n_mc})", leave=False, ascii=True)

    for batch in pbar_val:
    
        prompts, classes, _ = batch
        classes = classes.to(device)
        batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device)

        pred_probs = mc_predict_proba(
            model, batch_inputs, target_ids,
            n_samples=n_mc, use_train_mode=mc_use_train_mode
        )  # [B, C]

        # NLL on mixture
        gather_p = pred_probs.gather(1, classes.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        batch_nll = (-gather_p.log()).mean()

        acc_metric.update(pred_probs, classes)
        probs_all.append(pred_probs)
        labels_all.append(classes)

        total_nll += float(batch_nll.item())
        total_batches += 1

    val_nll = total_nll / max(1, total_batches)
    val_acc = float(acc_metric.compute().item())
    probs_all  = torch.cat(probs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    val_ece = simple_ece(probs_all, labels_all)

    logger.info(
        f"[EP {ep} | STEP {step_count}] LR: {optimizer.param_groups[0]['lr']:.6e} "
        f"NLL={val_nll:.4f} | ACC={val_acc:.4f} | ECE={val_ece:.4f} "
        f"(MC={n_mc}, train_mode={mc_use_train_mode})"
    )
    return float(val_ece), float(val_nll), float(val_acc)
        
@hydra.main(version_base="1.3", config_path="configs", config_name="example_usage_arc_c")
def main(cfg: DictConfig):
    os.makedirs("logs", exist_ok=True)
    device = "cuda:2" if t.cuda.is_available() else "cpu"
    logger = setup_logging(cfg)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
  
    logger.info(f"Using device: {device}")


    cfg.llm.use_quant = False
    cfg.llm.use_peft = True

    model, tokenizer, _ = setup_llm(**cfg.llm)
    bayesianize_(model, cfg.inducing)
    print(model)
    tparams, aparams = count_trainable_params(model)
    logger.info(f"[Inject] replaced Linear layers: {model}")
    logger.info(f"[Params] trainable={tparams/1e6:.2f}M / total={aparams/1e6:.2f}M")
    model = model.to(device)
    
    dset_class: dsets.ClassificationDataset = getattr(dsets, cfg.dset.name)
    dataset = dset_class(tokenizer, add_space=cfg.llm.add_space, name=cfg.dset.alias_name)

    train_loader = dataset.loader(
        is_sc=cfg.llm.is_sc,
        batch_size=cfg.dset.train_bs,
        split=cfg.dset.train_split,
        subset_size=cfg.dset.train_subset,
    )
    val_loader = dataset.loader(
        is_sc=cfg.llm.is_sc,
        batch_size=cfg.dset.eval_bs,
        split=cfg.dset.eval_split,
        subset_size=cfg.dset.eval_subset,
    )

    opt_cfg = dict(cfg.opt)
    opt_mod = importlib.import_module(opt_cfg.pop("module"))
    opt_cls = getattr(opt_mod, opt_cfg.pop("classname"))
    optimizer = opt_cls(model.parameters(), **opt_cfg)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 4], gamma=0.1)

    target_ids = dataset.target_ids.squeeze(-1)  # [num_labels]
    n_classes = int(dataset.n_labels)

    epochs = int(getattr(cfg, "epochs", 6))
    step_count = 0
    best_ece, best_nll, best_acc = 1e9, 1e9, 0.0
    last_metrics = (float("nan"), float("nan"), float("nan"))  # (ece, nll, acc)

    for ep in range(epochs):
        model.train()
        running_loss, step_in_ep = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Train ep{ep}", leave=False, ascii=True)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, 1):
            prompts, classes, _ = batch
          
            (model.module if hasattr(model, "module") else model).apply(
                lambda m: m.reset_cache() if hasattr(m, "reset_cache") else None
            )
            classes = classes.to(device)
            batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device)

            outputs: ModelOutput = model(**batch_inputs)
            logits = outputs.logits[:, -1, target_ids]  # [B, C]

            kl_sum = torch.tensor(0.0, device=device)
            if ep >= 2:
                for m in model.modules():
                    if hasattr(m, "kl_divergence"):
                        val = m.kl_divergence()
                        kl_sum = kl_sum + (val if isinstance(val, torch.Tensor)
                                           else torch.tensor(float(val), device=device))

            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            entropy = (-(probs * logp).sum(dim=-1)).mean()

            KL_scale = 1.0 / max(1, len(train_loader))  #
            loss = F.cross_entropy(logits, classes, label_smoothing=0.05) \
                   + 0.1 * KL_scale * kl_sum - 0.05 * entropy

            loss.backward()
            optimizer.step()
            step_count += 1
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            step_in_ep += 1
            pbar.set_postfix(loss=f"{running_loss/step_in_ep:.4f}")

        scheduler.step()

        last_metrics = eval_step(
            ep, model, tokenizer, target_ids, val_loader, n_classes,
            device, optimizer, logger, step_count, cfg
        )
        ece, nll, acc = last_metrics
        if ece < best_ece and nll < best_nll and acc > best_acc:
            best_ece = min(best_ece, ece)
            best_nll = min(best_nll, nll)
            best_acc = max(best_acc, acc)

    logger.info(
        f"[FINAL] lr={cfg.opt.lr:.3e}, wd={{:.3e}} ECE={best_ece:.4f} | NLL={best_nll:.4f} | ACC={best_acc:.4f}"
        .format(cfg.opt.weight_decay)
    )
   



if __name__ == "__main__":
    main()
