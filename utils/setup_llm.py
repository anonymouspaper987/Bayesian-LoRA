# Copyright (C) 2023-24 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Boilerplate for setting up a HuggingFace LLM.
"""

import logging
import torch as t
import transformers

from typing import Optional
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import BitsAndBytesConfig, GenerationConfig
from transformers.utils import is_flash_attn_2_available

# Avoid importing this globally for systems where peft is not installed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def str_to_torch_dtype(name: str) -> t.dtype:
    dt = t.__dict__[name]
    assert isinstance(dt, t.dtype)
    return dt
@torch.no_grad()
def compute_ece(
    probs_or_logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
    logits: bool = False,
    norm: str = "l1",
    ignore_index: int | None = None,
    return_bins: bool = False,
) -> tuple[torch.Tensor, dict] | torch.Tensor:
  
    assert probs_or_logits.ndim == 2, "probs_or_logits 应为 [N, C]"
    N, C = probs_or_logits.shape
    device = probs_or_logits.device
    dtype = probs_or_logits.dtype

    if logits:
        probs = torch.softmax(probs_or_logits, dim=-1)
    else:
        probs = probs_or_logits
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    conf, pred = probs.max(dim=-1)  # [N], [N]

    mask = torch.ones_like(targets, dtype=torch.bool, device=device)
    if ignore_index is not None:
        mask &= (targets != ignore_index)

    if mask.sum() == 0:
        empty = torch.tensor(0.0, device=device, dtype=dtype)
        return (empty, {
            "bin_centers": torch.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins, device=device, dtype=dtype),
            "accuracy": torch.zeros(n_bins, device=device, dtype=dtype),
            "confidence": torch.zeros(n_bins, device=device, dtype=dtype),
            "proportion": torch.zeros(n_bins, device=device, dtype=dtype),
            "counts": torch.zeros(n_bins, device=device, dtype=torch.long),
        }) if return_bins else empty

    conf = conf[mask]
    pred = pred[mask]
    y = targets[mask]

   
    bin_idx = torch.clamp((conf * n_bins).long(), 0, n_bins - 1)  # [M]

    counts = torch.bincount(bin_idx, minlength=n_bins).to(dtype)  # [B]

    sum_conf = torch.zeros(n_bins, device=device, dtype=dtype)
    sum_conf.scatter_add_(0, bin_idx, conf.to(dtype))

    correct = (pred == y).to(dtype)
    sum_acc = torch.zeros(n_bins, device=device, dtype=dtype)
    sum_acc.scatter_add_(0, bin_idx, correct)

    denom = counts.clamp_min(1.0)
    avg_conf = sum_conf / denom
    avg_acc = sum_acc / denom

    prop = counts / counts.sum().clamp_min(1.0)

    if norm.lower() == "l2":
        gap = (avg_conf - avg_acc).pow(2)
    else:  # "l1"
        gap = (avg_conf - avg_acc).abs()

    ece = (prop * gap).sum()

    if return_bins:
        bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=device, dtype=dtype)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return ece, {
            "bin_centers": bin_centers,        # [B]
            "accuracy": avg_acc,               # [B]
            "confidence": avg_conf,            # [B]
            "proportion": prop,                # [B]
            "counts": counts.to(torch.long),   # [B]
        }
    else:
        return ece

def setup_model_kwargs(
    model_kwargs: dict = dict(),
    use_quant: bool = False,
    quantization: Optional[BitsAndBytesConfig] = None,
) -> dict:
    """
    - Gets the config from hydra
    - Converts dtype strings to torch dtypes
    - Adds quantization configurations from hydra
    """
    try:
        model_kwargs = OmegaConf.to_object(model_kwargs)
    except Exception:
        pass
    assert isinstance(model_kwargs, dict)
    for k, v in model_kwargs.items():
        if "dtype" in k.lower() and v != "auto":
            model_kwargs[k] = str_to_torch_dtype(v)
        if "attn_implementation" in k.lower():
            if v == "flash_attention_2":
                model_kwargs[k] = (
                    "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
                )
    if use_quant and quantization is not None:
        model_kwargs["quantization_config"] = instantiate(quantization)
    return model_kwargs


def setup_llm(
    model_name_or_path: str,
    config_class: str = "AutoConfig",
    config_kwargs: dict = dict(),
    tokenizer_class: str = "AutoTokenizer",
    tokenizer_kwargs: dict = dict(),
    tokenizer_special_tokens: dict = dict(),
    model_class: str = "AutoModelForCausalLM",
    model_kwargs: dict = dict(),
    global_gen_kwargs: dict = dict(),
    use_peft: bool = False,
    peft: Optional[LoraConfig] = None,
    use_quant: bool = False,
    quantization: Optional[BitsAndBytesConfig] = None,
    **_kwargs,
):
    """
    A simple function to wrap all the HuggingFace boilerplate.
    This loads the model configuration, the model itself, apply any
    BitsAndBytes quantization configuration, and PEFT configuration, and return
    the prepared model, tokenizer and generation config.
    """
    # Load the HF model config
    config_cls = getattr(transformers, config_class)
    if not isinstance(config_kwargs, dict):
        config_kwargs = OmegaConf.to_object(config_kwargs)
    model_config = config_cls.from_pretrained(model_name_or_path, **config_kwargs)

    # Load the HF model
    model_cls = getattr(transformers, model_class)
    model_kwargs = setup_model_kwargs(model_kwargs, use_quant, quantization)
    model = model_cls.from_pretrained(
        model_name_or_path, config=model_config, **model_kwargs
    )
    if use_quant and quantization is not None:
        model = prepare_model_for_kbit_training(model)

    # Configure PEFT if required
    if use_peft and peft is not None:
        logging.info("Setting up PEFT")
        peft_cfg = instantiate(peft)
        peft_cfg.target_modules = OmegaConf.to_object(peft_cfg.target_modules)

        # model.add_adapter(peft_cfg)
        # model.enable_adapters()
        # model.train()

        # NOTE: this manner of setting up the configuration seems to cause
        # issues when saving with `save_pretrained`... Investigate.
        # # peft_cfg = OmegaConf.to_container(peft_cfg, resolve=True)
        model = get_peft_model(model, peft_cfg)
        # print(model)
        # input()
        

    # Load the HF tokenizer
    tokenizer_cls = getattr(transformers, tokenizer_class)
    if not isinstance(tokenizer_kwargs, dict):
        tokenizer_kwargs = OmegaConf.to_object(tokenizer_kwargs)
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    tokenizer_special_tokens = {
        k: (
            getattr(tokenizer, v.split(".")[-1])
            if isinstance(v, str) and v.startswith("tokenizer")
            else v
        )
        for k, v in tokenizer_special_tokens.items()
    }
    if len(tokenizer_special_tokens) > 0:
        tokenizer.add_special_tokens(tokenizer_special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the global genration config
    if not isinstance(global_gen_kwargs, dict):
        global_gen_kwargs = OmegaConf.to_object(global_gen_kwargs)
    gen_cfg = GenerationConfig.from_pretrained(model_name_or_path, **global_gen_kwargs)

    return model, tokenizer, gen_cfg