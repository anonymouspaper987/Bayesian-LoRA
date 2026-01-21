import torch
import torch.nn as nn
from bnn.nn.modules import InducingLinear
from typing import List, Tuple

class InducingAdapter(nn.Module):
    def __init__(self, base_linear: nn.Linear, inducing_cls, adapter_scale: float = 1.0, **inducing_kwargs):
        super().__init__()
        self.base_linear = base_linear
        self.adapter_scale = nn.Parameter(torch.tensor(0, dtype=self.base_linear.weight.dtype))
        for p in self.base_linear.parameters():
            p.requires_grad = False
        self.inducing_layer = inducing_cls(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            bias=base_linear.bias is not None,
            **inducing_kwargs
        )
        dev = next(self.base_linear.parameters()).device
        self.inducing_layer.to(device=dev, dtype=torch.float32)  # fp32
        if base_linear.bias is not None and hasattr(self.inducing_layer, "bias"):
            with torch.no_grad():
                try: self.inducing_layer.bias.copy_(base_linear.bias)
                except Exception: pass

    def forward(self, x):
        y_base = self.base_linear(x)                              # bf16/fp16
        ind_in = x.to(torch.float32)                              #  fp32
        y_ind  = self.inducing_layer(ind_in).to(y_base.dtype)     #  dtype
        return y_base + self.adapter_scale * y_ind

    def reset_cache(self):
        if hasattr(self.inducing_layer, "reset_cache"):
            self.inducing_layer.reset_cache()

    def kl_divergence(self):
        if hasattr(self.inducing_layer, "kl_divergence"):
            return self.inducing_layer.kl_divergence()
        p = next(self.base_linear.parameters())
        return torch.zeros((), device=p.device, dtype=p.dtype)


def inject_inducing_adapters(model: nn.Module,
                             target_modules: List[str],
                             inducing_cls,
                             adapter_scale: float = 1.0,
                             **inducing_kwargs) -> int:
    """
    """
    replaced = 0

    def _name_match(name: str) -> bool:
        return any(t in name for t in target_modules)

    def _should_wrap(full_name: str, child: nn.Module) -> bool:
        if "embed_tokens" in full_name or "lm_head" in full_name:
            return False
        if isinstance(child, (InducingAdapter, inducing_cls)):
            return False
        return isinstance(child, nn.Linear) and _name_match(full_name)

    def _recurse(mod: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(mod.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (InducingAdapter, inducing_cls)):
                continue
            if _should_wrap(full, child):
                new_mod = InducingAdapter(
                    base_linear=child,
                    inducing_cls=inducing_cls,
                    adapter_scale=adapter_scale,
                    **inducing_kwargs
                )
                setattr(mod, name, new_mod)
                replaced += 1
                continue
            _recurse(child, full)

    _recurse(model)
    return replaced
