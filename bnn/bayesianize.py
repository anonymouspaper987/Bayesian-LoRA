import torch as t
import torch.nn as nn

from .nn.modules import (
    InducingLinear,
    InducingConv1d,
    InducingConv2d,
    InducingConv3d,
)

def _force_fp32_io(module: nn.Module):
    def pre_hook(mod, inputs):
        (x,) = inputs
        mod._ext_dtype = x.dtype
        return (x.to(t.float32),)
    def post_hook(mod, inputs, out):
        return out.to(getattr(mod, "_ext_dtype", out.dtype))
    module.register_forward_pre_hook(pre_hook, with_kwargs=False)
    module.register_forward_hook(post_hook, with_kwargs=False)
    return module

def _name_matches_for_keys(full_name: str, keys) -> bool:
    return any(k in full_name for k in keys) if keys else True

def bayesianize_(model: nn.Module, config: dict, prefix: str = ""):
   
    per_layer_cfg = config.get("inference", {})
    global_keys = [
        "whitened_u", "q_inducing", "learn_lamda", "init_lamda",
        "max_lamda", "max_sd_u", "cache_cholesky",
        "prior_sd", "sqrt_width_scaling", "inducing_rows", "inducing_cols"
    ]
    global_cfg = {k: config[k] for k in global_keys if k in config}
    keys = config.get("key_layers", [])  # ["self_attn.q_proj","self_attn.k_proj","lm_head"]

    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        bayesianize_(child, config, prefix=full_name)

        is_target_leaf = (
            (full_name.endswith("lora_A.default") or full_name.endswith("lora_B.default")) and
            isinstance(child, nn.Linear) and
            _name_matches_for_keys(full_name, keys)
        )
        if not is_target_leaf:
            continue

        ctor_args = [child.in_features, child.out_features]
        ctor_kwargs = {"bias": (child.bias is not None)}  
        ctor_kwargs.update(global_cfg)
        new_layer = InducingLinear(*ctor_args, **ctor_kwargs)

        ref_w = child.weight
        new_layer = new_layer.to(device=ref_w.device, dtype=ref_w.dtype)
        _force_fp32_io(new_layer)  

        parent = model  #
        if isinstance(parent, nn.ModuleDict):
            parent[name] = new_layer 
        else:
            setattr(parent, name, new_layer)

        print(f"[Bayes-LoRA] replaced {full_name} with InducingLinear")
