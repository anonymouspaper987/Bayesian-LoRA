import torch
from bnn.nn.modules import BayesianLinear2
from peft.tuners.lora.layer import Linear
import json

class BayesianLoRALayer(Linear):
    def __init__(self, *args, bayesian_config=None, **kwargs):

        with open('./inducing.json', 'r') as f:
            self.bayesian_config = json.load(f)
        
        super().__init__(*args, **kwargs)

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora=False,
        use_dora=False,
        use_alora=False,
        use_qalora=False,
        lora_bias=False,
        arrow_config=None,
        qalora_group_size=32,
        inference_mode=False,
        **kwargs,
    ):
        super().update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights=False,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_alora=use_alora,
            use_qalora=use_qalora,
            lora_bias=lora_bias,
            arrow_config=arrow_config,
            qalora_group_size=qalora_group_size,
            inference_mode=inference_mode,
            **kwargs,
        )

        A = self.lora_A[adapter_name]
        B = self.lora_B[adapter_name]

        in_features = A.in_features
        out_features = B.out_features

  
        self.lora_A[adapter_name] = BayesianLinear2(
            in_features=in_features,
            out_features=r,
            bias=self.lora_bias[adapter_name],
            **self.bayesian_config
        )

        self.lora_B[adapter_name] = BayesianLinear2(
            in_features=r,
            out_features=out_features,
            bias=self.lora_bias[adapter_name],
            **self.bayesian_config
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        # ========= base forward (FP16 allowed) =========
        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype

        if self.disable_adapters:
            return result

        # ========= LoRA Bayesian A/B forward (forced FP32) =========
        for adapter_name in self.active_adapters:
            if adapter_name not in self.lora_A:
                continue

            # cast input to FP32 for BayesianLinear (important!)
            x32 = x.float()

            # dropout still in FP32
            x32_drop = self.lora_dropout[adapter_name](x32)

            # A(x) sample weight in FP32
            Ax = self.lora_A[adapter_name](x32_drop)  # BayesianLinear -> FP32 ops

            # B(Ax) sample weight in FP32
            BAx = self.lora_B[adapter_name](Ax)

            # scale
            scaled32 = BAx * self.scaling[adapter_name]

            # convert result back to the model dtype (fp16/bf16)
            result = result + scaled32.to(result_dtype)

        return result


    # =============== merge/unmerge (posterior mean) ===================

    def get_delta_weight(self, adapter_name):
        A = self.lora_A[adapter_name].mean_weight()
        B = self.lora_B[adapter_name].mean_weight()
        scale = self.scaling[adapter_name]
        return (B @ A) * scale

    def merge(self, safe_merge=False, adapter_names=None):
        if adapter_names is None:
            adapter_names = self.active_adapters

        base = self.get_base_layer()
        for name in adapter_names:
            delta = self.get_delta_weight(name)
            if delta.device != base.weight.device:
                delta = delta.to(base.weight.device)
            base.weight.data += delta

    def unmerge(self, adapter_names=None):
        if adapter_names is None:
            adapter_names = self.active_adapters

        base = self.get_base_layer()
        for name in adapter_names:
            delta = self.get_delta_weight(name).to(base.weight.dtype)
            base.weight.data -= delta
