from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False).float()
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False).float()

        # Initialize LoRA weights. Scaling by 1/lora_dim helps stabilize training
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5, nonlinearity="leaky_relu")
        self.lora_b.weight.data.zero_()

        # Freeze original weights
        for param in self.parameters():
            param.requires_grad = False  # Freeze everything initially

        # Unfreeze LoRA layers
        for param in self.lora_a.parameters():
            param.requires_grad = True
        for param in self.lora_b.parameters():
            param.requires_grad = True

        # raise NotImplementedError()

        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"Input dtype: {x.dtype}")
        
        lora_out = self.lora_b(self.lora_a(x.to(torch.float32))).to(self.linear_dtype)
        print(f"LoRA output dtype: {lora_out.dtype}")
        
        base_out = super().forward(x)
        print(f"Base output dtype: {base_out.dtype}")

        final_out = (base_out.to(self.linear_dtype) + lora_out).to(x.dtype)

        print(f"Final output dtype: {final_out.dtype}")

        return final_out


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            # raise NotImplementedError()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        # raise NotImplementedError()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
