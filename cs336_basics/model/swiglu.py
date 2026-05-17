import torch
import torch.nn as nn
from einops import einsum

class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1_weight = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.w2_weight = nn.Parameter(
            torch.empty((d_model, d_ff), device=device, dtype=dtype)
        )
        self.w3_weight = nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Using Kaiming uniform initialization, consistent with nn.Linear default
        nn.init.kaiming_uniform_(self.w1_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.w3_weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        x_up = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        # SwiGLU(x) = (SiLU(xW1) * xW3)W2
        swish_gate = x_gate * torch.sigmoid(x_gate)
        intermediate = swish_gate * x_up
        return einsum(intermediate, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
