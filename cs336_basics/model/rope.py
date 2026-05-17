import torch
import torch.nn as nn
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # (d/2,)
        freq_seq = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (freq_seq / d_k))

        # (max_seq_len, d/2)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, inv_freq)

        # precompute sin/cos
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # register buffer (not trainable)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def rotate_half(self, x):
        # x: (..., seq, d)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """

        *batch_shape, seq_len, d = x.shape
        # assert d == self.d_k

        # (...., seq_len, d/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # expand to match x: (..., seq_len, d)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        # apply RoPE
        x_rot = x * cos + self.rotate_half(x) * sin

        return x_rot
