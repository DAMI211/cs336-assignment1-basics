import torch
from einops import rearrange, einsum

# case 1
# Example shapes
# D: (batch, sequence, d_in)
# A: (d_out, d_in)
batch = 4
sequence = 5
d_in = 3
d_out = 2

# Random tensors
D = torch.randn(batch, sequence, d_in)
A = torch.randn(d_out, d_in)

## Basic implementation
# Y = D @ A.T
# 这里用传统矩阵乘法可能很难理解维度
# D: (batch, sequence, d_in)
# A.T: (d_in, d_out)
# Y: (batch, sequence, d_out)
Y = D @ A.T
print("Y shape using @:", Y.shape)

## Einsum is self-documenting and robust
# D A -> Y
Y_einsum = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
print("Y shape using einsum:", Y_einsum.shape)

## Or, a batched version where D can have any leading dimensions but A is constrained.
# Using ... to represent any number of leading batch dimensions
Y_batched = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
print("Y shape using batched einsum:", Y_batched.shape)


# case 2
images = torch.randn(64, 128, 128, 3)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

# method 1
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
imgae_rearr = rearrange(images, "b h w c -> b 1 h w c")
dimmed_images = imgae_rearr * dim_value
print("dimmed images shape: ", dimmed_images.shape)

# method 2
dimmed_images = einsum(
    images, dim_by,
    "b h w c, d -> b d h w c"
)
print("dimmed images shape: ", dimmed_images.shape)

# case 3
channels_last = torch.randn(64, 32, 32, 3) # (b, h, w, c)
B = torch.randn(32*32, 32*32) # (out, in)

# method 1
## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
-1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
) # (b, p, c)
channels_first_flat = channels_last_flat.transpose(1, 2) # (b, c, p)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)