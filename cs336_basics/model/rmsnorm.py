import torch
import torch.nn as nn

# Root Mean Square Layer Normalization
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 定义可学习的增益参数 gain
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.gain = nn.Parameter(torch.empty(d_model, **factory_kwargs))
        
        # 初始化 gain 为全 1
        # 这样在训练刚开始时，RMSNorm 只做归一化，而不对信号做额外的缩放。
        nn.init.ones_(self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录原始类型，用于最后转回
        in_dtype = x.dtype
        
        # 为了计算精确和数值稳定（防止平方后数值溢出），先转成 float32
        x_fp32 = x.to(torch.float32)

        # 1. 计算均方值 (Mean Square)：对最后一个维度求平方的平均值
        ms = x_fp32.pow(2).mean(dim=-1, keepdim=True)

        # 2. 计算均方根 (Root Mean Square) 并加上 eps 防止除以 0
        rms = torch.sqrt(ms + self.eps)

        # 3. 归一化并乘以可学习参数 gain
        x_normed = x_fp32 / rms
        result = x_normed * self.gain

        # 4. 转回原始的数据类型
        return result.to(in_dtype)
