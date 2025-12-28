import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(d_embed)
        self.Q = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.K = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.V = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.d_head = d_embed // n_heads
        self.n_heads = n_heads
        self.out_layer = nn.Linear(d_embed, d_embed, bias = out_proj_bias)

    def forward(self, x : torch.Tensor, causal_mask: False) -> torch.Tensor:
        # x: (Batch_size, Length, d_embed)
        n, T, C = x.shape
        residue = x
        # (Batch_size, Length, d_embed) -> (Batch_size, Length, d_embed)
        q = self.Q(x)
        # (Batch_size, Length, d_embed) -> (Batch_size, Length, d_embed)
        k = self.K(x)
        # (Batch_size, Length, d_embed) -> (Batch_size, Length, d_embed)
        v = self.V(x)
        # (Batch_size, Length, d_embed) -> (Batch_size, d_embed, Length)
        q = q.view(n, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(n, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(n, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.transpose(-1, -2)
        # (Batch_size, n_heads, Length, d_head) @ (Batch_size, n_heads, d_head, Length) -> (Batch_size, n_heads, Length, Length)
        weight = q @ k / math.sqrt(self.d_head)
        if causal_mask:
            mask = torch.ones_like(weight, dtype = torch.bool).triu(diagonal = 1)
            weight = weight.masked_fill_(mask, -torch.inf)
        
        # (Batch_size, n_heads, Length, Length) -> (Batch_size, n_heads, Length, Length)
        weight = F.softmax(weight, dim = -1)
        # (Batch_size, n_heads, Length, Length) @ (Batch_size, n_heads, Length, d_head) -> (Batch_size, n_heads, Length, d_head)
        x = weight @ v
        # (Batch_size, n_heads, Length, d_head) -> (Batch_size, Length, n_heads, d_head)
        x = x.transpose(1, 2).contiguous().view(n, T, C)
        # (Batch_size, Length, d_embed) -> (Batch_size, Length, d_embed)
        x = self.out_layer(x)
        return x