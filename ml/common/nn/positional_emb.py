import torch.nn as nn
import torch
import math

class TimeEmbedding(nn.Module):
    """
    Stanndard sinusuidal embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = torch.log(t).unsqueeze(1) * embeddings.unsqueeze(0)  # Use log(t)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
