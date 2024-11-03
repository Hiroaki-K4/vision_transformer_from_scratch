import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim=768, key_dim=64):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim  # D -> embedding dimensionaltiy
        self.key_dim = key_dim  # D_h -> key, query, value dimensionality
        # U_kqv weight matrix
        self.W = nn.Parameter(torch.randn(embedding_dim, 3 * key_dim))

    def forward(self, x):
        key_dim = self.key_dim
        # get query, key and value projection
        qkv = torch.matmul(x, self.W)
        # get query, key, value
        q = qkv[:, :, :key_dim]
        k = qkv[:, :, key_dim : key_dim * 2]
        v = qkv[:, :, key_dim * 2 :]
        # compute dot product of the all query with all keys
        k_T = torch.transpose(k, -2, -1)  # get transpose of key
        dot_products = torch.matmul(q, k_T)
        # divide each by √Dh
        scaled_dot_products = dot_products / np.sqrt(key_dim)
        # apply a softmax function to obtain attention weights -> A
        attention_weights = F.softmax(scaled_dot_products, dim=1)
        # self.attention_weights = [w.detach().numpy() for w in attention_weights]
        # get weighted values
        weighted_values = torch.matmul(attention_weights, v)

        return weighted_values
