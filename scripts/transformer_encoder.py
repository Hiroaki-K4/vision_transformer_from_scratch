import torch.nn as nn

from multi_head_self_attention import MultiHeadSelfAttention
from multi_layer_perceptron import MultiLayerPerceptron


class TransformerEncoder(nn.Module):
    def __init__(
        self, embedding_dim=768, num_heads=12, hidden_dim=3072, dropout_prob=0.1
    ):
        super(TransformerEncoder, self).__init__()

        self.MSA = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.MLP = MultiLayerPerceptron(embedding_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        out_1 = self.dropout1(x)  # apply dropout
        out_2 = self.layer_norm1(out_1)  # apply layer normalization
        msa_out = self.MSA(out_2)  # compute multi-head self-attention
        out_3 = self.dropout2(msa_out)  # apply dropout
        res_out = x + out_3  # apply residual connection
        out_4 = self.layer_norm2(res_out)  # apply layer normalization
        mlp_out = self.MLP(out_4)  # compute mlp output
        out_5 = self.dropout3(mlp_out)  # apply dropout
        output = res_out + out_5  # apply residual connection

        return output
