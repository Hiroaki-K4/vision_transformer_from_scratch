class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads  # set number of heads (k)
        self.embedding_dim = embedding_dim  # set dimensionality
        assert (
            embedding_dim % num_heads == 0
        )  # dimensionality should be divisible by number of heads
        self.key_dim = embedding_dim // n_head  # set key,query and value dimensionality
        # init self-attentions
        self.attention_list = [
            SelfAttention(embedding_dim, self.key_dim) for _ in range(num_heads)
        ]
        self.multi_head_attention = nn.ModuleList(self.attention_list)
        # init U_msa weight matrix
        self.W = nn.Parameter(torch.randn(num_heads * self.key_dim, embedding_dim))

    def forward(self, x):
        # compute self-attention scores of each head
        attention_scores = [attention(x) for attention in self.multi_head_attention]
        # concat attentions
        Z = torch.cat(attention_scores, -1)
        # compute multi-head attention score
        attention_score = torch.matmul(Z, self.W)

        return attention_score
