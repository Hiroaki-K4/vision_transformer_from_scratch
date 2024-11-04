import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=10, fine_tune=False):
        super(MLPHead, self).__init__()
        self.num_classes = num_classes
        if not fine_tune:
            # hidden layer with tanh activation function
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, 3072), nn.Tanh(), nn.Linear(3072, num_classes)
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.mlp_head(x)
        return x
