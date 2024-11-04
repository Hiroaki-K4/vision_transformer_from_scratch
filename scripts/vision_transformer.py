import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from mlp_head import MLPHead
from transformer_encoder import TransformerEncoder


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size=16,
        image_size=224,
        channel_size=3,
        num_layers=12,
        embedding_dim=768,
        num_heads=12,
        hidden_dim=3072,
        dropout_prob=0.1,
        num_classes=10,
        pretrain=True,
        D=768,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.channel_size = channel_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.pretrain = pretrain
        self.D = D

        # Get number of patches of the image
        self.num_patches = int(image_size / patch_size) ** 2
        # trainable linear projection for mapping dimnesion of patches (weight matrix E)
        self.E = nn.Parameter(
            torch.randn(patch_size * patch_size * channel_size, embedding_dim)
        )

        # position embeddings (E_pos)
        self.pos_embedding = nn.Parameter(
            torch.randn(self.num_patches + 1, embedding_dim)
        )

        # learnable class token embedding (x_class)
        self.class_token = nn.Parameter(torch.rand(1, self.D))

        # stack transformer encoder layers
        transformer_encoder_list = [
            TransformerEncoder(embedding_dim, num_heads, hidden_dim, dropout_prob)
            for _ in range(num_layers)
        ]
        self.transformer_encoder_layers = nn.Sequential(*transformer_encoder_list)

        # mlp head
        self.mlp_head = MLPHead(embedding_dim, num_classes)

    def forward(self, x):
        # get patch size and channel size
        P, C = self.patch_size, self.channel_size

        # split image into patches
        patches = x.unfold(0, C, C).unfold(1, P, P).unfold(2, P, P)
        patches = patches.contiguous().view(patches.size(0), -1, C * P * P).float()

        # linearly embed patches
        patch_embeddings = torch.matmul(patches, self.E)

        # add class token
        batch_size = patch_embeddings.shape[0]
        patch_embeddings = torch.cat(
            (self.class_token.repeat(batch_size, 1, 1), patch_embeddings), 1
        )

        # add positional embedding
        patch_embeddings = patch_embeddings + self.pos_embedding

        # feed patch embeddings into a stack of Transformer encoders
        transformer_encoder_output = self.transformer_encoder_layers(patch_embeddings)

        # extract [class] token from encoder output
        output_class_token = transformer_encoder_output[:, 0]

        # pass token through mlp head for classification
        y = self.mlp_head(output_class_token)

        return y


def main():
    image_size = 224
    channel_size = 3
    n_class = 10  # number of classes CIFAR-10
    dropout_prob = 0.1
    # Vit-base model configurations
    n_layer = 12
    embedding_dim = 768
    n_head = 12
    hidden_dim = 3072
    image = Image.open("../resources/car.png").resize((image_size, image_size))
    X = T.PILToTensor()(image)  # Shape [channel_size, image_size, image_size]
    patch_size = 16
    patches = (
        X.unfold(0, channel_size, channel_size)
        .unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
    )  # Shape [1, image_size/patch_size, image_size/patch_size, channel_size, patch_size, patch_size]
    patches = (
        patches.contiguous()
        .view(patches.size(0), -1, channel_size * patch_size * patch_size)
        .float()
    )  # Shape [1, Number of patches, channel_size*patch_size*patch_size]
    # init vision transformer model
    vision_transformer = VisionTransformer(
        patch_size,
        image_size,
        channel_size,
        n_layer,
        embedding_dim,
        n_head,
        hidden_dim,
        dropout_prob,
        n_class,
    )

    # compute vision transformer output
    vit_output = vision_transformer(X)

    assert vit_output.size(dim=1) == n_class

    # get class probabilities
    probabilities = F.softmax(vit_output[0], dim=0)
    print("Class probabilities: ", probabilities)


if __name__ == "__main__":
    main()
