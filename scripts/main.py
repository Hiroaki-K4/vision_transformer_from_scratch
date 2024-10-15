import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


def main():
    # unit test on patching an image

    # read image and resize to 128
    image = Image.open("resources/car.png").resize((128, 128))

    # convert to numpy array
    x = np.array(image)

    # An Image Is Worth 16x16 Words
    P = 16  # patch size
    C = 3  # number of channels (RGB)

    # split image into patches using numpy
    patches = (
        x.reshape(x.shape[0] // P, P, x.shape[1] // P, P, C)
        .swapaxes(1, 2)
        .reshape(-1, P, P, C)
    )

    # flatten patches
    x_p = np.reshape(patches, (-1, P * P * C))

    # get number of patches
    N = x_p.shape[0]

    print("Image shape: ", x.shape)  # width, height, channel
    print("Number of patches: {} with resolution ({}, {})".format(N, P, P))
    print("Patches shape: ", patches.shape)
    print("Flattened patches shape: ", x_p.shape)

    # visualize data
    #
    # display image and patches side-by-side

    fig = plt.figure()

    gridspec = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gridspec[0])
    ax1.set(title="Image")

    # display image
    ax1.imshow(x)

    subgridspec = gridspec[1].subgridspec(8, 8, hspace=-0.8)

    # display patches
    for i in range(8):  # N = 64, 8x8 grid
        for j in range(8):
            num = i * 8 + j
            ax = fig.add_subplot(subgridspec[i, j])
            ax.set(xticks=[], yticks=[])
            ax.imshow(patches[num])

    plt.show()

    # visualize data
    #
    # display flattened patches

    # display first 10 flattened patches up to 25 values
    heat_map = x_p[:10, :25]

    yticklabels = ["patch " + str(i + 1) for i in range(10)]

    plt.title("First 10 Flattened Patches")
    ax = sns.heatmap(
        heat_map,
        cmap=sns.light_palette("#a275ac", as_cmap=True),
        xticklabels=False,
        yticklabels=yticklabels,
        linewidths=0.01,
        linecolor="white",
    )
    plt.show()

    # unit test on patch embeddings

    # dimensionality of patch embeddings
    D = 768

    # batch size
    B = 1

    # convert flattened patches to tensor
    x_p = torch.Tensor(x_p)

    # add batch dimension
    x_p = x_p[None, ...]

    # weight matrix E
    E = nn.Parameter(torch.randn(1, P * P * C, D))

    patch_embeddings = torch.matmul(x_p, E)

    assert patch_embeddings.shape == (B, N, D)
    print(patch_embeddings.shape)

    # unit test on class token

    # init class token
    class_token = nn.Parameter(torch.randn(1, 1, D))

    patch_embeddings = torch.cat((class_token, patch_embeddings), 1)

    print(patch_embeddings.shape)
    assert patch_embeddings.shape == (B, N + 1, D)

    # unit test on position embedddings

    # position embeddings
    E_pos = nn.Parameter(torch.randn(1, N + 1, D))

    z0 = patch_embeddings + E_pos

    print(z0.shape)
    assert z0.shape == (B, N + 1, D)


if __name__ == "__main__":
    main()
