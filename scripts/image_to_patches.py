import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns


def main(image, P):
    x = np.array(image)

    C = 3  # Number of channels (RGB)

    # Split image into patches using numpy
    patch_h_num, patch_w_num = x.shape[0] // P, x.shape[1] // P
    patches = (
        x.reshape(patch_h_num, P, patch_w_num, P, C).swapaxes(1, 2).reshape(-1, P, P, C)
    )

    # Flatten patches
    x_p = np.reshape(patches, (-1, P * P * C))

    # Get number of patches
    N = x_p.shape[0]

    print("Image shape:", x.shape)  # width, height, channel
    print("Number of patches: {0} with resolution ({1}, {2})".format(N, P, P))
    print("Patches shape:", patches.shape)
    print("Flattened patches shape:", x_p.shape)

    # Display image and patches side-by-side
    fig = plt.figure()
    gridspec = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gridspec[0])
    ax1.set(title="Image to Patches")
    ax1.imshow(x)

    subgridspec = gridspec[1].subgridspec(patch_h_num, patch_w_num, hspace=-0.8)

    # Display patches
    for i in range(patch_h_num):
        for j in range(patch_w_num):
            num = i * patch_h_num + j
            ax = fig.add_subplot(subgridspec[i, j])
            ax.set(xticks=[], yticks=[])
            ax.imshow(patches[num])
    plt.show()

    # Display first 10 flattened patches up to 25 values 
    heat_map = x_p[:10, :25]
    print(heat_map)
    yticklabels = ['patch ' + str(i + 1) for i in range(10)]
    plt.title('First 10 Flattened Patches')
    ax = sns.heatmap(heat_map,  
                    cmap=sns.light_palette("#79C", as_cmap=True),
                    xticklabels=False, yticklabels=yticklabels,
                    linewidths=0.01, linecolor='white'
                    )
    plt.show()


if __name__ == "__main__":
    image = Image.open("../resources/car.png").resize((128, 128))
    P = 16  # patch size
    main(image, P)
