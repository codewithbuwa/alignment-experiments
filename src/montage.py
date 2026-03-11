import math
import os

import matplotlib.pyplot as plt


def make_montage(paths, output_path, cols: int = 3, figsize=(12, 8)):
    if not paths:
        raise ValueError("No paths provided for montage")

    images = [plt.imread(p) for p in paths]
    rows = int(math.ceil(len(images) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if idx < len(images):
                ax.imshow(images[idx])
                idx += 1

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
