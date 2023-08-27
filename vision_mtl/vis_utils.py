import matplotlib.pyplot as plt
import numpy as np


def plot_segm_class(idx, mask):
    empty_canvas = np.zeros((*mask.shape, 3)).astype(np.uint8)
    empty_canvas[mask == idx, ...] = 255
    print(empty_canvas.max(), empty_canvas.min())
    plt.imshow(empty_canvas)


def plot_annotated_segm_mask_v1(mask, class_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask)
    for idx, class_name in [(-1, "artifact"), *enumerate(class_names)]:
        mask_class = mask == idx
        if mask_class.sum() == 0:
            continue
        y, x = np.where(mask_class)
        y = y.mean()
        x = x.mean()
        ax.text(x, y, class_name, fontsize=16, color="white")
    plt.show()
