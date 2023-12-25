import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from vision_mtl.cfg import cfg


def plot_batch(batch):
    batch_size = len(batch["img"])
    ncols = 3
    fig, axs = plt.subplots(batch_size, 3, figsize=(batch_size * 5, ncols*2))
    for i in range(batch_size):
        sample = {k: v[i] for k, v in batch.items()}
        plot_sample(sample, axs=axs[i] if batch_size > 1 else axs)
    plt.show()


def plot_sample(x, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, len(x), figsize=(10, 10))
    for i, (k, v) in enumerate(x.items()):
        if v.dim() == 3:
            axs[i].imshow(v.permute(1, 2, 0))
        else:
            axs[i].imshow(v)
        axs[i].set_title(k)
        if i != 0:
            axs[i].axis("off")
    return axs


def plot_segm_class(idx, mask):
    empty_canvas = np.zeros((*mask.shape, 3)).astype(np.uint8)
    empty_canvas[mask == idx, ...] = 255
    print(empty_canvas.max(), empty_canvas.min())
    plt.imshow(empty_canvas)


def plot_annotated_segm_mask(mask, class_names=None, img=None, alpha=1.0):
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()

    def remap_class_idx_to_rgb(mask, palette):
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for idx, class_idx in enumerate(np.unique(mask)):
            rgb_mask[mask == class_idx, ...] = palette[class_idx]
        return rgb_mask

    colored_rgb_palette = cfg.vis.rgb_palette

    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(colored_rgb_palette))]

    legend_data = [(colored_rgb_palette[k], class_names[k]) for k in np.unique(mask)]

    handles = [
        Rectangle((0, 0), 1, 1, color=[v / 255 for v in c]) for c, _ in legend_data
    ]
    labels = [n for _, n in legend_data]

    fig, ax = plt.subplots(figsize=(10, 10))
    if img is not None:
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        ax.imshow(img)
    plt.imshow(remap_class_idx_to_rgb(mask, colored_rgb_palette), alpha=alpha)
    plt.legend(handles, labels)
    plt.show()


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
