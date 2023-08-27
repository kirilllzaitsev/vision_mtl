
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from vision_mtl.cfg import cfg


def plot_segm_class(idx, mask):
    empty_canvas = np.zeros((*mask.shape, 3)).astype(np.uint8)
    empty_canvas[mask == idx, ...] = 255
    print(empty_canvas.max(), empty_canvas.min())
    plt.imshow(empty_canvas)


def plot_annotated_segm_mask(mask, class_names, img=None, alpha=1.0):
    def remap_class_idx_to_rgb(mask, palette):
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for idx, class_idx in enumerate(np.unique(mask)):
            rgb_mask[mask == class_idx, ...] = palette[class_idx]
        return rgb_mask

    colored_rgb_palette = cfg.vis.rgb_palette

    legend_data = [(colored_rgb_palette[k], class_names[k]) for k in np.unique(mask)]

    handles = [
        Rectangle((0, 0), 1, 1, color=[v / 255 for v in c]) for c, _ in legend_data
    ]
    labels = [n for _, n in legend_data]

    fig, ax = plt.subplots(figsize=(10, 10))
    if img is not None:
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
