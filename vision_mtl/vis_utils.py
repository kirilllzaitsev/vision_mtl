import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from vision_mtl.cfg import cfg


def plot_batch(batch: dict) -> plt.Figure:
    batch_size = len(batch["img"])
    ncols = 3
    fig, axs = plt.subplots(batch_size, 3, figsize=(batch_size * 5, ncols * 2))
    for i in range(batch_size):
        sample = {k: v[i] for k, v in batch.items()}
        sample_axs = plot_sample(sample, axs=axs[i] if batch_size > 1 else axs)
        if i != batch_size - 1:
            for ax in sample_axs:
                ax.axis("off")
    return fig


def plot_sample(x: dict, axs: t.Optional[np.ndarray] = None) -> np.ndarray:
    if axs is None:
        fig, axs = plt.subplots(1, len(x), figsize=(10, 10))
    for i, (k, v) in enumerate(x.items()):
        v = v.detach().cpu().squeeze()
        if v.dim() == 3:
            axs[i].imshow(v.permute(1, 2, 0))
        else:
            axs[i].imshow(v)
        axs[i].set_title(k)
        if i != 0:
            axs[i].axis("off")
    return axs


def plot_segm_class(idx: int, mask: np.ndarray) -> np.ndarray:
    empty_canvas = np.zeros((*mask.shape, 3)).astype(np.uint8)
    empty_canvas[mask == idx, ...] = 255
    print(empty_canvas.max(), empty_canvas.min())
    plt.imshow(empty_canvas)
    return empty_canvas


def plot_annotated_segm_mask(
    mask: np.ndarray,
    class_names: t.Optional[list] = None,
    img: t.Optional[t.Union[np.ndarray, torch.Tensor]] = None,
    alpha: float = 1.0,
    ax: t.Optional[plt.Axes] = None,
    fontsize: int = 10,
) -> plt.Axes:
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()
    if mask.ndim == 3:
        mask = mask.squeeze(axis=0)

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

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    if img is not None:
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        ax.imshow(img)
    ax.imshow(remap_class_idx_to_rgb(mask, colored_rgb_palette), alpha=alpha)
    ax.legend(handles, labels, fontsize=fontsize)
    return ax


def plot_annotated_segm_mask_v1(mask: np.ndarray, class_names: list) -> plt.Figure:
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
    return fig


def plot_preds(batch_size: int, inputs_batch: dict, preds_batch: dict) -> plt.Figure:
    ncols = 1 + 2 + 2
    fig, ax = plt.subplots(batch_size, ncols, figsize=(10, ncols * batch_size))
    for row_idx in range(batch_size):
        if batch_size == 1:
            ax_0 = ax[0]
            ax_1 = ax[1]
            ax_2 = ax[2]
            ax_3 = ax[3]
            ax_4 = ax[4]
        else:
            ax_0 = ax[row_idx, 0]
            ax_1 = ax[row_idx, 1]
            ax_2 = ax[row_idx, 2]
            ax_3 = ax[row_idx, 3]
            ax_4 = ax[row_idx, 4]

        img = inputs_batch["img"][row_idx]
        gt_depth = inputs_batch["depth"][row_idx]
        pred_depth = preds_batch["depth"][row_idx].detach()
        gt_segm = inputs_batch["mask"][row_idx]
        pred_segm = preds_batch["segm"][row_idx].detach()

        ax_0.imshow(img.squeeze().cpu().numpy().transpose(1, 2, 0))
        ax_1.imshow(gt_depth.squeeze().cpu().numpy())
        ax_2.imshow(pred_depth.squeeze().cpu().numpy())
        ax_3.imshow(gt_segm.squeeze().cpu().numpy())
        ax_4.imshow(pred_segm.squeeze().cpu().numpy())

        ax_0.set_title("Input image")
        ax_1.set_title("Ground truth depth")
        ax_2.set_title("Prediction depth")
        ax_3.set_title("Ground truth segm")
        ax_4.set_title("Prediction segm")

        for ax_ in [ax_1, ax_2, ax_3, ax_4]:
            ax_.axis("off")

        ax_0.set_ylabel(f"Sample {row_idx}")
    return fig


def convert_figure_to_image(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    return image
