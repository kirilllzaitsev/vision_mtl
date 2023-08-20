import matplotlib.pyplot as plt
import numpy as np


def plot_segm_class(idx, mask):
    empty_canvas = np.zeros((*mask.shape, 3)).astype(np.uint8)
    empty_canvas[mask == idx, ...] = 255
    print(empty_canvas.max(), empty_canvas.min())
    plt.imshow(empty_canvas)
