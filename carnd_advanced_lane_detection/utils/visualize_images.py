import numpy as np
import matplotlib.pyplot as plt


def stack_images(first, second):
    # TODO: just copy-pasting for now
    np.dstack(( np.zeros_like(first), first, second))


def one_by_two_plot(first, second, first_cmap=None, second_cmap=None, first_title="First image", second_title="Second image"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title(first_title)
    if first_cmap is not None:
        ax1.imshow(first, cmap=first_cmap)
    else:
        ax1.imshow(first)

    ax2.set_title(second_title)
    if second_cmap is not None:
        ax2.imshow(second, cmap=second_cmap)
    else:
        ax2.imshow(second)
    plt.show()


def two_by_two_plot(first, second, third, fourth):
    pass

