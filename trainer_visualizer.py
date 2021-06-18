import numpy as np
import matplotlib.pyplot as plt


def show_val_samples(x, y, y_hat):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:  # segmentation
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        for i in range(imgs_to_draw):
            # print(np.unique(x)[:5], np.unique(y)[:5], np.unique(y_hat)[:5]) x has values from ~[-2, +2], y_hat [-0.5, +0.5], y [0, 1]
            # Image plot
            axs[0, i].imshow(np.moveaxis(x[i], 0, -1)) # Yields clipping warning, as deeplabv3 has input floats of <0 and >1

            # Target plots
            # axs[1, i].imshow(np.moveaxis(y_hat[i], 0, -1), cmap="gray") # Equal now. Yields too much "white" somehow -> because of no sigmoid!
            # axs[2, i].imshow(np.moveaxis(y[i], 0, -1), cmap="gray") # Equal now. Yielded too much "white" somehow -> because of no sigmoid!
            axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1)) # No warning now. yielded clipping warning -> because of no sigmoid func. in predictions!
            axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)]*3, -1))

            axs[0, i].set_title(f'Sample {i}')
            axs[1, i].set_title(f'Predicted {i}')
            axs[2, i].set_title(f'True {i}')
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
            axs[2, i].set_axis_off()
            
    plt.show()