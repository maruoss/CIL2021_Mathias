# Metric functions
def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def patch_accuracy(y_hat, y, patch_size=16, cutoff=0.25):
    # computes accuracy weighted by patches
    h_patches = y.shape[-2] // patch_size # Number of patches in the height
    w_patches = y.shape[-1] // patch_size

    # Reshape to patches x patchsize to take mean across patchsize 
    patches_hat = y_hat.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff
    patches = y.reshape(-1, 1, h_patches, patch_size, w_patches, patch_size).mean((-1, -3)) > cutoff

    return (patches_hat == patches).float().mean()
