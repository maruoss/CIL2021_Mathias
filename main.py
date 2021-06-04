# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import os

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
import time

from sklearn.model_selection import train_test_split


# %%
# Define random seeds
random_seed = 343
torch.manual_seed(17)

# Define root to current working directory + "/Data"
root = Path.cwd() / 'Data'
print("Your current working directory + '/ Data' path: \n" + str(root))


# %%

# Get path names in a list
def get_filenames_of_path(path: Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# %%
# Apply function to get list of pathnames
image_paths = get_filenames_of_path(root / "training" / "training" / "images")
groundtruth_paths = get_filenames_of_path(root / "training" / "training" / "groundtruth")

# %%
# Image Transform function
transform_fn = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Image Transform function
target_transform_fn = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])





# %%
# Split data into training validation set
train_size = 0.9

train_image_paths, val_image_paths, train_groundtruth_paths, val_groundtruth_paths = train_test_split(
    image_paths, groundtruth_paths, random_state=random_seed, train_size=train_size, shuffle=True)

# Assert that images and groundtruth picture numbers coincide
assert [y[-7:] for y in [str(x) for x in train_image_paths]] == [y[-7:] for y in [str(x) for x in train_groundtruth_paths]]
assert [y[-7:] for y in [str(x) for x in val_image_paths]] == [y[-7:] for y in [str(x) for x in val_groundtruth_paths]]


# %%
# Create custom Dataset class to load one sample of tuple (image, groundtruth)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, groundtruth_paths: list, device: str, transforms=None, target_transforms=None):
        # Initialize imagepaths, grountruthpaths and transform
        self.image_paths = image_paths
        self.groundtruth_paths = groundtruth_paths
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.device = device

    def __getitem__(self, idx: int):

        # Load image and groundtruth: sample path, open via PIL and convert to ndarray, float32 for tensors
        im = np.array(Image.open(self.image_paths[idx]))
        gt = np.array(Image.open(self.groundtruth_paths[idx]))

        # Move axis: convert HWC into CHW, PyTorch needs C, H, W
        # im = np.moveaxis(im, -1, 0) # not needed with ToTensor()

        # Transform to tensor, attach to device
        if self.transforms:
            im = self.transforms(im).to(self.device)
        # else:
        #     im = torch.from_numpy(im).to(self.device)

        if self.target_transforms:
            gt = self.target_transforms(gt).to(self.device)
        # else:
        #     gt = torch.from_numpy(gt).to(self.device)

        return im, gt

    def __len__(self):
        return len(self.image_paths)


# %%

# Define device "cuda" for GPU, or "cpu" for CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate Datasets for train and validation
train_dataset = CustomDataset(train_image_paths, train_groundtruth_paths, device, transform_fn, target_transform_fn)
val_dataset = CustomDataset(val_image_paths, val_groundtruth_paths, device, transform_fn, target_transform_fn)

# Instantiate Loaders for these datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)


# %%

# Display image and labels
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = torch.moveaxis(train_features[0], 0, -1) #Select first image out of batch, reshape CHW to HWC
label = train_labels[0] # Select first label out of shape BHW
# Show image, .cpu() otherwise imshow doesnt work with cuda
plt.imshow(img.cpu())
plt.show()
# Squeeze only if target_transform is provided -> transforms int into torch.float [0, 1] range, and adds channel
label = label.squeeze()
# Show groundtruth in greyscale (original interpretation)
plt.imshow(label.cpu(), cmap="gray") 
plt.show()

# %%


