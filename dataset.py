from PIL import Image
import numpy as np
import torch
from augmentation import transform_fn

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images: np.array, groundtruths: np.array, train=True, resize_to=(400, 400)): #RESIZE TO?
        # Initialize imagepaths, grountruthpaths and transform
        self.image_paths = images # np.array of np.arrays (1c)
        self.groundtruth_paths = groundtruths # np.array of np.arrays (1c)
        self.train = train # Train or validation mode?
        self.resize_to = resize_to # Resize image to which size?

    def __getitem__(self, idx: int):
        """loads image, applies transformation function"""
        # Numpy not necessary, gives errors on transforms. ToTensor transforms at the end transforms to tensor anyway.
        # im = Image.open(self.image_paths[idx]) # used when paths are provided instead of images, slower for small scale in training
        # gt = Image.open(self.groundtruth_paths[idx]) # used when paths are provided instead of images, slower for small scale in training
        im = self.image_paths[idx] #faster for small scale, use above to scale to lot of images
        gt = self.groundtruth_paths[idx]

        # Apply augmentation function to images and label
        im, gt = transform_fn(im, gt, train=self.train, resize_to=self.resize_to)

        return im, gt


    def __len__(self):
        return len(self.image_paths)