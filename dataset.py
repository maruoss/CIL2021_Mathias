from PIL import Image
import torch
from augmentation import transform_fn

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, groundtruth_paths: list, train=True, resize_to=(400, 400)): #RESIZE TO?
        # Initialize imagepaths, grountruthpaths and transform
        self.image_paths = image_paths
        self.groundtruth_paths = groundtruth_paths
        self.train = train # Train or validation mode?
        self.resize_to = resize_to # Resize image to which size?

    def __getitem__(self, idx: int):
        """loads image, applies transformation function"""
        # Numpy not necessary, gives errors on transforms. ToTensor transforms at the end transforms to tensor anyway.
        im = Image.open(self.image_paths[idx])
        gt = Image.open(self.groundtruth_paths[idx])

        # Apply augmentation function to images and label
        im, gt = transform_fn(im, gt, train=self.train, resize_to=self.resize_to)

        return im, gt


    def __len__(self):
        return len(self.image_paths)