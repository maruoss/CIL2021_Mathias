from numpy import imag
import torch
import torchvision.transforms.functional as TF
from torch import nn

# Deeplabv3 test augmentation: model["out"] in this function

def patch_test_augmentation(imagebatch: torch.tensor, model, device, patch_size: tuple):
    """Patch Test augmentation: 8 square symmetric augmentations + patch predictions based on Bastian Morath, Daniel Peter, Michael Seeber, Kam-Ming Mark Tam
    ETH CIL Project 2020: Blending Generative and Discriminative U-nets for Road Segmentation of Aerial Images"""
    # image size (H, W) of dataloader -> list [H, W]
    input_spatialsize = TF._get_image_size(imagebatch)
    # batch size of dataloader
    batch_size = imagebatch.shape[0]
    # Patch size, tuple
    patch_size = patch_size

    # Instantiate Unfold/ Fold classes
    fold_params = dict(kernel_size=patch_size, dilation=1, padding=0, stride=144)
    unfold = nn.Unfold(**fold_params)
    fold = nn.Fold(output_size=input_spatialsize, **fold_params) #final output size = input size

    # divisor: normalizes the overlapping folds
    input_ones = torch.ones(imagebatch[:, :1].shape, dtype=imagebatch.dtype) # need shape [B, 1, H, W] not [B, 3, H, W] to fold preds back
    divisor = fold(unfold(input_ones)).to(device) # attach to cuda, otherwise error


    # Collect patches in list
    collect_patches = []
    num_patches = 4 # fixed here, calc. from Pytorch L = ... formula. -> depends on dilation, padding, stride

    for i in range(num_patches):
        # Unfold image batch into [batchsize (8), 3*kernelsize[0]*kernelsize[1] (196608), num_patches (25)] -> take 1 patch and reshape to [B, C, K[0], K[1]]
        image_patch = unfold(imagebatch)[:, :, i].reshape(batch_size, 3, patch_size[0], patch_size[1])
        
        # Transform input batch with 8 square symmetric augm.
        t_imgs = []
        t_imgs.append(image_patch) # original image batch
        t_imgs.append(TF.rotate(image_patch, 90)) # Rotate back 90°
        t_imgs.append(TF.rotate(image_patch, 180)) # Rotate back 180° 
        t_imgs.append(TF.rotate(image_patch, 270)) # Rotate back 270°
        t_imgs.append(TF.hflip(image_patch)) # hflip back
        t_imgs.append(TF.vflip(image_patch)) #vlip fack
        t_imgs.append(TF.rotate(TF.hflip(image_patch), 90)) # reflect about diagonal ac (a=(0,0), c=(1, 1))
        t_imgs.append(TF.rotate(TF.vflip(image_patch), 90)) # reflect about diagonal bd (b=(1,0), d=(0, 1))

        # Inference on each transformed batch (torch.sigmoid for probs., otherwise logits are returned here)
        t_preds = torch.stack([model(i.to(device))["out"] for i in t_imgs]).view(-1, 1, patch_size[0], patch_size[1]) # predict on each batch, take probs., reshape as (t*B, 1, H, W)
        
        # Inverse transforms on predictions
        origimg, img90, img180, img270, imghflip, imgvflip, imgdiagac, imgdiagbd = torch.split(t_preds, batch_size, dim=0) # split up tensor into tuple of each tensor batch prediction
        img90 = TF.rotate(img90, -90) # Rotate back 90°
        img180 = TF.rotate(img180, -180) # Rotate back 180° 
        img270 = TF.rotate(img270, -270) # Rotate back 270°
        imghflip = TF.hflip(imghflip) # hflip back
        imgvflip = TF.vflip(imgvflip) #vlip back
        imgdiagac = TF.hflip(TF.rotate(imgdiagac, -90)) # reflect about diagonal ac (a=(0,0), c=(1, 1))
        imgdiagbd = TF.vflip(TF.rotate(imgdiagbd, -90)) # reflect about diagonal bd (b=(1,0), d=(0, 1))
        y_ensemble = torch.stack((origimg, img90, img180, img270, imghflip, imgvflip, imgdiagac, imgdiagbd))
        # Take mean of all predictions for final prediction
        y_hat_mean = y_ensemble.mean(dim=0) # should be shape [batchsize, 1, H, W]

        # Reshape to [8, -1] and append to patch list
        collect_patches.append(y_hat_mean.reshape(batch_size, -1))

    # Fold predictions back together to [B, 1, H, W] + Normalize
    y_hat_patch_aug = (fold(torch.stack(collect_patches, dim=2))) / divisor # Normalize: divide through sum of 1's (middle of image larger sum)

    return y_hat_patch_aug #logits, no torch.sigmoid is used above