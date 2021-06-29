import torch
import torchvision.transforms.functional as TF

# Deeplabv3 test augmentation: model["out"] in this function

def test_augmentation(imagebatch: torch.tensor, model, device):
    """Test augmentation: 8 square symmetric augmentations based on Bastian Morath, Daniel Peter, , Michael Seeber, Kam-Ming Mark Tam
    ETH CIL Project 2020: Blending Generative and Discriminative U-nets for Road Segmentation of Aerial Images"""
    # image size (H, W) of dataloader
    img_size = TF._get_image_size(imagebatch)
    # batch size of dataloader
    batch_size = imagebatch.shape[0]
    
    # Transform input batch with 8 square symmetric augm.
    t_imgs = []
    t_imgs.append(imagebatch) # original image batch
    t_imgs.append(TF.rotate(imagebatch, 90)) # Rotate back 90°
    t_imgs.append(TF.rotate(imagebatch, 180)) # Rotate back 180° 
    t_imgs.append(TF.rotate(imagebatch, 270)) # Rotate back 270°
    t_imgs.append(TF.hflip(imagebatch)) # hflip back
    t_imgs.append(TF.vflip(imagebatch)) #vlip fack
    t_imgs.append(TF.rotate(TF.hflip(imagebatch), 90)) # reflect about diagonal ac (a=(0,0), c=(1, 1))
    t_imgs.append(TF.rotate(TF.vflip(imagebatch), 90)) # reflect about diagonal bd (b=(1,0), d=(0, 1))

    # Inference on each transformed batch (torch.sigmoid for probs., otherwise logits are returned here)
    t_preds = torch.stack([model(i.to(device))["out"] for i in t_imgs]).view(-1, 1, img_size[0], img_size[1]) # predict on each batch, take probs., reshape as (t*B, 1, H, W)
    
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

    return y_hat_mean #logits, if no torch.sigmoid is used in inference