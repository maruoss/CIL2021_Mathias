# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from pathlib import Path
import time
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils import tensorboard
import torchvision
from torchvision import models
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Local Module imports
from dataset import CustomDataset
from losses.diceloss import BinaryDiceLoss_Logits
from losses.bce_diceloss import BCEDiceLoss_Logits
from models.model import createDeepLabHead
from trainer import train_model
from metrics import accuracy_fn, patch_accuracy
from augmentation import test_transform_fn
from trainer_baseline import train_baseline
from models.unet_baseline import UNet_baseline
from torchinfo import summary
from models.trivial_baseline import Trivial_baseline
from models.fcn_resnet50_baseline import createFCNHead
from losses.cldiceloss import SoftDiceCLDice
from models.deeplav3_mobilenet import createDeepLabHead_mobilenet
from models.deeplabv3_resnet50 import createDeepLabHead_resnet50
from losses.focalloss_kornia import BinaryFocalLossWithLogits
from losses.bce_cldiceloss import BCE_SoftDiceCLDice
from patch_test_augmentation import patch_test_augmentation

# %%
# Define random seeds
random_seed = 343
#
torch.manual_seed(random_seed)
random.seed(random_seed) # also needed for transforms.RandomRotation.get_params...
np.random.seed(random_seed) # global numpy RNG

# Define root to current working directory + "/Data" -> LEONHARD: MAY HELP DETERM ROOT PATH
root = Path.cwd() / 'Data'
print("Your current working directory + '/ Data' path: \n" + str(root))


# %%
# Get path names in a list
def get_filenames_of_path(path: Path, ext: str = '*') -> list:
    """Returns a list of sorted? files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()] # path.glob is windowspath, with extension "*" for all images in respective folder
    return filenames


# %%
# Apply function to get list of pathnames
image_paths = get_filenames_of_path(root / "training" / "training" / "images")
print("Train set first path: ", image_paths[0])
groundtruth_paths = get_filenames_of_path(root / "training" / "training" / "groundtruth")
print("Mask set first path: ", groundtruth_paths[0])

# Test pictures
test_image_paths = get_filenames_of_path(root / "test_images" / "test_images") # sorted already by pathlib somehow, see below
print("Test set first path: ", test_image_paths[0])


# %%
# import torch
# torch.tensor(3) + torch.tensor(3)

# # %%
# import numpy as np
# a = np.array([1, 2, 3])

# if isinstance(a, np.ndarray):
#     print("yes")

# %%
# test_image_paths[0]

# for i in sorted(test_image_paths):
#     print(int(re.search(r"\d+", i.name).group(0)))

# %%

# with open("dummy_test_file.csv", "w") as f:
#     f.write("id, prediction\n")
#     for fn in sorted(test_image_paths):
#         img_number = int(re.search(r"\d+", fn.name).group(0))
#         print(img_number)

# %%
# a = [file for file in sorted(root.glob("test_images/test_images/*"))] # we see: pathlib sorts already somehow
# b = [file for file in root.glob("test_images/test_images/*")]
# assert a == b # pathlib.glob sorts already

# %%
# TUTORIAL WAY OF DOING IT (WITHOUT PATH)
# test_path = 'Data/test_images/test_images'
# test_filenames = sorted(glob(test_path + '/*.png'))

# test_filenames
# # %%
# test_path = 'Data/training/training/images'
# test_filenames = sorted(glob(test_path + '/*.png'))

# test_filenames


#%%

# for f in root.glob("*"):

#     print(f)

# type(root)

# len(sorted(root.glob("training/training/images/*")))

# %%
# a = [1, 2, 3]
# b = {"one_key": ["one_value"], "two_key": ["two_value"], "three_key": ["three_value"], "four_key": ["four_value"]}

# for c, d, e in zip(a, b.keys(), b.values()):
#     print(c, d, e)


# for k, v in b.items():
#     print(k, v)


# %%
# # Image Transform function
# transform_fn = transforms.Compose(
#     [
#     # transforms.Resize(256), # has to be multiple of 16 for patch acc. metric
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalize needed for deeplabv3
#     ])

# # Target Transform function
# target_transform_fn = transforms.Compose(
#     [
#     # transforms.Resize(256), # has to be multiple of 16 for patch acc. metric
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5), (0.5)) # Normalize to [-1, 1]
#     ])


# %%
# Split data into training validation set
train_size = 0.9

train_image_paths, val_image_paths, train_groundtruth_paths, val_groundtruth_paths = train_test_split(
    image_paths, groundtruth_paths, random_state=random_seed, train_size=train_size, shuffle=True)

# Assert that images and groundtruth picture numbers in train and val paths coincide, respectively
assert [y[-7:] for y in [str(x) for x in train_image_paths]] == [y[-7:] for y in [str(x) for x in train_groundtruth_paths]]
assert [y[-7:] for y in [str(x) for x in val_image_paths]] == [y[-7:] for y in [str(x) for x in val_groundtruth_paths]]

# Print length of train/ val sets
print(len(train_image_paths))
print(len(val_image_paths))

# %%

# Define device "cuda" for GPU, or "cpu" for CPU
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define batch size for Dataloaders
BATCH_SIZE = 8 # not too large, causing memory issues!

# Set picture size on which model will be trained
resize_to = (400, 400)  

# ***************************************************************************
# 3 Options how to load in images:
# 1a) Load from paths, LARGE SCALE: best if goal is to scale to a lot of images, dont have to load all images at once)

# # 1b) Load images as PIL images, in a numpy array (gives PIL image loading warning, rather use 1c))
# train_images = np.array([Image.open(img) for img in sorted(train_image_paths)])
# train_groundtruths = np.array([Image.open(img) for img in sorted(train_groundtruth_paths)])
# val_images = np.array([Image.open(img) for img in sorted(val_image_paths)])
# val_groundtruths = np.array([Image.open(img) for img in sorted(val_groundtruth_paths)])

# # 1c) Load images as np.array, in a numpy array # SMALL SCALE: reload seems to be faster at training. better for small set of images.
train_images = np.stack([np.array(Image.open(img)) for img in sorted(train_image_paths)]) # no astype(float32)/255., as otherwise transform wouldnt work (needs PIL images, and TF.to_pil_image needs it in this format)
train_groundtruths = np.stack([np.array(Image.open(img)) for img in sorted(train_groundtruth_paths)])
val_images = np.stack([np.array(Image.open(img)) for img in sorted(val_image_paths)])
val_groundtruths = np.stack([np.array(Image.open(img)) for img in sorted(val_groundtruth_paths)])

# # For 1b), 1c): Instantiate Datasets for train and validation IMAGES
train_dataset = CustomDataset(train_images, train_groundtruths, train=True, resize_to=resize_to) # train=True
val_dataset = CustomDataset(val_images, val_groundtruths, train=False, resize_to=resize_to) # train=False

# # For 1a): Instantiate Datasets for train and validation PATHS
# train_dataset = CustomDataset(train_image_paths, train_groundtruth_paths, train=True, resize_to=resize_to) # train=True
# val_dataset = CustomDataset(val_image_paths, val_groundtruth_paths, train=False, resize_to=resize_to) # train=False

# %% ***************************************************************************
# Instantiate Loaders for these datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) # pin memory speeds up the host to device transfer
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


# %%

# Display image and labels
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = torch.moveaxis(train_features[0], 0, -1) #Select first image out of batch, reshape CHW to HWC (plt.imshow needs shape H, W, C)
label = train_labels[0] # Select first label out of shape BHW
# Show image, .cpu() otherwise imshow doesnt work with cuda
plt.imshow(img)
plt.show()
# Squeeze only if target_transform is provided -> transformed int into torch.float [0, 1] range, and added channel
label = label.squeeze()
# Show groundtruth in greyscale (original interpretation)
plt.imshow(label, cmap="gray") 
plt.show()

# torch.unique(label)
# %%

# train_features[0].shape



# %%
# Count class imbalance
# Percent pixel of images
Threshold = 0.25
perc_pixel_train = (train_groundtruths / 255. > Threshold).sum() / np.ones(train_groundtruths.shape).sum()
print(f"{perc_pixel_train*100:.2f}% of train images consist of roads")

# Positive class weight
road_weight = (train_groundtruths/255. <= Threshold).sum() / (train_groundtruths/255. > Threshold).sum()
print(f'Road weight: Sum Train Pixel Background/ Sum Train Pixel Road = {road_weight:.2f}')

# perc_pixel_train = (val_groundtruths / 255. > Threshold).sum() / np.ones(val_groundtruths.shape).sum()
# print(f"{perc_pixel_train*100:.2f}% of val images consist of roads")

# perc_pixel_train = ((train_groundtruths / 255. > Threshold).sum() + (val_groundtruths / 255. > 0.25).sum()) / (np.ones(train_groundtruths.shape).sum() + np.ones(val_groundtruths.shape).sum())
# print(f"{perc_pixel_train*100:.2f}% of all images consist of roads")

# (val_groundtruths / 255. > 0.).sum() / np.ones(val_groundtruths.shape).sum()


# (train_groundtruths[0] /255.  > 0.25).sum()
# 400*400
# np.unique(train_groundtruths[0] / 255.)
# np.ones(train_groundtruths.shape).sum()
# np.ones(train_groundtruths.shape).shape
# np.ones(train_groundtruths.shape).sum()
# train_groundtruths / 255.

#%%

# model = createDeepLabHead() # function that loads pretrained deeplabv3 and changes classifier head
# model.to(default_device) #add to gpu
# model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
# %%


# for i in model.backbone:
#     print(i)

# for x in model.backbone.layer4.parameters():
#     x.requires_grad = False

# for x in model.backbone.parameters():
#     print(x.requires_grad)

#%%

# for i in model.classifier:
    # print(i)

# %%

# Finetuning or Feature extraction? Freeze backbone of resnet101
# for x in model.backbone.parameters():
#     x.requires_grad = False

# %%

train_features, train_labels = next(iter(train_dataloader))

# %%

len(train_labels.shape)

#%%

(train_labels.sum(dim=1) == train_labels.squeeze(1))
# %%

train_labels

#%% PATCH TEST AUGMENTATION
# a = train_features

# # fold_params = dict(kernel_size=256, dilation=1, padding=0, stride=36)
# # fold = nn.Fold(output_size=400, **fold_params)
# # unfold = nn.Unfold(**fold_params)

# # # divisor
# # input_ones = torch.ones(a.shape, dtype=a.dtype)
# # divisor = fold(unfold(input_ones))

# model = createDeepLabHead()
# model.to(default_device)
# model.load_state_dict(torch.load("state_bcedice0.5e100lr.001batch32img224+e50lr.001batch8img304+e50lr.001batch8img400+FINE.e50lr.0001batch2img224.pt"))
# # Finetuning or Feature extraction? Freeze backbone of resnet101
# model.eval()
# with torch.no_grad():
#     y_hat, divisor = patch_test_augmentation(a.to(default_device), model=model, device=default_device, patch_size=(256, 256))
 
# %%
# plt.imshow(torch.sigmoid(y_hat)[0][0].cpu(), cmap="gray")
# plt.show()

# %%
# torch.unique(divisor)
# k = torch.sigmoid(y_hat[0])
# plt.imshow((torch.sigmoid(y_hat[0][0].cpu() * divisor[0][0].cpu()) > 0.25).float(), cmap="gray")
# plt.show()


# %%
# UNFOLDING / FOLDING
# import torch.nn.functional as F
# a = train_features

# transpose is necessary if manually folding/ unfolding
# a = a.unfold(2, 256, 36).unfold(3, 256, 36).reshape(-1, 3, 256, 256)
# g = a.unfold(2, 256, 36).unfold(3, 256, 36).transpose(1, 3).reshape(-1, 3, 256, 256)
# plt.imshow(TF.to_pil_image(g[0]))
# plt.show()
# (a.unfold(2, 256, 36).unfold(3, 256, 36).transpose(1, 3).reshape(-1, 3, 256, 256)[0] == F.unfold(a, 256, stride=36)[0][:, 0].reshape(3, 256, 256)).all()

# %%
# plt.imshow(TF.to_pil_image(a[0]))
# plt.show()

# b = F.unfold(a, kernel_size=(256, 256), dilation=1, padding=0, stride=36)

# # %% F.fold
# print("b shape:", b.shape)
# print("b reshaped:", b.reshape(-1, 3, 256, 256)[0].shape)
# plt.imshow(TF.to_pil_image(b.reshape(-1, 3, 256, 256)[0]))
# plt.show()
# c = F.fold(b, output_size=[400, 400], kernel_size=(256, 256), dilation=1, padding=0, stride=36)
# print("c shape:", c.shape)

# # divisor
# input_ones = torch.ones(a.shape, dtype=a.dtype)
# divisor = F.fold(F.unfold(input_ones, kernel_size=(256, 256), dilation=1, padding=0, stride=36), output_size=400, kernel_size=(256, 256),
# dilation=1, padding=0, stride=36)


# plt.imshow(TF.to_pil_image(c[0] / divisor[0]))
# plt.show()
# %%
# DIFFERENCE UNFOLD AND SLICING
# a[:, :, :256, :256].shape
# plt.imshow(TF.to_pil_image(a[:, :, :256, :256][0]))
# plt.show()

# b = F.unfold(a, kernel_size=(256, 256), dilation=1, padding=0, stride=36)
# print("b shape:", b.shape)
# print("b reshaped:", b.reshape(-1, 3, 256, 256)[0].shape)
# plt.imshow(TF.to_pil_image(b.contiguous().view(-1, 3, 256, 256)[0]))
# plt.show()

# %%
# Class syntax
# fold_params = dict(kernel_size=256, dilation=1, padding=0, stride=36)
# fold = nn.Fold(output_size=400, **fold_params)
# unfold = nn.Unfold(**fold_params)

# # divisor
# input_ones = torch.ones(a.shape, dtype=a.dtype)
# divisor = fold(unfold(input_ones))

# #apply classes
# d = fold(unfold(a))

# plt.imshow(TF.to_pil_image(d[0] / divisor[0]))
# plt.show()

# # %%

# plt.imshow(TF.to_pil_image(unfold(a)[0][:, 0].reshape(3, 256, 256)))
# plt.show()

# %%
# a = train_features
# model = createDeepLabHead()
# model.to(default_device)
# model.load_state_dict(torch.load("state_bcedice0.5auxpatch.e100lr.001.batch8.img400+fine.bcedice0.5.e100.lr.00001bat2img224.pt"))
# # Finetuning or Feature extraction? Freeze backbone of resnet101
# model.eval()
# with torch.no_grad():
#     a = unfold(a)[:, :, 0].reshape(8, 3, 256, 256)
#     a = torch.sigmoid(model(a.to(default_device))["out"])

# plt.imshow(a.cpu()[0].squeeze(0))
# plt.show()


# %%
# a = train_features
# input_ones = torch.ones(a.shape, dtype=a.dtype)
# divisor = fold(unfold(input_ones))

# torch.mean(a, dim=1, keepdim=True).shape

# a[:, :1].shape
# a = train_features
# unfold(a)[:, :, 24].shape


# %%
# plt.imshow(TF.to_pil_image(unfold(a)[:, :, 0].reshape(8, 3, 256, 256)[0]))
# plt.show()
# torch.stack(([a.reshape(8, -1)]*25), dim=2).shape

# unfold(a)[:, :, 0].reshape(8, 3, 256, 256).shape
# fold(torch.stack(([a.reshape(8, -1)]*25), dim=2)).shape
# patcher = nn.AvgPool2d(16)
# mask = patcher(train_labels)

# # mask[:, 0][mask[:, 0] > 0.25] = 1
# mask[:, 0][mask[:, 0] > 0.25] = 1
# mask
# torch.unique(mask[:, 0])
# (divisor[:, 0] == divisor[:, 2])

# # train_features.shape
# grid = torchvision.utils.make_grid(train_features)
# grid.shape

# x0, y0 = train_features[0], train_labels[0]
# x1.shape
# y0.requires_grad

# np.equal(y0.repeat([3, 1, 1]).numpy() , np.concatenate(([y0.numpy()] * 3), 0)).all()

#  np.concatenate(([y0.numpy()] * 3), 0)

# np.concatenate(([y0.numpy()] * 3), 0)
# y0.unsqueeze(0).repeat((1, 3, 1, 1)).shape


#%%
# orig_img = Image.open(image_paths[1])

# def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]

#     num_rows = len(imgs)
#     num_cols = len(imgs[0]) + with_orig
#     fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         row = [orig_img] + row if with_orig else row
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     if with_orig:
#         axs[0, 0].set(title='Original image')
#         axs[0, 0].title.set_size(8)
#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set(ylabel=row_title[row_idx])

#     plt.tight_layout()
#     plt.show()

# (top_left, top_right, bottom_left, bottom_right, center) = transforms.FiveCrop(size=(256, 256))(orig_img)

# plot([top_left, top_right, bottom_left, bottom_right, center])


# TF.to_tensor(top_left)




#%%
label = train_labels[0].squeeze()
label = (label > 0.9).float()
# label = transforms.functional.pad(label, padding=100, padding_mode="reflect")
# label.shape
# plt.imshow(torch.moveaxis(label.cpu(), 0, -1), cmap="gray")
# plt.show()
plt.imshow(label, cmap="gray") 
plt.show()

# %%

# Test for image transformations:
image = torch.from_numpy(np.moveaxis(np.array(Image.open(image_paths[1])), -1, 0))
# label = torch.unsqueeze(torch.from_numpy(np.array(Image.open(groundtruth_paths[0]))), dim=0)
# image = train_features[0]

# print(image.shape)
plt.imshow(torch.moveaxis(image.cpu(), 0, -1))
plt.show()

# print(label.shape)

# transform = transforms.Compose([transforms.RandomRotation(90)])
# image = transform(image)

# image = TF.rgb_to_grayscale(image, num_output_channels=3)
# image = TF.gaussian_blur(image, kernel_size=9, sigma=0.5)
# image = transforms.functional.adjust_brightness(image, brightness_factor=1.4)
# image = transforms.functional.adjust_contrast(image, contrast_factor=1.2)
# image = transforms.functional.adjust_hue(image, hue_factor=0.1)
# image = transforms.functional.adjust_saturation(image, saturation_factor=0.8)
# image = transforms.functional.adjust_sharpness(image, sharpness_factor=0.8)
# image = transforms.functional.equalize(image)


# image = transforms.functional.pad(image, padding=100, padding_mode="reflect")
# image = TF.rotate(image, 45)
# image = transforms.functional.center_crop(image, (400, 400))


i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 0.5), ratio=(1, 1))
image = transforms.functional.resized_crop(image, i, j, h, w, (400, 400))

# i, j, h, w = transforms.RandomCrop.get_params(image, (400, 400))
# image = transforms.functional.crop(image, i, j, h, w)

# i, j, h, w, v = transforms.RandomErasing.get_params(image, scale=(0.1, 0.1), ratio=(1, 1), value=[0])
# image = TF.erase(image, i=i, j=j, h=h, w=w, v=v)


# image = TF.hflip(image)
# image = TF.rotate(image, 90)
# image = transforms.functional.resize(image, (256, 256))
# image = transforms.functional.center_crop(image, (128, 128))

# image = TF.hflip(image)
# image = TF.vflip(image)
print(image.shape)
plt.imshow(torch.moveaxis(image.cpu(), 0, -1))
plt.show()

# %%
# # train_features.shape

# # TF.to_tensor(train_images)
# list((8,8)) == [8, 8]
# # a = Image.open(image_paths[1])
# test = []
# test.append(train_features)
# test.append(TF.rotate(train_features, 90))
# test.append(TF.rotate(train_features, 180)) # Rotate back 180° 
# test.append(TF.rotate(train_features, 270)) # Rotate back 270°
# test.append(TF.hflip(train_features)) # hflip back
# test.append(TF.vflip(train_features)) #vlip fack
# # test.append(TF.rotate(TF.hflip(train_features), 90)) # reflect about diagonal ac (a=(0,0), c=(1, 1))
# # test.append(TF.rotate(TF.vflip(train_features), 90)) # reflect about diagonal bd (b=(1,0), d=(0, 1))

# # torch.stack(test).shape


# # a = torch.stack((train_features, TF.rotate(train_features, 90), TF.rotate(train_features, 180), 
# #     TF.rotate(train_features, 270), TF.hflip(train_features), TF.vflip(train_features),
# #     TF.rotate(TF.hflip(train_features), 90), TF.rotate(TF.vflip(train_features), 90))).view(-1, 3, 400, 400)


# model = createDeepLabHead()
# model.to(default_device)
# model.load_state_dict(torch.load("state_bcedice0.5auxpatch.e100lr.001.batch8.img400+fine.bcedice0.5.e100.lr.00001bat2img224.pt"))
# # Finetuning or Feature extraction? Freeze backbone of resnet101
# model.eval()
# with torch.no_grad():
#     # a, b, c, d, e, f = torch.split(a, BATCH_SIZE)
#     # a = torch.sigmoid(model(a.to(default_device))["out"])
#     x = torch.stack([torch.sigmoid(model(i.to(default_device))["out"]) for i in test]).view(-1, 1, 400, 400)
#     origimg, img90, img180, img270, imghflip, imgvflip, imgdiagac, imgdiagbd = torch.split(x, BATCH_SIZE)
#     img90 = TF.rotate(img90, -90) # Rotate back 90°
#     img180 = TF.rotate(img180, -180) # Rotate back 180° 
#     img270 = TF.rotate(img270, -270) # Rotate back 270°
#     imghflip = TF.hflip(imghflip) # hflip back
#     imgvflip = TF.vflip(imgvflip) #vlip back
#     imgdiagac = TF.hflip(TF.rotate(imgdiagac, -90)) # reflect about diagonal ac (a=(0,0), c=(1, 1))
#     imgdiagbd = TF.vflip(TF.rotate(imgdiagbd, -90)) # reflect about diagonal bd (b=(1,0), d=(0, 1))
#     y = torch.stack((origimg, img90, img180, img270, imghflip, imgvflip, imgdiagac, imgdiagbd))


# %%

# y.shape
# plt.imshow(TF.to_pil_image(y.mean(0)[1]), cmap="gray")
# plt.show()

# #%%
# plt.imshow(TF.to_pil_image(x[1]), cmap="gray")
# plt.show()

# # %%
# train_features.shape[0]

# train_features.shape
# # %%
# train_features_raw = torch.from_numpy(train_images).moveaxis(-1, 1)[8:16]
# a = torch.stack((train_features_raw, TF.rotate(train_features_raw, 90), TF.rotate(train_features_raw, 180), 
#     TF.rotate(train_features_raw, 270), TF.hflip(train_features_raw), TF.vflip(train_features_raw),
#     TF.rotate(TF.hflip(train_features_raw), 90), TF.rotate(TF.vflip(train_features_raw), 90))).view(-1, 3, 400, 400)

# %%

# def pred_inv_transform(images: torch.tensor):
#     origimg, img90, img180, img270, imghflip, imgvflip = torch.split(images, BATCH_SIZE)
#     img90 = TF.rotate(img90, -90) # Rotate back 90°
#     img180 = TF.rotate(img180, -180) # Rotate back 180° 
#     img270 = TF.rotate(img270, -270) # Rotate back 270°
#     imghflip = TF.hflip(imghflip) # hflip back
#     imgvflip = TF.imgvflip(imgvflip) #vlip fack

#     return 

# Simulate output
# a = torch.stack((train_features, TF.rotate(train_features, 90), TF.rotate(train_features, 180), 
#     TF.rotate(train_features, 270), TF.hflip(train_features), TF.vflip(train_features),
#     TF.rotate(TF.hflip(train_features), 90), TF.rotate(TF.vflip(train_features), 90))).view(-1, 3, 400, 400)
# # a = a.to(default_device)
# model = createDeepLabHead()
# model.to(default_device)
# model.load_state_dict(torch.load("state_bcedice0.5auxpatch.e100lr.001.batch8.img400+fine.bcedice0.5.e100.lr.00001bat2img224.pt"))
# # Finetuning or Feature extraction? Freeze backbone of resnet101
# model.eval()
# with torch.no_grad():
#     # a, b, c, d, e, f = torch.split(a, BATCH_SIZE)
#     # a = torch.sigmoid(model(a.to(default_device))["out"])
#     x = torch.stack([torch.sigmoid(model(i.to(default_device))["out"]) for i in torch.split(a, BATCH_SIZE)]).view(-1, 1, 400, 400)
#     origimg, img90, img180, img270, imghflip, imgvflip, imgdiagac, imgdiagbd = torch.split(x, BATCH_SIZE)
#     img90 = TF.rotate(img90, -90) # Rotate back 90°
#     img180 = TF.rotate(img180, -180) # Rotate back 180° 
#     img270 = TF.rotate(img270, -270) # Rotate back 270°
#     imghflip = TF.hflip(imghflip) # hflip back
#     imgvflip = TF.vflip(imgvflip) #vlip back
#     imgdiagac = TF.hflip(TF.rotate(imgdiagac, -90)) # reflect about diagonal ac (a=(0,0), c=(1, 1))
#     imgdiagbd = TF.vflip(TF.rotate(imgdiagbd, -90)) # reflect about diagonal bd (b=(1,0), d=(0, 1))
#     y = torch.stack((origimg, img90, img180, img270, imghflip, imgvflip, imgdiagac, imgdiagbd))

# %%

# y.shape
# plt.imshow(TF.to_pil_image(y.mean(0)[1]), cmap="gray")
# plt.show()

# %%
# torch.split(x, BATCH_SIZE)[0].shape
# a, b, c, d, e, f, g, h= torch.split(a, BATCH_SIZE)

# b = TF.rotate(b, -90)
# c = TF.rotate(c, -180)
# d = TF.rotate(d, -270)
# e = TF.hflip(e)
# f = TF.vflip(f)
# g = TF.hflip(TF.rotate(g, -90))
# h = TF.vflip(TF.rotate(h, -90))

# %%
# plt.imshow(TF.to_pil_image(x[1]), cmap="gray")
# plt.show()
# %%
# TF.to_pil_image(torch.stack((x[0][0], x[1][0])).mean(dim=0))


# %%

# TF.to_pil_image(g[0])




# %%
# tuple(TF._get_image_size(image))
# len(list(range(-180, 181, 15)))
# x.shape
# print(image.shape)

# %%
# import timeit
# since = time.time()
# %timeit -n 10000 -r 50 random.uniform(0.5, 1)
# print(time.time() - since)

# # %%
# since = time.time()
# %timeit -n 10000 -r 50 transforms.RandomRotation.get_params([0.5, 1])
# print(time.time() - since)
# # %%
# since = time.time()
# %timeit -n 10000 -r 50 torch.FloatTensor(0).uniform_(0.5, 1)
# print(time.time() - since)

# %%
# image.view(3, -1).shape


# %%
# model.classifier
# print(model)
# a = Image.open(image_paths[0])
# # print(len(a))
# a = TF._get_image_size(a)
# a

# %%
# BATCH_SIZE=3
# def f():
#     return (3 + batch_size)
# f()

# %%
# def f():
#     if random.random() > 0.5:
#         print("yes")
#     if random.random() < 0.5:
#         print("no")
#     else: print("\nno result")
# f()

# %%
# class FirstNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(400*400*3, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1),
#             # nn.Sigmoid() # use Sigmoid + BCELoss, or BCEwithLogitLoss without Sigmoid -> numerical stability (logsumexptrick)
#         )
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


# %%
# DICE LOSS

# Test output with Loss functions
# print(train_features[0].shape) # We need shape [Batch, C, H, W] for model
# print(torch.stack([train_features[0], train_features[1]]).shape)
# logits = model(torch.stack([train_features[0], train_features[1]]).to(default_device)) # unsqueeze first dim to get Batch=1, for 1 example

# print(logits["out"].shape)
# # target = torch.tensor([1]).unsqueeze(0).to(device) # target class 0, shape [B, scalar]

# a = logits["out"].view(logits["out"].shape[0], -1)
# b = (logits["out"]+1.).view(logits["out"].shape[0], -1)

# a - b

# %%
# DICELOSS
# a = torch.tensor(([1., 2, 3, 4], [5, 6, 7, 8]))
# b = torch.tensor([[1., 1., 1, 1], [0, 0, 0, 1]])
# print(a.shape)
# print(b.shape)
# c = torch.mul(a, b)
# print(c)
# d = 1 - (torch.sum(c, dim=1) + 1)
# print(d)
# d.mean()
# #%%
# print(a.pow(2))
# print(b.pow(2))
# print(a.pow(2) + b.pow(2))
# torch.sum(a.pow(2) + b.pow(2), dim=1)


# %%
# # Train one batch, without learning anything
# model = createDeepLabHead()
# model.to(default_device)
# model.load_state_dict(torch.load("state_bcedice0.5e100lr.001batch8img400.pt"))
# # Finetuning or Feature extraction? Freeze backbone of resnet101
# for x in model.backbone.parameters():
#     x.requires_grad = False
# model = Trivial_baseline()
# model.eval()
# img = train_features.to(default_device) # Take first image from DataLoader
# model(img)
# torch.sigmoid(model(img))
# # a = model(img)["aux"]
# # b = model(img)["out"]
# # # %%
# # plt.imshow(torch.sigmoid(a[0][0]).detach().cpu().numpy(), cmap="gray")
# # plt.show()
# # plt.imshow(torch.sigmoid(b[0][0]).detach().cpu().numpy(), cmap="gray")
# # plt.show()

# %%
# print(train_features.shape)
# torch.sigmoid(torch.ones_like(train_features)[:,0:1]*-1e10)
# (nn.functional.avg_pool2d(torch.sigmoid(b), 16) < 0.25).sum()

# torch.zeros_like(train_features[:,0:1]).shape

# %%
# print(img.shape)
# with torch.no_grad():
#     out = model(img)["out"].cpu()
# print(out.shape)
# plt.imshow(out.squeeze().detach().numpy(), cmap="gray")
# plt.imshow(np.concatenate([np.moveaxis(out.detach().numpy(), 0, -1)]*3, -1).squeeze())

# %%
# np.unique(out.detach().numpy())[-5:]
# %%

# plt.imshow(np.concatenate([np.moveaxis(out.detach().numpy(), 1, -1)]*3, -1).squeeze())
# plt.imshow((torch.moveaxis(torch.sigmoid(out), 1, -1)).squeeze(0))

# # Normalize to [0, 1]
# def normalize(x):
#     x = (x - np.min(x)) / (np.max(x) - np.min(x))
#     return x

# plt.imshow(normalize(np.concatenate([np.moveaxis(out.detach().numpy(), 1, -1)]*3, -1).squeeze()))
# plt.imshow(transforms.ToPILImage()(torch.cat([torch.moveaxis(out, 0, -1)]*3, 0).squeeze()))


# %% ######################################### TRAIN ############################################################
# Define global variables
PATCH_SIZE = 16
CUTOFF = 0.25

# Instantiate model
# function that loads pretrained deeplabv3 and changes classifier head
# Model sizes calculated with input size (8, 3, 400, 400):

model = createDeepLabHead() # 19GB (total size), 60Mio. Param, 18Mio. Trainable
# model = createFCNHead() # 10GB (total size), 35 Mio. Params., 35 Mio. Trainable
# Baseline model
# model = UNet_baseline(output_prob=False) # 8GB (total size), 31 Mio. Params, 31Mio trainable
# model = createDeepLabHead_resnet50() # 11GB (total size), 41 Mio. Params, 18Mio. Trainable
# model = createDeepLabHead_mobilenet() # 2.3GB (total size), 11 Mio. Total Params, 8 Mio. Trainable

# Assign model to device. Important!
model.to(default_device) #add to gpu (if available)

# Finetuning or Feature extraction? Freeze backbone of resnet101
for x in model.backbone.parameters():
    x.requires_grad = False

# Set some layers of the backbone to learnable
# for x in model.backbone.layer4.parameters():
#     x.requires_grad = True

# Instantiate optimizer
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Define loss function, BCEWithLogitLoss -> needs no sigmoid layer in neural net (num. stability)
# loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor(2))
# loss_fn = BinaryDiceLoss_Logits()
loss_fn = BCEDiceLoss_Logits(weight_dice=0.5) #weight of dce loss in (weight*DiceLoss + (1-weight)*BCELoss)
# loss_fn = SoftDiceCLDice(alpha=1., iter_=4)
# loss_fn = BCE_SoftDiceCLDice()
# loss_fn = FocalLoss()
# loss_fn = BinaryFocalLossWithLogits(alpha=0.25, reduction="mean")
print("Loss function used:" + str(loss_fn))

# patch_accuracy = patch_accuracy(y_hat, y, patch_size=16, cutoff=0.5)

# Define metrics
metric_fns = {'acc': accuracy_fn, "patch_acc": patch_accuracy}

# %%
# # Load model from saved model to finetune:
# model.load_state_dict(torch.load("state_bcedice0.5e100lr.001batch8img400_fullaug.pt"))
# Check if model is on cuda
next(model.parameters()).device

# Show summary of model
summary(model, (5, 3, 280, 280))


# print(model.backbone.conv1)
# model.classifier

# for i in model.backbone.keys():
#     print(i)

# str(BCEDiceLoss_Logits(weight_dice=0.8))[:7]
# %%
name_loss = str(loss_fn)[:7]
hyperparam_string = f".loss{name_loss}.lr{LEARNING_RATE}.batch{BATCH_SIZE}.img{resize_to[0]}"
comment = "" + hyperparam_string
# Train
# model =, since train_model returns model with best val_loss: "early stopped model"
model = train_model(train_dataloader, eval_dataloader=val_dataloader, model=model, loss_fn=loss_fn, 
             metric_fns=metric_fns, optimizer=optimizer, device=default_device, n_epochs=1, comment=comment)

# %% Save model for tinetuning conv layers
# torch.save(model.state_dict(), "state_bcedice0.5e100lr.001batch8img400+FINE.e100lr.00001.batch2img256_fullaug.pt")

# %%
# %load_ext tensorboard
# %%
# !rm -rf ./tensorboard
# %tensorboard --logdir tensorboard

# To launch tensorboard successfully: Open terminal. Enter: "tensorboard --logdir tensorboard" or "tensorboard --logdir=tensorboard" (both work)  -> should open on localhost
# %tensorboard --logdir=runs --host localhost --port 6006

# %%

# torch.save(model.state_dict(), "statetest.pt")Bd86bbd5Bd86bbd5

# import torch
# dict1 = torch.load("statetest.pt")
# model.load_state_dict(dict1)
# # Check if model is on cuda
# next(model.parameters()).device



# %%

# nest_dict = {1: {"train": 0.9, "val": 0.8}, 2: {"train": 0.7, "val": 0.6}}

# %%
# import numpy as np
# a = [1, 2, 3]
# b = np.array(a)

# b[1]

# %%

# nest_dict[1]["val"]

# %%
# for k, v in nest_dict.items():
#     print(k, v)


# model.load_state_dict(torch.load("state_bcedice0.5e100lr.001batch32img224+e50lr.001batch8img304+e50lr.001batch8img400+FINE.e50lr.0001batch2img224.pt"))

# ***************************************** PREDICTION ************************************************************
# Test prediction
# %% Taken from CIL Tutorial, https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_3.ipynb#scrollTo=cKQvTg8gE9JC
# with slight adjustment as we use pathlib instead of glob.glob
def create_submission(labels, test_filenames, submission_filename):
    test_path='test_images/test_images'
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), labels):
            img_number = int(re.search(r"\d+", fn.name).group(0)) # Adjusted: fn.name since here its a Path not a string, have to take string with .name
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))

# Function to show first and last test images and predicted masks
def show_first_last_pred(pred_torch_list: list, image_list: list, first_last=5):
    fig, axs = plt.subplots(first_last, 4, figsize=(10, 3*first_last))
    for i in range(first_last):
        # Image plot
        axs[i, 0].imshow(image_list[i])
        axs[i, 1].imshow(np.concatenate([np.moveaxis(pred_torch_list[i][0].detach().cpu().numpy(), 0, -1)] * 3, -1)) # Yields clipping warning, as deeplabv3 has input floats of <0 and >1
        axs[i, 2].imshow(image_list[-(i+1)])
        axs[i, 3].imshow(np.concatenate([np.moveaxis(pred_torch_list[-(i+1)][0].detach().cpu().numpy(), 0, -1)] * 3, -1)) # No warning now. yielded clipping warning -> because of no sigmoid func. in predictions!
        axs[i, 0].set_title(f'First Image {i+1}')
        axs[i, 1].set_title(f'First Pred {i+1}')
        axs[i, 2].set_title(f'Last Image {i+1}')
        axs[i, 3].set_title(f'Last Pred {i+1}')
        axs[i, 0].set_axis_off()
        axs[i, 1].set_axis_off()
        axs[i, 2].set_axis_off()
        axs[i, 3].set_axis_off()
    plt.show()

# %%
# Open first test image to have a look at it, resizing
a = Image.open(test_image_paths[0]).resize((400, 400))
a #should be test img number 10. test_image_paths should be sorted from get_filenames_of_path function.

#%%
# Load PIL Images from path list to a list
test_images = [Image.open(img) for img in sorted(test_image_paths)]
orig_size = test_images[0].size #should be test img number 10 (first image when sorted)
# size should be 608 (original test size)

# Predict on test images
test_pred_list = [] # empty list to collect tensor predictions shape [1, 1, H, W]
model.eval() # eval mode
with torch.no_grad():  # do not keep track of gradients
    for x in tqdm(test_images):
        x = test_transform_fn(x, resize_to=None) # apply test transform first. Resize to same shape model was trained on.
        print("x shape:", x.shape)
        x = torch.unsqueeze(x, 0) # unsqueeze first dim. for batch dim
        # PATCH + TEST AUGMENTATION:
        # test_pred = patch_test_augmentation(x, model=model, device=default_device, patch_size=(400, 400))
        # print("test pred shape:", test_pred.shape)
        # Standard prediction:
        x = x.to(default_device)
        # probability of pixel being 0 or 1: (sigmoid since model outputs logits)
        # test_pred = torch.sigmoid(test_pred) # SIGMOID WHEN USING TEST AUGMENTATION
        test_pred = torch.sigmoid(model(x)["out"]) # ADJUST: ["out"] only needed for Deeplabv3!. forward pass + sigmoid
        test_pred_list.append(test_pred) # append to list

# %%
# Show first and last predicted test masks and test images
show_first_last_pred(test_pred_list, test_images, first_last=5)

# %%
#  Convert model outputs to mask labels
test_pred = torch.cat(test_pred_list, 0) # concatenate to [94, 1, H, W]
test_pred = TF.resize(test_pred.detach().cpu(), (608, 608)) # resize back to original size: [94, 1, 608, 608]
# cpu is faster here
# Use Numpy below, faster than torch here
# Reshape to [94, 38, 16, 38, 16]:
test_pred = test_pred.numpy().reshape(-1, orig_size[0] // PATCH_SIZE, PATCH_SIZE, orig_size[0] // PATCH_SIZE, PATCH_SIZE)
assert (np.equal((test_pred.mean((-1, -3)) > CUTOFF), (np.moveaxis(test_pred, 2, 3).mean((-1, -2)) > CUTOFF))).all() #taking mean across -1, -3 is equal to moving axes and then taking mean...
test_pred = test_pred.mean((-1, -3)) > CUTOFF # Take mean along batches, is it above CUTOFF?
test_pred = test_pred.astype(float) #astype(float) converts Boolean to float32

# %%
# Create csv file for submission
create_submission(test_pred, test_image_paths, "deeplabv3_firstsub.csv")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %% Numpy is faster ############################################################################
# TORCH TEST -> 1.27s
# def f():
#     test_pred = torch.cat(test_pred_list, 0) # concatenate to [94, 1, H, W]
#     test_pred = TF.resize(test_pred.detach().cpu(), (608, 608)) # resize back to original size: [94, 1, 608, 608]
#     orig_size = (608, 608)
#     test_pred = test_pred.reshape(-1, orig_size[0] // PATCH_SIZE, PATCH_SIZE, orig_size[0] // PATCH_SIZE, PATCH_SIZE)
#     assert torch.equal((test_pred.mean((-1, -3)) > CUTOFF), (test_pred.moveaxis(2, 3).mean((-1, -2)) > CUTOFF))
#     test_pred = test_pred.mean((-1, -3)) > CUTOFF
#     test_pred = test_pred.float()
#     create_submission(test_pred, test_image_paths, "test.csv")
# %timeit -n 10 -r 1 f()
# # %%
# # NUMPY TEST -> 500ms
# def g():
#     test_pred = torch.cat(test_pred_list, 0) # concatenate to [94, 1, H, W]
#     test_pred = TF.resize(test_pred.detach().cpu(), (608, 608)) # resize back to original size: [94, 1, 608, 608]
#     orig_size = (608, 608)
#     test_pred = test_pred.numpy().reshape(-1, orig_size[0] // PATCH_SIZE, PATCH_SIZE, orig_size[0] // PATCH_SIZE, PATCH_SIZE)
#     # assert np.equal((test_pred.mean((-1, -3)) > CUTOFF), (test_pred.moveaxis(2, 3).mean((-1, -2)) > CUTOFF))
#     test_pred = test_pred.mean((-1, -3)) > CUTOFF
#     test_pred = np.round(test_pred).astype(float)
#     create_submission(test_pred, test_image_paths, "test2.csv")
# %timeit -n 10 -r 1 g()

# %%
# def f():
#     # since = time.time()
#     tens_conc = torch.cat(y_pred_list, 0)
#     tens_conc.shape
    # print(time.time()-since)
# %%
# %timeit -n 100 -r 10 f()
# %%
# Transform list of tensors to list of numpy arrays for faster matrix manipulation
# numpy_list = [pred.detach().cpu().numpy() for pred in y_pred_list]

# %%timeit
# def g():
#     # since = time.time()
#     numpy_conc = np.concatenate(numpy_list, 0)
#     numpy_conc.shape
#     # print(time.time()-since)

# %timeit -n 100 -r 10 g()


# %%
# TF.to_pil_image(y_pred_list[-1][0])

# %%

# np.concatenate([np.moveaxis(y_pred_list[1][0].detach().cpu().numpy(), 0, -1)] * 3, -1).shape

# # %%
# a = "text1"

# print(a)
# assert str(a) in ("song", "text1"), "not text or song string"

#%%

# import settings
# print(settings.current_settings)
# foo = settings.default_settings['foo']
# bar = settings.current_settings['bar']
# settings.current_settings['bar'] = True
# print(settings.current_settings)

# # %%

# default_settings = {'foo': True, 'bar': False}
# my_settings = {'foo': False}
# current_settings = default_settings.copy()
# current_settings.update(my_settings)
# print(default_settings)
# print(current_settings)


# #%%

# for x in model.backbone.parameters():
#     print(x)

# # %%
# train_features, train_labels = next(iter(train_dataloader))
# label = train_labels[0]
# torch.stack((label, label)).shape


# # %%
# (torch.stack((label, label)).round() == torch.stack((label, label -1)).round()).float().mean()

# # %%

# labelround = label.round().mean()

# # %%

# # print the version of CUDA being used by pytorch
# # print(torch.version.cuda)
# # !nvidia-smi


# # %%


# # for k, _  in dict1.items():
# #     print(k)

# # %%

# (torch.count_nonzero(label.round(), dim=0) == label).float().mean()

# # (label == label).float().mean()


# # %%
# (torch.stack((label, label)).round() == torch.stack((label, label)).round()).float().mean()



# # %%
# a = torch.tensor([[2., 3.], [4, 5]])
# # %%
# print("x", "\n\nb")

# # %%
# a.mean(1, keepdim=True)

# # %%
# a[::-1][-1]


# # %%

# b = {"test": [], "val": []}

# # %%

# b["test"].append(1)
# b["val"].append(1)

# # %%

# {k: sum(v) / len(v) for k, v in b.items()}

# # %%

# a = 0

# # %%

# a += torch.tensor([1])

# # %%
# a = {"test":1}
# # %%
# b = {"train": 2}
# # %%

# c = a | b
# # %%

# img, label = next(iter(train_dataloader))
# # %%

# img.shape


# # %%

# test = np.concatenate([img[0]]*10, -1)
# # %%

# # %%
# test.shape
# # %%

# plt.imshow(np.concatenate([np.moveaxis(label[0].detach().cpu().numpy(), 0, -1)] * 3, -1))


# # %%
# plt.imshow(label[0].squeeze().detach().cpu().numpy(), cmap="gray")

# # %%

# a = np.array([1, 2, 3, 4])
# # %%
# b= np.array([5, 6, 7, 8])
# # %%
# print(a.head(), b.head())
# # %%


# a = np.array(([1, 2], [3, 4]))
# # %%

# b = torch.from_numpy(a)

# # %%

