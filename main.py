# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from pathlib import Path
import time
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Local Module imports
from dataset import CustomDataset
from diceloss import BinaryDiceLoss_Logits
from model import createDeepLabHead
from trainer import train_epoch
from metrics import accuracy_fn, patch_accuracy

# %%
# Define random seeds
random_seed = 343
#
torch.manual_seed(random_seed)
random.seed(random_seed) # also needed for transforms.RandomRotation.get_params...
np.random.seed(random_seed) # global numpy RNG

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


# %%

# Define device "cuda" for GPU, or "cpu" for CPU
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define batch size for Dataloaders
BATCH_SIZE = 8 # cannot be

# Instantiate Datasets for train and validation
train_dataset = CustomDataset(train_image_paths, train_groundtruth_paths, train=True) # train=True
val_dataset = CustomDataset(val_image_paths, val_groundtruth_paths, train=False) # train=False

# Instantiate Loaders for these datasets, # SET BATCH SIZE HERE
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) # BATCH SIZE
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) # BATCH SIZE


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


#%%


label = (label > 0.9).float()
plt.imshow(label, cmap="gray") 
plt.show()


# %%

# Test for image transformations:
image = torch.from_numpy(np.moveaxis(np.array(Image.open(image_paths[0])), -1, 0))
# label = torch.unsqueeze(torch.from_numpy(np.array(Image.open(groundtruth_paths[0]))), dim=0)

print(image.shape)

# print(label.shape)

# transform = transforms.Compose([transforms.RandomRotation(90)])
# image = transform(image)

# image = transforms.functional.adjust_brightness(image, brightness_factor=3.)
# image = transforms.functional.adjust_contrast(image, contrast_factor=3)
# image = transforms.functional.adjust_hue(image, hue_factor=0.5)
# image = transforms.functional.adjust_saturation(image, saturation_factor=3)
# image = transforms.functional.adjust_sharpness(image, sharpness_factor=5)
# image = transforms.functional.equalize(image)

# i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.3, 0.3), ratio=(0.75, 1.33))
# print(i, j, h, w)
# image = transforms.functional.resized_crop(image, i, j, h, w, (400, 400))
# image = transforms.functional.resize(image, (224, 224))
# image = transforms.functional.center_crop(image, (224, 224))

# image = TF.vflip(image)

plt.imshow(torch.moveaxis(image.cpu(), 0, -1))

print(image)

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

# transforms.RandomResizedCrop.get_params(image, scale=(0.08, 1.0), ratio=(0.75, 1.3333))

# %%

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
# from torchinfo import summary ->have to install with Pip, ideally after every other necessary package is installed with conda
torch.empty(3).random_(2)


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
# Predict one instance, without learning anything
# model = createDeepLabHead()
# model.to(device)
# model.eval()
# img = train_features[0] # Take first image from DataLoader
# img = img.unsqueeze(0) # Unsqueeze for batch size = 1 -> [1, C, H, W]
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




# %%
# Define global variables
PATCH_SIZE = 16
CUTOFF = 0.25

# Instantiate model
model = createDeepLabHead() # function that loads pretrained deeplabv3 and changes classifier head
model.to(default_device) #add to gpu

# Finetuning or Feature extraction? Freeze backbone of resnet101
for x in model.backbone.parameters():
    x.requires_grad = False

# Instantiate optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Define loss function, BCEWithLogitLoss -> needs no sigmoid layer in neural net (num. stability)
# loss_fn = nn.BCEWithLogitsLoss()
# from diceloss import BinaryDiceLoss_Logits
loss_fn = BinaryDiceLoss_Logits()

# patch_accuracy = patch_accuracy(y_hat, y, patch_size=16, cutoff=0.5)

# Define metrics
metric_fns = {'acc': accuracy_fn, "patch_acc": patch_accuracy}

# %%
# Train
train_epoch(train_dataloader, eval_dataloader=val_dataloader, model=model, loss_fn=loss_fn, 
             metric_fns=metric_fns, optimizer=optimizer, device=default_device, n_epochs=3)

# %%

# dict1 = {"test": 1, "train": 2}

# # %%
# for k, v in dict1.items():
#     print(k, v)


# *****************************************************************************************************
# %%
a = "text1"

print(a)
assert str(a) in ("song", "text1"), "not text or song string"

# %%


#%%

for x in model.backbone.parameters():
    print(x)

# %%
train_features, train_labels = next(iter(train_dataloader))
label = train_labels[0]
torch.stack((label, label)).shape


# %%
(torch.stack((label, label)).round() == torch.stack((label, label -1)).round()).float().mean()

# %%

labelround = label.round().mean()

# %%

# print the version of CUDA being used by pytorch
# print(torch.version.cuda)
# !nvidia-smi


# %%


# for k, _  in dict1.items():
#     print(k)

# %%

(torch.count_nonzero(label.round(), dim=0) == label).float().mean()

# (label == label).float().mean()


# %%
(torch.stack((label, label)).round() == torch.stack((label, label)).round()).float().mean()



# %%
a = torch.tensor([[2., 3.], [4, 5]])
# %%
print("x", "\n\nb")

# %%
a.mean(1, keepdim=True)

# %%
a[::-1][-1]


# %%

b = {"test": [], "val": []}

# %%

b["test"].append(1)
b["val"].append(1)

# %%

{k: sum(v) / len(v) for k, v in b.items()}

# %%

a = 0

# %%

a += torch.tensor([1])

# %%
a = {"test":1}
# %%
b = {"train": 2}
# %%

c = a | b
# %%

img, label = next(iter(train_dataloader))
# %%

img.shape


# %%

test = np.concatenate([img[0]]*10, -1)
# %%

# %%
test.shape
# %%

plt.imshow(np.concatenate([np.moveaxis(label[0].detach().cpu().numpy(), 0, -1)] * 3, -1))


# %%
plt.imshow(label[0].squeeze().detach().cpu().numpy(), cmap="gray")

# %%

a = np.array([1, 2, 3, 4])
# %%
b= np.array([5, 6, 7, 8])
# %%
print(a.head(), b.head())
# %%


a = np.array(([1, 2], [3, 4]))
# %%

b = torch.from_numpy(a)

