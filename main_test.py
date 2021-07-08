# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
from trainer_test import train_test
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



# Assert that images and groundtruth picture numbers in train and val paths coincide, respectively
assert [y[-7:] for y in [str(x) for x in image_paths]] == [y[-7:] for y in [str(x) for x in groundtruth_paths]]
assert len(image_paths) == 100
assert len(groundtruth_paths) == 100


# %%

# Define device "cuda" for GPU, or "cpu" for CPU
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define batch size for Dataloaders
BATCH_SIZE = 8 # not too large, causing memory issues!

# Set picture size on which model will be trained
resize_to = (400, 400)  

# ***************************************************************************
# # 1c) Load images as np.array, in a numpy array # SMALL SCALE: reload seems to be faster at training. better for small set of images.
train_images = np.stack([np.array(Image.open(img)) for img in sorted(image_paths)]) # no astype(float32)/255., as otherwise transform wouldnt work (needs PIL images, and TF.to_pil_image needs it in this format)
train_groundtruths = np.stack([np.array(Image.open(img)) for img in sorted(groundtruth_paths)])


# # For 1b), 1c): Instantiate Datasets for train and validation IMAGES
train_dataset = CustomDataset(train_images, train_groundtruths, train=True, resize_to=resize_to) # train=True


assert len(train_images) == 100
assert len(train_groundtruths) == 100
print("length train images:", len(train_images), "length groundtruth images:", len(train_groundtruths))


# %% ***************************************************************************
# Instantiate Loaders for these datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) # pin memory speeds up the host to device transfer
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


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
summary(model, (8, 3, 400, 400))


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
train_test(train_dataloader, model=model, loss_fn=loss_fn, 
             metric_fns=metric_fns, optimizer=optimizer, device=default_device, n_epochs=2, comment=comment)

# %% Save model for tinetuning conv layers
# torch.save(model.state_dict(), "state_bcedice0.5e100lr.001batch8img400+FINE.e100lr.00001.batch2img256_fullaug.pt")


#%%

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

