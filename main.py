# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import os

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from numpy.ma import concatenate

import torch
from torch import nn
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
import time
from tqdm import tqdm

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
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalize for deeplabv3
    ])

# Target Transform function
target_transform_fn = transforms.Compose(
    [
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5)) # Normalize to [-1, 1]
    ])



# %%
# Split data into training validation set
train_size = 0.9

train_image_paths, val_image_paths, train_groundtruth_paths, val_groundtruth_paths = train_test_split(
    image_paths, groundtruth_paths, random_state=random_seed, train_size=train_size, shuffle=True)

# Assert that images and groundtruth picture numbers in train and val paths coincide, respectively
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
        # im = np.moveaxis(im, -1, 0) # not needed with ToTensor() self.transforms

        # Transform to tensor, attach to device
        # Images
        if self.transforms:
            im = self.transforms(im).to(self.device)

        # Targets
        if self.target_transforms:
            gt = self.target_transforms(gt).to(self.device)
        # else:
        #     gt = torch.from_numpy(gt).to(self.device)
        #     gt = gt.unsqueeze(0) / 255. # add channel dimension of 1

        return im, gt

    def __len__(self):
        return len(self.image_paths)


# %%

# Define device "cuda" for GPU, or "cpu" for CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate Datasets for train and validation
train_dataset = CustomDataset(train_image_paths, train_groundtruth_paths, device, transform_fn, target_transforms=target_transform_fn)
val_dataset = CustomDataset(val_image_paths, val_groundtruth_paths, device, transform_fn, target_transforms=target_transform_fn)

# Instantiate Loaders for these datasets, # SET BATCH SIZE HERE
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)


# %%

# Display image and labels
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = torch.moveaxis(train_features[0], 0, -1) #Select first image out of batch, reshape CHW to HWC (plt.imshow needs shape H, W, C)
label = train_labels[0] # Select first label out of shape BHW
# Show image, .cpu() otherwise imshow doesnt work with cuda
plt.imshow(img.cpu())
plt.show()
# Squeeze only if target_transform is provided -> transformed int into torch.float [0, 1] range, and added channel
label = label.squeeze()
# Show groundtruth in greyscale (original interpretation)
plt.imshow(label.cpu(), cmap="gray") 
plt.show()

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
# # Test output with Loss functions
# print(train_features[0].shape) # We need shape [Batch, C, H, W] for model
# logits = model(train_features[0].unsqueeze(0)) # unsqueeze first dim to get Batch=1, for 1 example
# print(logits)
# target = torch.tensor([1]).unsqueeze(0).to(device) # target class 0, shape [B, scalar]



# %%
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


# Function to load deeplabv3 pretrained and change outputchannel of its classifier head to number of classes
def createDeepLabHead(outputchannels= 1):
    """ Custom Deeplab Classifier head """

    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, outputchannels) # Adjust classifier head, resnet101 has backbone output of 2048
    model.aux_classifier = FCNHead(1024, outputchannels) # Adjust aux classifier
    return model


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

# # %% 

# plt.imshow((torch.moveaxis(torch.sigmoid(out), 1, -1)).squeeze(0))

# # %%
# # Normalize to [0, 1]
# def normalize(x):
#     x = (x - np.min(x)) / (np.max(x) - np.min(x))
#     return x

# plt.imshow(normalize(np.concatenate([np.moveaxis(out.detach().numpy(), 1, -1)]*3, -1).squeeze()))

# plt.imshow(transforms.ToPILImage()(torch.cat([torch.moveaxis(out, 0, -1)]*3, 0).squeeze()))





# %%
def show_val_samples(x, y, y_hat, segmentation=False):
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
    # else:  # classification
        # fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        # for i in range(imgs_to_draw):
        #     axs[i].imshow(np.moveaxis(x[i], 0, -1))
        #     axs[i].set_title(f'True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}')
        #     axs[i].set_axis_off()
    plt.show()

# %%

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def train_epoch(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs):
    # training loop
    # logdir = './tensorboard/net'
    # writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)
    since = time.time()

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)["out"]  # forward pass #MATHIAS: ADJUSTED "OUT"
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            assert str(loss_fn) == "BCEWithLogitsLoss()" # Otherwise, torch sigmoid is not necessary here, e.g. with BCELoss  
            y_hat = torch.sigmoid(y_hat) # For metrics, torch.sigmoid needed!
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item()) # TORCH SIGMOID HERE AS WELL -> LIKE A PREDICTION
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                # probability of pixel being 0 or 1:
                y_hat = model(x)["out"] # forward pass #MATHIAS: added "out". removed torch.sigmoid -> logits are needed for loss_fn
                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                assert str(loss_fn) == "BCEWithLogitsLoss()" # Otherwise, torch sigmoid is not necessary here, e.g. with BCELoss  
                y_hat = torch.sigmoid(y_hat) # For metrics, torch.sigmoid needed!
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        # for k, v in history[epoch].items():
        #   writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# %%
metric_fns = {'acc': accuracy_fn}

# %%

# Instantiate model
model = createDeepLabHead() # function that loads pretrained deeplabv3 and changes classifier head
model.to(device) #add to gpu

# Finetuning or Feature extraction? Freeze backbone of resnet101
for x in model.backbone.parameters():
    x.requires_grad = False

# Instantiate optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Define loss function, BCEWithLogitLoss -> needs no sigmoid layer in neural net (num. stability)
loss_fn = nn.BCEWithLogitsLoss()


# %%
# Train
train_epoch(train_dataloader, eval_dataloader=val_dataloader, model=model, loss_fn=loss_fn, 
             metric_fns=metric_fns, optimizer=optimizer, n_epochs=20)

# %%

model





# *****************************************************************************************************
# %%
from torch import nn
a = nn.BCEWithLogitsLoss()


#%%

str(a)

# %%

if str(a) == "BCEWithLogitsLoss":
    print("yes")


# %%

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


# %%

b.float()


# %%
# # Training for one epoch
# def train_epoch(model, dataloader, optimizer=None, metric_fns=None, loss_fn=None, epoch=None, epochs=None): # no lr as param., will be set at optimizer instantiation in main file
#     """train one epoch"""
#     model.train() # Set in train mode

#      # initialize metric list
#     metrics = {'loss': []}
#     for k, _ in metric_fns.items():
#         metrics[k] = []
    
#     # Define progressbar for training
#     pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')

#     for features, labels in pbar:
#         optimizer.zero_grad() # Zero out gradients
#         out = model(features)["out"] # Forward pass (run feature batch through model, get prediction), ["out"] because deeplabv3 returns OrderedDict
#         loss = loss_fn(out, labels) # Calculate scalar loss
#         loss.backward() # Backward pass (backpropagate loss)
#         optimizer.step() # Take step towards negative gradient w.r.t model parameters

#          # log partial metrics
#         metrics['loss'].append(loss.item())
#         for k, fn in metric_fns.items():
#             metrics[k].append(fn(out, labels).item())
#         pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

#     # return total_loss.item()/ count, acc.item()/ count # Means over epoch, acc/loss already takes mean over batch (.item() takes value from tensor)
#     return {k: sum(v) / len(v) for k, v in metrics.items()}

# # Evaluate on val dataset
# def val(model, dataloader, metric_fns=None, loss_fn=None):
#     model.eval() # Set in eval mode, important!

#     # initialize metric list
#     metrics = {'val_loss': []}
#     for k, _ in metric_fns.items():
#         metrics['val_'+k] = []

#     with torch.no_grad(): # Stop tracking computation on tensors
#         for features, labels in tqdm(dataloader):
#             out = model(features)["out"] # Forward pass (run feature batch through model, get prediction), ["out"] because deeplabv3 returns OrderedDict
#             loss = loss_fn(out, labels) # Calculate scalar loss
#             # no backward pass!

#             # log partial metrics
#             metrics['val_loss'].append(loss.item())
#             for k, fn in metric_fns.items():
#                 metrics['val_'+k].append(fn(out, labels).item())

#     return {k: sum(v) / len(v) for k, v in metrics.items()} # Means over epoch, acc already takes mean/loss over batch (.item() takes value from tensor)


# # Full training
# def train(model, train_dataloader, val_dataloader, optimizer=None, metric_fns=None, loss_fn=None, epochs=5): # no lr as param., will be set at optimizer instantiation in main file
    
#     since = time.time()
    
#     # Initialize hist metric dict
#     history = {}  # collects metrics at the end of each epoch

#     # Run through epochs
#     for epoch in range(epochs):
#         train_metrics_dict = train_epoch(model, train_dataloader, optimizer=optimizer, metric_fns=metric_fns, loss_fn=loss_fn, epoch=epoch, epochs=epochs) # Get train batch mean results
#         val_metrics_dict = val(model, val_dataloader, metric_fns=metric_fns, loss_fn=loss_fn) # Get val batch mean results
        

#         # summarize metrics, log to tensorboard and display
#         history[epoch] = train_metrics_dict | val_metrics_dict # Taking the union of dicts, needs Python >= 3.9.0
#         # for k, v in history[epoch].items():
#         #   writer.add_scalar(k, v, epoch)
#         print(' '.join(['\n- '+str(k)+' = '+str(v) for (k, v) in history[epoch].items()]))
#         # show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

#     print('Finished Training')
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))

# # Define classification accuracy (whole image)
# def accuracy_fn(y_hat, y):
#     # computes classification accuracy
#     return (y_hat.round() == y.round()).float().mean()


# Train
# train(model, train_dataloader, val_dataloader, optimizer=optimizer, metric_fns=metric_fns, loss_fn=loss_fn, epochs=1)