# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import os

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import torch
from torch import optim
from torch.nn.modules import loss
import torchvision
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

# Image Transform function
target_transform_fn = transforms.Compose(
    [transforms.ToTensor(),
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

# Instantiate Loaders for these datasets, # SET BATCH SIZE HERE
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)


# %%

# # Display image and labels
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = torch.moveaxis(train_features[0], 0, -1) #Select first image out of batch, reshape CHW to HWC (plt.imshow needs shape H, W, C)
# label = train_labels[0] # Select first label out of shape BHW
# # Show image, .cpu() otherwise imshow doesnt work with cuda
# plt.imshow(img.cpu())
# plt.show()
# # Squeeze only if target_transform is provided -> transformed int into torch.float [0, 1] range, and added channel
# label = label.squeeze()
# # Show groundtruth in greyscale (original interpretation)
# plt.imshow(label.cpu(), cmap="gray") 
# plt.show()

# %%

# Define model
from torch import nn

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
# model = FirstNet().to(device) #add to GPU
# print(model) # Look at structure of model


# %%
# # Test output with Loss functions
# print(train_features[0].shape) # We need shape [Batch, C, H, W] for model
# logits = model(train_features[0].unsqueeze(0)) # unsqueeze first dim to get Batch=1, for 1 example
# print(logits)
# target = torch.tensor([1]).unsqueeze(0).to(device) # target class 0, shape [B, scalar]



# %%
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


# Function to load deeplabv3 pretrained and change outputchannel of its classifier head to number of classes
def createDeepLabHead(outputchannels= 1):
    """ Custom Deeplab Classifier head """

    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, outputchannels) # Adjust classifier head, resnet101 has backbone output of 2048

    return model


# %%
# Predict one instance, without learning anything
# model.to(device)
# model.eval()
# img = train_features[0] # Take first image from DataLoader
# img = img.unsqueeze(0) # Unsqueeze for batch size = 1 -> [1, C, H, W]
# print(img.shape)
# out = model(img)["out"].cpu()
# plt.imshow(out.squeeze().detach().numpy(), cmap="gray")

# %%

# Training for one epoch
def train_epoch(model, dataloader, optimizer=None, loss_fn=None): # no lr as param., will be set at optimizer instantiation in main file
    """train one epoch"""
    model.train() # Set in train mode
    total_loss, acc, count = 0, 0, 0 # Initialize train metrics
    for features, labels in tqdm(dataloader):
        optimizer.zero_grad() # Zero out gradients
        out = model(features)["out"] # Forward pass (run feature batch through model, get prediction), ["out"] because deeplabv3 returns OrderedDict
        loss = loss_fn(out, labels) # Calculate scalar loss
        loss.backward() # Backward pass (backpropagate loss)
        optimizer.step() # Take step towards negative gradient w.r.t model parameters

        # Collect metrics of total batch
        total_loss += loss
        acc += (out.round() == labels.round()).float().mean() # Round to closest integer, compare Bool to Float, take mean
        count += 1 # since loss and acc are already means over batch
        print("count train:", count)
        print("accuracy train:", acc.item())
    return total_loss.item()/ count, acc.item()/ count # Means over epoch, acc/loss already takes mean over batch (.item() takes value from tensor)


# Evaluate on val dataset
def val(model, dataloader, loss_fn=None):
    model.eval() # Set in eval mode, important!
    loss, acc, count = 0, 0, 0 # Initialize val metrics
    with torch.no_grad(): # Stop tracking computation on tensors
        for features, labels in tqdm(dataloader):
            out = model(features)["out"] # Forward pass (run feature batch through model, get prediction), ["out"] because deeplabv3 returns OrderedDict
            loss += loss_fn(out, labels) # Calculate scalar loss
            # no backward pass!

            # Metrics
            acc += (out.round() == labels.round()).float().mean() # Round to closest integer, compare Bool to Float, take mean
            count += 1 # since loss and acc are already means over batch
            print("count val:", count)
            print("accuracy val:", acc.item())
    return loss.item()/ count, acc.item()/ count # Means over epoch, acc already takes mean/loss over batch (.item() takes value from tensor)


# Full training
def train(model, train_dataloader, val_dataloader, optimizer=None, loss_fn=None, epochs=5): # no lr as param., will be set at optimizer instantiation in main file
    # Initialize metric dict
    res = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Run through epochs
    for epoch in range(epochs):
        tl, ta = train_epoch(model, train_dataloader, optimizer=optimizer, loss_fn=loss_fn) # Get train batch mean results
        vl, va = val(model, val_dataloader, loss_fn=loss_fn) # Get val batch mean results
        
        # Print losses and metrics (used f strings: :2 -> integer, .3f -> 3 decimal floats)
        print(f"Epoch: {epoch+1:2}/{epochs:2}, Train loss={tl:.3f}, Train acc={ta:.3f}, Val loss={vl:.3f}, Val acc={va:.3f}")
        # Collect losses and metric in metric dict
        res["train_loss"].append(tl)
        res["train_acc"].append(ta)
        res["val_loss"].append(vl)
        res["val_acc"].append(va)
    return res

        
# %%

# Instantiate model
model = createDeepLabHead() # function that loads pretrained deeplabv3 and changes classifier head
model.to(device) #add to gpu

# Instantiate optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# Define loss function, BCEWithLogitLoss -> needs no sigmoid layer in neural net (num. stability)
loss_fn = nn.BCELoss()

# Finetuning or Feature extraction? Freeze backbone of resnet101
for x in model.backbone.parameters():
    x.requires_grad = False

# %% 
# Train
train(model, train_dataloader, val_dataloader, optimizer=optimizer, loss_fn=loss_fn)



# %%

for x in model.parameters():
    x.requires_grad = False


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

import torch
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
a

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

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def train_tutorial(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs):
    # training loop
    # logdir = './tensorboard/net'
    # writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

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
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = model(x)["out"] # forward pass #MATHIAS: ADJUSTED "OUT"
                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        # for k, v in history[epoch].items():
        #   writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        # show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
# %%
metric_fns = {'acc': accuracy_fn}

train_tutorial(train_dataloader, eval_dataloader=val_dataloader, model=model, loss_fn=loss_fn, 
             metric_fns=metric_fns, optimizer=optimizer, n_epochs=5)


# %%

a = 0

# %%

a += torch.tensor([1])

# %%
a