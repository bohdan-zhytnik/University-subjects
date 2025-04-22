import torch, torch.nn as nn, torch.functional as F
import numpy as np
import torchvision
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

image_transform = transforms.Compose([
                        transforms.Resize(128),
                        transforms.CenterCrop(128),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])

mask_transform = transforms.Compose([
                        transforms.Resize(128),
                        transforms.CenterCrop(128),
                    ])

class PetsDataset(Dataset):
    def __init__(self, dataset_path):
        self.image_paths = sorted(list(Path(dataset_path / "images").iterdir())) # list of paths to individual images
        self.map_paths = sorted(list(Path(dataset_path / "trimaps").iterdir())) # list of paths to individual annotation files

        assert len(self.image_paths) == len(self.map_paths)
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image_transform(image)
        
        mask = Image.open(self.map_paths[idx])
        mask = mask_transform(mask)
        mask = torch.tensor(np.array(mask)).long() - 1
        return image, mask
    
dataset_path = Path("/mnt/datasets/urob/pets/")
dataset = PetsDataset(dataset_path)

val_set_coef = 0.1
trainset_size = int(len(dataset) - val_set_coef * len(dataset))
trainset, validset = torch.utils.data.random_split(dataset, [trainset_size, len(dataset) - trainset_size])
print(f"trainset_sz={len(trainset)}, validset_sz={len(validset)}")

train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)
valid_loader = DataLoader(validset, batch_size=8, shuffle=True, num_workers=4)

x, mask = next(iter(train_loader))
x.shape, mask.shape

from torchvision.utils import make_grid

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def mask_to_img(mask):
    mask_img = torch.zeros((mask.shape[0], 128, 128, 3)).to(device)
    mask_img[mask == 0] = torch.tensor([1., 0., 0.], device=device)
    mask_img[mask == 1] = torch.tensor([0., 1., 0.], device=device)
    mask_img = mask_img.permute(0, 3, 1, 2)
    return mask_img

to_viz = torch.cat([x, mask_to_img(mask)], dim=0)

imshow(make_grid(to_viz))

# Transposed Convolution
# PyTorch LEGO (for segmentation)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
    def forward(self, x):
        return self.layer(x)

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.ReLU())
    def forward(self, x):
        return self.layer(x)

class TransConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TransConvBlock, self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                                   nn.ReLU())
    def forward(self, x):
        return self.layer(x)

print(ConvBlock(3, 6))
print(MLPBlock(32, 64))
print(TransConvBlock(32, 64))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128)
        )

        self.mlp = nn.Sequential(
            MLPBlock(16*16*128, 128),
            MLPBlock(128, 256),
            nn.Linear(256, 37)
        )

        self.segment_head = nn.Sequential(
            TransConvBlock(128, 64),
            TransConvBlock(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features_flat = torch.flatten(features, 1)
        print("features: ", features.shape)
        class_pred = self.mlp(features_flat)

        mask_pred = self.segment_head(features)
        print(mask_pred.shape)
        
        return class_pred, mask_pred

print(Net())

net = Net()
breed_pred, mask_pred = net(torch.randn(8, 3, 128, 128))
print(f"{breed_pred.shape=}, {mask_pred.shape=}")

mask.shape, mask_pred.shape

# how we compute classification loss
loss_fn = nn.CrossEntropyLoss()
fake_label = torch.zeros((8,)).long()
classif_loss = loss_fn(breed_pred, fake_label)
print(f"loss=", classif_loss)

# how to compute mask loss
mask_loss = loss_fn(mask_pred, mask)
print(f"loss=", mask_loss)
