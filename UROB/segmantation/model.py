import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

# Image and mask transformations (same as during training)
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
        self.image_paths = sorted(list(Path(dataset_path / "images").iterdir()))
        self.map_paths = sorted(list(Path(dataset_path / "trimaps").iterdir()))

        # Species names
        self.species_names = ["dog", "cat"]

        # Species classes: 0 for dog, 1 for cat
        self.species_classes = [0 if image_path.name[0].islower() else 1 for image_path in self.image_paths]

        # Extract breed names
        breed_names = [image_path.stem.rsplit('_', 1)[0] for image_path in self.image_paths]
        self.breed_names = sorted(set(breed_names))

        # Separate dog and cat breeds
        self.dog_breed_names = sorted([name for name in self.breed_names if name[0].islower()])
        self.cat_breed_names = sorted([name for name in self.breed_names if name[0].isupper()])

        # Mapping from breed names to indices
        self.breed_name2idx = {breed: idx for idx, breed in enumerate(self.breed_names)}
        self.breed_classes = [self.breed_name2idx[breed] for breed in breed_names]

        assert len(self.image_paths) == len(self.species_classes), \
            f"Number of images and species_classes do not match: {len(self.image_paths)} != {len(self.species_classes)}"
        assert len(self.image_paths) == len(self.breed_classes), \
            f"Number of images and breeds do not match: {len(self.image_paths)} != {len(self.breed_classes)}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load the image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image_transform(image)

        # 2. Load the mask
        mask = Image.open(self.map_paths[idx])
        mask = mask_transform(mask)
        mask = torch.tensor(np.array(mask)).long() - 1  # Adjust labels to 0-based

        # 3. Class tensors for classification
        species_tensor = torch.tensor(self.species_classes[idx])
        breed_tensor = torch.tensor(self.breed_classes[idx])

        return image, species_tensor, breed_tensor, mask

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the model architecture
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Species classifier
        self.species_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

        # Breed classifier
        self.breed_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 37)  # Total number of breeds
        )

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 3, kernel_size=1)
            # nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
        )

        # idx to string mappings for the predict method
        self.idx2species = {0: 'dog', 1: 'cat'}
        # self.idx2breed = {idx: breed for breed, idx in breed_name2idx.items()}
        # print('idx2breed',self.idx2breed)
        self.idx2breed = {0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 4: 'British_Shorthair', 
                            5: 'Egyptian_Mau', 6: 'Maine_Coon', 7: 'Persian', 8: 'Ragdoll', 9: 'Russian_Blue', 
                            10: 'Siamese', 11: 'Sphynx', 12: 'american_bulldog', 13: 'american_pit_bull_terrier', 
                            14: 'basset_hound', 15: 'beagle', 16: 'boxer', 17: 'chihuahua', 18: 'english_cocker_spaniel', 
                            19: 'english_setter', 20: 'german_shorthaired', 21: 'great_pyrenees', 22: 'havanese', 
                            23: 'japanese_chin', 24: 'keeshond', 25: 'leonberger', 26: 'miniature_pinscher', 
                            27: 'newfoundland', 28: 'pomeranian', 29: 'pug', 30: 'saint_bernard', 31: 'samoyed', 32: 'scottish_terrier', 
                            33: 'shiba_inu', 34: 'staffordshire_bull_terrier', 35: 'wheaten_terrier', 36: 'yorkshire_terrier'}

        # Map breed indices to species indices
        # self.breed_idx2species = {}
        # for breed_name, idx in breed_name2idx.items():
        #     if breed_name[0].islower():
        #         self.breed_idx2species[idx] = 0  # dog
        #     else:
        #         self.breed_idx2species[idx] = 1  # cat
        # print('breed_idx2species',self.breed_idx2species)
        self.breed_idx2species = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 0, 
                                  13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 
                                  25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0}

    def forward(self, x):
        # Implement the forward pass
        features = self.backbone(x)
        species_pred = self.species_classifier(features)
        breed_pred = self.breed_classifier(features)
        mask_pred = self.segmentation_head(features)
        return species_pred, breed_pred, mask_pred

    def predict(self, image):
        """
        Receives an image and returns predictions
        input: image (torch.Tensor) - C x H x W
        output: species_pred (string), breed_pred (tuple of strings), mask_pred (torch.Tensor) - H x W
        """
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            species_logits, breed_logits, mask_logits = self.forward(image)

            species_pred_idx = species_logits.argmax(dim=1).item()
            species_pred_class = self.idx2species[species_pred_idx]

            breed_probs = torch.softmax(breed_logits, dim=1).squeeze(0)  # Shape: [num_breeds]
            # print('breed_probs',breed_probs)
            # print('max', torch.max(breed_probs))
            # print('argmax', torch.argmax(breed_probs))
            
            breed_indices = torch.arange(len(breed_probs)).to(device)
            breed_species_mask = torch.tensor([self.breed_idx2species[idx.item()] == species_pred_idx for idx in breed_indices]).to(device)
            breed_probs_filtered = breed_probs * breed_species_mask

            top3_probs, top3_idx = torch.topk(breed_probs_filtered, k=3)
            
            top3_breed_classes = [self.idx2breed[idx.item()] for idx in top3_idx]
            breed_pred_classes = tuple(top3_breed_classes)

            mask_pred = torch.argmax(mask_logits, dim=1).squeeze(0).cpu()  # Shape: [H, W]

        return species_pred_class, breed_pred_classes, mask_pred