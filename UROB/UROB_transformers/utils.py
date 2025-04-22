import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

classes = ('AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake')

class EuroSatDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data = []
        self.targets = []
        self.classes = classes
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.load_data()

    def load_data(self):
        path = os.path.join(self.root, self.split)
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(path, class_name)
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                self.data.append(image_path)
                self.targets.append(i)

    def __getitem__(self, index):
        image_path, target = self.data[index], self.targets[index]
        image = Image.open(image_path)
        # image = image.convert('RGB')
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, target
    
    def __len__(self):
        return len(self.data)


@torch.no_grad()
def visualize_attention(vit_classifier, dataset, output=None, device="cuda"):
    """
    Visualize the attention map for a single random image.
    Args:
        vit_classifier: The ViT model
        dataset: Dataset containing images
        output: Optional path to save visualization
        device: Device to run inference on
    """
    vit_classifier.eval()
    vit_classifier.to(device)

    # Get single random sample
    idx = torch.randint(len(dataset), (1,)).item()
    image, label = dataset[idx]
    
    # Process image
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    raw_image = image.permute(1, 2, 0).cpu().numpy()
    
    # Prepare image tensor
    image_tensor = image.float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    
    # Get prediction and attention maps
    logits, attention_maps = vit_classifier(image_tensor)
    prediction = torch.argmax(logits, dim=1).item()

    # Process attention maps
    attention_maps = torch.stack(attention_maps)  # [num_layers, 1, num_heads, seq_len, seq_len]
    attention_maps = attention_maps.mean(dim=[0, 2])  # Average across layers and heads [1, seq_len, seq_len]
    attention_maps = attention_maps[0, 0, 1:]  # Get CLS token attention [num_patches]

    # Reshape to square grid
    patch_size = int(math.sqrt(attention_maps.size(-1)))
    attention_maps = attention_maps.view(patch_size, patch_size)

    # Resize to match image size
    attention_maps = F.interpolate(
        attention_maps.unsqueeze(0).unsqueeze(0),
        size=(64, 64),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # reordering the image tensor
    raw_image = raw_image.transpose(2, 0, 1)
    # Original image
    ax1.imshow(raw_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Attention overlay
    ax2.imshow(raw_image)
    ax2.imshow(attention_maps.cpu().numpy(), alpha=0.5, cmap='jet')
    gt = classes[label]
    pred = classes[prediction]
    ax2.set_title(f'Attention Map\nGT: {gt}\nPred: {pred}', 
                color=('green' if gt == pred else 'red'))
    ax2.axis('off')

    plt.tight_layout()
    if output:
        plt.savefig(output, bbox_inches='tight', dpi=300)
    plt.show()



if __name__ == "__main__":
    dataset = EuroSatDataset(root='EuroSAT_RGB', split='train')
    print("Number of samples in the dataset: ", len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    x, label = next(iter(train_loader))
    # plot the image and labels
    plt.figure(figsize=(10, 10))    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(x[i])
        plt.title(classes[label[i].item()])
    plt.show()


