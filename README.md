## Report: Implementation of SimCLR from ICML 2020

### Introduction

For this project, I selected the paper **"SimCLR: A Simple Framework for Contrastive Learning of Visual Representations"** presented at ICML 2020. This paper proposes a framework for contrastive learning. By using data augmentations and a contrastive loss function, SimCLR achieves great results without specialized architectures or memory banks.

### Importance of the Paper

An aspect of SimCLR's success is the strategic composition of multiple data augmentation operations, such as random cropping, color distortion, and Gaussian blur. These augmentations define effective contrastive prediction tasks, enhancing the quality of the learned representations.

The key in SimCLR is the introduction of a learnable nonlinear transformation, or MLP projection head, between the representation and the contrastive loss. This transformation retains more useful information for downstream tasks. Also, SimCLR uses larger batch sizes and longer training steps compared to traditional supervised learning. This approach provides more negative examples per batch, making it easier for learning and resulting in higher-quality representations.

SimCLR also incorporates `L2` normalization and temperature scaling in the contrastive loss function (NT-Xent loss). These techniques weight different examples and contribute to the overall performance. By combining these elements, SimCLR achieves state-of-the-art results in self-supervised and semi-supervised learning on ImageNet, outperforming previous methods and even matching the performance of supervised ResNet-50 in some cases. This combination of simplicity, efficiency, and superior performance makes SimCLR a significant advancement in the field of contrastive learning.

### Implementation Details

In my implementation, I followed the methodology outlined in the SimCLR paper.

#### Data Augmentation

The data augmentation module generates different views of the same data sample using a series of random transformations. This is crucial for creating diverse and challenging positive pairs for contrastive learning.

```python
import torch
from torchvision import transforms

class Augmentations:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)
```

#### Base Encoder and Projection Head

The base encoder is a standard ResNet model with the fully connected layer replaced by an identity layer. A small projection head is added on top to map representations to a space where the contrastive loss is applied.

```python
import torch.nn as nn
from torchvision import models

class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z
```

#### Contrastive Loss Function

The NT-Xent loss function is used to maximize agreement between different views of the same image in the latent space.

```python
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim_matrix = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(N).cuda()
        labels = (labels + self.batch_size) % N
        loss = self.criterion(sim_matrix, labels)
        loss = loss / (2 * self.batch_size)
        return loss
```

#### Training

The training loop iterates over the dataset, applying augmentations, computing embeddings, and updating the model based on the contrastive loss.

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

def train_simclr(model, data_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j), _ in data_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentations = Augmentations()
    dataset = datasets.CIFAR10(root='data', train=True, transform=augmentations, download=True)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)

    base_encoder = models.resnet18(pretrained=False)
    base_encoder.fc = nn.Identity()
    model = SimCLR(base_encoder).to(device)

    criterion = NTXentLoss(batch_size=256, temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train_simclr(model, data_loader, criterion, optimizer, device)
```

