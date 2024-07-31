import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Data Augmentation
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

# Base Encoder and Projection Head
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        in_features = self.encoder.fc.in_features  # Get in_features before setting fc to Identity
        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

# NT-Xent Loss
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

# Training
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

    # Data
    augmentations = Augmentations()
    dataset = datasets.CIFAR10(root='data', train=True, transform=augmentations, download=True)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    # Model
    base_encoder = models.resnet50(weights=None)
    model = SimCLR(base_encoder).to(device)

    # Loss and Optimizer
    criterion = NTXentLoss(batch_size=64, temperature=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    train_simclr(model, data_loader, criterion, optimizer, device, epochs=20)
