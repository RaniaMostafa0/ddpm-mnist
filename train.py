import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import UNet, NoiseSchedule
from tqdm import tqdm
import json

# hyperparameters
T = 1000
epochs = 100
batch_size = 128
lr = 2e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def add_noise(x0, t, noise_schedule):
    noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(noise_schedule.alpha_bars[t])[:, None, None, None]
    sqrt_1_ab = torch.sqrt(1 - noise_schedule.alpha_bars[t])[:, None, None, None]
    x_t = sqrt_ab * x0 + sqrt_1_ab * noise
    return x_t, noise

noise_schedule = NoiseSchedule(T).to(device)
unet = UNet().to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

def train_one_epoch():
    unet.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')

    for batch, _ in progress_bar:
        batch = batch.to(device)
        t = torch.randint(0, T, (batch.size(0),), device=device)
        x_t, noise = add_noise(batch, t, noise_schedule)
        predicted_noise = unet(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(train_loader)

def validate():
    unet.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, _ in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)
            t = torch.randint(0, T, (batch.size(0),), device=device)
            x_t, noise = add_noise(batch, t, noise_schedule)
            predicted_noise = unet(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)
            total_loss += loss.item()
    return total_loss / len(val_loader)

all_train_losses = []
all_val_losses = []
best_val_loss = float('inf')

for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)

    print(f'Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(unet.state_dict(), 'best_ddpm_weights.pth')
        print(f'Best model saved at epoch {epoch+1}')

torch.save(unet.state_dict(), 'final_ddpm_weights.pth')
with open('losses.json', 'w') as f:
    json.dump({'train': all_train_losses, 'val': all_val_losses}, f)
print('Done.')