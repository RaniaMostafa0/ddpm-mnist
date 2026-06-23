import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import UNet, NoiseSchedule, sample
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = 1000
noise_schedule = NoiseSchedule(T).to(device)
unet = UNet().to(device)
unet.load_state_dict(torch.load('best_ddpm_weights.pth'))
unet.eval()

def preprocess_for_fid(images):
    images = images.repeat(1, 3, 1, 1)          # grayscale -> 3 channels (fake RGB)
    images = F.interpolate(images, size=(75, 75), mode='bilinear', align_corners=False)
    images = (images * 255).to(torch.uint8)     # [0,1] float -> [0,255] uint8
    return images

fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        real = preprocess_for_fid(images)
        fid.update(real, real=True)

with torch.no_grad():
    for _ in tqdm(range(40), desc='Generating for FID'):
        gen = sample(unet, noise_schedule, n_images=256, device=device)
        gen = gen.clamp(-1, 1)
        gen = (gen + 1) / 2                    # rescale to [0,1]
        gen = gen.repeat(1, 3, 1, 1)           # grayscale to RGB
        gen = F.interpolate(gen, size=(75, 75), mode='bilinear', align_corners=False)
        gen = (gen * 255).to(torch.uint8)
        fid.update(gen, real=False)

fid_value = fid.compute().item()
print(f'DDPM FID Score: {fid_value:.4f}')

with open('results/fid_score.json', 'w') as f:
    json.dump({'model': 'DDPM', 'fid_score': fid_value}, f)
print('FID score saved.')