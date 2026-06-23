import torch
import matplotlib.pyplot as plt
from model import UNet, NoiseSchedule, sample
import json

T = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noise_schedule = NoiseSchedule(T).to(device)
unet = UNet().to(device)
unet.load_state_dict(torch.load('best_ddpm_weights.pth'))
unet.eval()

@torch.no_grad()
def sample_with_intermediates(unet, noise_schedule, device='cuda'):
    x = torch.randn(1, 1, 28, 28).to(device)
    intermediates = []
    save_at = [999, 750, 500, 250, 100, 0]
    
    for t in reversed(range(noise_schedule.T)):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        predicted_noise = unet(x, t_batch)
        alpha = noise_schedule.alphas[t]
        alpha_bar = noise_schedule.alpha_bars[t]
        beta = noise_schedule.betas[t]
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise) + torch.sqrt(beta) * z
        
        if t in save_at:
            img = (x.clamp(-1, 1) + 1) / 2
            intermediates.append((t, img.squeeze().cpu()))
    
    return intermediates

# denoising visualization
intermediates = sample_with_intermediates(unet, noise_schedule, device=device)
fig, axes = plt.subplots(1, len(intermediates), figsize=(12, 2))
for i, (t, img) in enumerate(intermediates):
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f't={t}', fontsize=8)
plt.suptitle('Denoising Process', fontsize=12)
plt.tight_layout()
plt.savefig('results/denoising_process.png')
plt.close()


# generate images
images = sample(unet, noise_schedule, n_images=8, device=device)

# rescale from [-1,1] to [0,1] for display
images = (images.clamp(-1, 1) + 1) / 2

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    axes[i].imshow(images[i].cpu().squeeze(), cmap='gray')
    axes[i].axis('off')
axes[0].set_title('Generated', loc='left', fontsize=12)
plt.tight_layout()
plt.savefig('results/generated.png')
plt.close()

# loss curve
with open('losses.json', 'r') as f:
    losses = json.load(f)

plt.figure(figsize=(8, 4))
plt.plot(losses['train'], label='Train Loss')
plt.plot(losses['val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('DDPM Training and Validation Loss on MNIST')
plt.legend()
plt.savefig('results/training_loss.png')
plt.close()

print('Done.')