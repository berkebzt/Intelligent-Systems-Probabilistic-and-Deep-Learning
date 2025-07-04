
import torch
import torch.nn as nn

# Same architecture as original model
class FaceAutoencoder(nn.Module):
    """Convolutional autoencoder with 16-D bottleneck for face image compression."""
    def __init__(self):
        super().__init__()
        # Constants
        IMG_H, IMG_W, LATENT = 192, 160, 16
        h, w = IMG_H // 16, IMG_W // 16  # four 2× downsamples
        self.h, self.w = h, w  # Store for reshape in decode

        # Encoder: 1→32→64→128→256 channels
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,256,3,2,1),nn.BatchNorm2d(256),nn.ReLU(),
        )
        # Bottleneck
        self.fc1 = nn.Linear(256*h*w, LATENT)
        self.fc2 = nn.Linear(LATENT, 256*h*w)
        # Decoder: mirror
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),   nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1),    nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decode(self, z):
        z = self.fc2(z).view(-1, 256, self.h, self.w)
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

def encode(images):
    """
    Encode face images to a 16D latent representation.

    Args:
        images: A B×1×192×160 PyTorch tensor containing grayscale face images.
                Intensity values are in range [0, 1].

    Returns:
        latents: A B×16 PyTorch tensor containing the encoded latents.
    """
    # Load model and weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceAutoencoder().to(device)
    model.load_state_dict(torch.load('face_autoencoder.pth', map_location=device))
    model.eval()

    # Move images to device and encode
    images = images.to(device)
    with torch.no_grad():
        latents = model.encode(images)

    return latents

def decode(latents):
    """
    Decode latent representations back to face images.

    Args:
        latents: A B×16 PyTorch tensor containing latent representations.

    Returns:
        images: A B×1×192×160 PyTorch tensor containing reconstructed face images.
    """
    # Load model and weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceAutoencoder().to(device)
    model.load_state_dict(torch.load('face_autoencoder.pth', map_location=device))
    model.eval()

    # Move latents to device and decode
    latents = latents.to(device)
    with torch.no_grad():
        images = model.decode(latents)

    return images
