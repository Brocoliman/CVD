import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, img_dim=256):
        super(VAE, self).__init__()

        self.num_dist = 16
        self.pre_reparam = 512
        self.img_dim = img_dim

        # Encoder: 
        self.encode_seq = nn.Sequential(
            nn.Upsample(size=(img_dim,img_dim),mode='bilinear'),

            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),  # -> 6x[d-2]^2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 6x[(d-2)/2]^2
            
            nn.Conv2d(6, 16, kernel_size=3),  # -> 16x[(d-6)/2]^2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 16x[(d-6)/4]^2
            
            nn.Flatten(),
            nn.Linear(16 * ((img_dim-4)/4)**2, self.pre_reparam),  # -> prp
            nn.ReLU(),
        )

        self.mu = nn.Linear(self.pre_reparam, self.num_dist)  # mu
        self.sigma = nn.Linear(self.pre_reparam, self.num_dist)  # logvar

        # Decoder:
        self.decode_seq1 = nn.Sequential(
            nn.Linear(self.num_dist, self.pre_reparam),  # -> prp
            nn.ReLU(),
            
            nn.Linear(self.pre_reparam, 16 * ((img_dim-4)/4)**2), # -> 6x[(d-6)/4]^2
            nn.ReLU(),
        )
        self.decode_seq2 = nn.Sequential(
            nn.Upsample(size=11),  # 16x11x11
            nn.ConvTranspose2d(16, 6, kernel_size=3),  # -> 6x13x13
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),  # -> 6x26x26
            nn.ConvTranspose2d(6, 1, kernel_size=3),  # -> 1x28x28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encode_seq(x)
        return self.mu(h), self.sigma(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decode_seq1(z)
        h = h.view(-1, 16, 5, 5)
        return self.decode_seq2(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar