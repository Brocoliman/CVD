import torch
from torch import nn

class CVDVAE(nn.Module):
    def __init__(self, img_dim=256):
        super(CVDVAE, self).__init__()

        self.num_dist = 16
        self.pprp = 16*62*62
        self.pre_reparam = 512
        self.img_dim = img_dim

        # Encoder: 
        self.encode_seq = nn.Sequential(
            nn.Upsample(size=(img_dim,img_dim),mode='bilinear'),

            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),  # 3,256,256 -> 6,254,254
            nn.ReLU(),


            nn.MaxPool2d(2, 2),  # -> 6,127,127
            
            nn.Conv2d(6, 16, kernel_size=3),  # -> 16,125,125
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 16,62,62
            
            nn.Flatten(),
            nn.Linear(self.pprp, self.pre_reparam),  # -> prp
            nn.ReLU(),
        )

        self.mu = nn.Linear(self.pre_reparam, self.num_dist)  # mu
        self.sigma = nn.Linear(self.pre_reparam, self.num_dist)  # logvar

        # Decoder:
        self.decode_seq1 = nn.Sequential(
            nn.Linear(self.num_dist, self.pre_reparam),  # -> prp
            nn.ReLU(),
            
            nn.Linear(self.pre_reparam, self.pprp), # -> 16,62,62
            nn.ReLU(),
        )
        self.decode_seq2 = nn.Sequential(
            nn.Upsample(size=125),  # -> 16,125,125
            nn.ConvTranspose2d(16, 6, kernel_size=3),  # -> 6,127,127
            nn.ReLU(),
            
            nn.Upsample(size=254),  # -> 6,254,254
            nn.ConvTranspose2d(6, 3, kernel_size=3),  # -> 3,256,256
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
        h = h.view(-1, 16, 62, 62)
        return self.decode_seq2(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar