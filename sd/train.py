import torch
from torch import nn
from torch.nn import functional as F
from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder

import torch
from torch import nn
from torch.utils.data import DataLoader # Recommended over raw Dataset

class VAE_Trainer():
    def __init__(self, lr=1e-4, sigma=1.0):
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
        self.sigma = sigma

    def train_epoch(self, dataloader):
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        
        for batch in dataloader:
            x = batch['image'].to(self.device)
            n, c, h, w = x.shape
            
            # noise: (Batch_size, 4, Width/8, Height/8)
            noise = torch.randn(n, 4, h//8, w//8).to(self.device)
            # Forward pass
            z, mean, stdev = self.encoder(x, noise)
            x_hat = self.decoder(z)
            
            # Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(x_hat, x, reduction='sum') / (2 * self.sigma**2)
            
            # KL Divergence: 0.5 * sum(mean^2 + stdev^2 - log(stdev^2) - 1)
            kl_loss = 0.5 * torch.sum(mean**2 + stdev**2 - torch.log(stdev**2) - 1)
            
            loss = (recon_loss + kl_loss) / n
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        
        for batch in dataloader:
            x = batch['image'].to(self.device)
            n, c, h, w = x.shape
            noise = torch.randn(n, 4, h//8, w//8).to(self.device)

            z, mean, stdev = self.encoder(x, noise)
            x_hat = self.decoder(z)
            
            # Use same logic as training for consistency
            recon_loss = F.mse_loss(x_hat, x, reduction='sum') / (2 * self.sigma**2)
            kl_loss = 0.5 * torch.sum(mean**2 + stdev**2 - torch.log(stdev**2) - 1)
            
            total_loss += (recon_loss + kl_loss).item() / x.size(0)
            
        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, epochs: int):
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('models/best_models/')
            print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    @torch.no_grad()
    def generate(self, z: torch.Tensor):
        x_hat = self.decoder(z)
        generated_image = x_hat + self.sigma * torch.randn_like(x_hat)
        return generated_image
    
    def save_model(self, path: str):
        torch.save(self.encoder.state_dict(), path + 'encoder.pth')
        torch.save(self.decoder.state_dict(), path + 'decoder.pth')
    def load_model(self, path: str):
        self.encoder.load_state_dict(torch.load(path + 'encoder.pth'))
        self.decoder.load_state_dict(torch.load(path + 'decoder.pth'))
    