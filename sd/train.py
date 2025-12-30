import torch
from torch import nn
from torch.nn import functional as F
from encoder import VAE_Encoder
from decoder import VAE_Decoder

class VAE_Trainer():
    def __init__(self):
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.optimizer = torch.optim.Adam(self.encoder.parameters() + self.decoder.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.sigma = 1

    def train_epoch(self, dataset: Dataset):
        for batch in dataset:
            x = batch['image']
            n, c, h, w = x.shape
            z, mean, stdev = self.encoder(x)
            x_hat = self.decoder(z)
            loss = self.loss_fn(x, x_hat)/(2*self.sigma**2)
            loss += (mean**2 + stdev**2 - 1 - 2*stdev.log()).abs()/n
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss
    
    def train(self, dataset: Dataset, epochs: int):
        for epoch in range(epochs):
            loss = self.train_epoch(dataset)
            val_loss = self.evaluate(dataset)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        return loss
    def save_model(self, path: str):
        torch.save(self.encoder.state_dict(), path + 'encoder.pth')
        torch.save(self.decoder.state_dict(), path + 'decoder.pth')
    def load_model(self, path: str):
        self.encoder.load_state_dict(torch.load(path + 'encoder.pth'))
        self.decoder.load_state_dict(torch.load(path + 'decoder.pth'))
    @torch.no_grad()
    def evaluate(self, dataset: Dataset):
        for batch in dataset:
            x = batch['image']
            n, c, h, w = x.shape
            z, mean, stdev = self.encoder(x)
            x_hat = self.decoder(z)
            loss = self.loss_fn(x, x_hat)/(2*self.sigma**2)
            loss += (mean**2 + stdev**2 - 1 - 2*stdev.log()).abs()/n
            return loss
    @torch.no_grad()
    def generate(self, z: torch.Tensor):
        x_hat = self.decoder(z)
        generated_image = x_hat + self.sigma * torch.randn_like(x_hat)
        return generated_image
    