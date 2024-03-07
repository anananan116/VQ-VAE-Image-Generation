from .layers import Encoder, Decoder, QuantizationLayer
import torch
import torch.nn as nn

class VQ_VAE(torch.nn.Module):
    def __init__(self, in_channels, latent_dimension, kernel_sizes, res_layers, code_book_size):
        super(VQ_VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dimension, kernel_sizes, res_layers)
        self.decoder = Decoder(latent_dimension, in_channels, kernel_sizes, res_layers)
        self.quantization = QuantizationLayer(latent_dimension, code_book_size)

        
    def forward(self, x, count_low_usage = False):
        z = self.encoder(x)
        if count_low_usage:
            z_hat, z_q, count = self.quantization(z, count_low_usage = True)
            x_hat = self.decoder(z_hat)
            return x_hat, z_q, z, count
        else:
            z_hat, z_q = self.quantization(z)
            x_hat = self.decoder(z_hat)
            return x_hat, z_q, z
