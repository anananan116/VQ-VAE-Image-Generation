from .layers import Encoder, Decoder, QuantizationLayer
import torch
import torch.nn as nn

class VQ_VAE(torch.nn.Module):
    def __init__(self, in_channels, latent_dimension, kernel_sizes, res_layers, code_book_size):
        super(VQ_VAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dimension, kernel_sizes, res_layers)
        self.decoder = Decoder(latent_dimension, in_channels, kernel_sizes, res_layers)
        self.quantization = QuantizationLayer(latent_dimension, code_book_size)

        
    def forward(self, x):
        z = self.encoder(x)
        z_hat, quant_loss = self.quantization(z)
        x_hat = self.decoder(z_hat)
        return x_hat, quant_loss

class VQ_VAE2(torch.nn.Module):
    def __init__(self, in_channels, latent_dimension, kernel_sizes, res_layers, code_book_size):
        super(VQ_VAE2, self).__init__()
        self.encoder_bottom = Encoder(in_channels, latent_dimension, kernel_sizes[:-1], res_layers)         # B*3*H*W -> B*C*h_b*w_b
        self.quat_cov_bottom = nn.Conv2d(latent_dimension, latent_dimension//2, 1)                          # B*C*h_b*w_b -> B*(C/2)*h_b*w_b
        self.quantization_bottom = QuantizationLayer(latent_dimension, code_book_size)
        
        self.encoder_top = Encoder(latent_dimension, latent_dimension, kernel_sizes[-1:], res_layers)       # B*C*h_b*w_b -> B*C*h_t*w_t
        self.quat_cov_top = nn.Conv2d(latent_dimension, latent_dimension//2, 1)                             # B*C*h_t*w_t -> B*(C/2)*h_t*w_t
        self.quantization_top = QuantizationLayer(latent_dimension // 2, code_book_size)
        self.upsample_top = nn.ConvTranspose2d(latent_dimension, latent_dimension//2, kernel_size=3, stride= 2, padding = 1, output_padding= 1)           # B*C*h_t*w_t -> B*(C/2)*2h_b*2w_b
        
        self.decoder_top = Decoder(latent_dimension//2, latent_dimension//2, kernel_sizes[-1:], res_layers) # B*(C/2)*h_t*w_t -> B*C*h_b*w_b
        self.decoder_bottom = Decoder(latent_dimension + latent_dimension//2, in_channels, kernel_sizes[:-1], res_layers)         # B*C*h_b*w_b -> B*3*H*W

    def encode(self, x):
        b_encoded = self.encoder_bottom(x)
        t_encoded = self.encoder_top(b_encoded)
        b_encoded = self.quat_cov_bottom(b_encoded)
        t_encoded_q = self.quat_cov_top(t_encoded)
        t_quantized, t_quant_loss = self.quantization_top(t_encoded_q)
        t_upsampled = self.upsample_top(t_encoded)
        t_quantized = self.decoder_top(t_quantized)
        b_encoded = torch.concat((b_encoded, t_quantized), dim=1)
        b_quantized, b_quant_loss = self.quantization_bottom(b_encoded)
        return b_quantized, t_upsampled, b_quant_loss + t_quant_loss

    def decode(self, b_quantized):
        return self.decoder_bottom(b_quantized)
    
    def forward(self, x):
        b_quantized, t_upsampled, quant_loss = self.encode(x)
        b_quantized = torch.concat((b_quantized, t_upsampled), dim=1)
        x_hat = self.decode(b_quantized)
        return x_hat, quant_loss