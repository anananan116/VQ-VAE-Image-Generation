import torch
import torch.nn
import torch.nn.functional as F

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_layer = False):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding = "same")
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding = "same")
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)
        self.last_layer = last_layer

    def forward(self, x):
        residual = x
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        if self.last_layer:
            out += F.relu(residual)
        else:
            out += F.relu(residual)
        return out
    
class convBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(convBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = kernel_size//2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = F.relu(self.batch_norm(self.conv(x)))
        return out

class decovBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, last_layer = False):
        super(decovBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        if kernel_size % 2 == 1:  # If kernel size is odd
            self.deconv.padding = ((kernel_size - 1) // 2,) * 2  # Typical padding formula for odd kernels
            self.deconv.output_padding = (1,) * 2  # Often needed adjustment for stride of 2
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.last_layer = last_layer
        
    def forward(self, x):
        if self.last_layer:
            out = self.deconv(x)
        else:
            out = F.relu(self.batch_norm(self.deconv(x)))
        return out

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, latent_dimension, kernel_sizes, n_res_layers):
        super(Encoder, self).__init__()
        conv_layers = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                conv_layers.append(convBlock(in_channels, latent_dimension, kernel_sizes[i], 2))
            else:
                conv_layers.append(convBlock(latent_dimension, latent_dimension, kernel_sizes[i], 2))
        self.conv_layers = torch.nn.Sequential(*conv_layers)
        res_layers = []
        for i in range(n_res_layers):
            if i == n_res_layers - 1:
                res_layers.append(ResidualBlock(latent_dimension, latent_dimension, kernel_sizes[-1], last_layer = True))
            res_layers.append(ResidualBlock(latent_dimension, latent_dimension, kernel_sizes[-1]))
        self.res_layers = torch.nn.Sequential(*res_layers)
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = self.res_layers(out)
        return out
    
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dimension, in_channels, kernel_sizes, n_res_layers):
        super(Decoder, self).__init__()
        kernel_sizes.reverse()
        decov_layers = []
        for i in range(len(kernel_sizes)):
            if i == len(kernel_sizes) - 1:
                decov_layers.append(decovBlock(latent_dimension, in_channels, kernel_sizes[i], 2, last_layer = True))
            else:
                decov_layers.append(decovBlock(latent_dimension, latent_dimension, kernel_sizes[i], 2))
        self.decov_layers = torch.nn.Sequential(*decov_layers)
        res_layers = []
        for i in range(n_res_layers):
            res_layers.append(ResidualBlock(latent_dimension, latent_dimension, kernel_sizes[-1]))
        self.res_layers = torch.nn.Sequential(*res_layers)
        
    def forward(self, x):
        out = self.res_layers(x)
        out = self.decov_layers(out)
        return out
    
class QuantizationLayer(torch.nn.Module):
    def __init__(self, latent_dimension, code_book_size):
        super(QuantizationLayer, self).__init__()
        self.latent_dimension = latent_dimension
        self.code_book_size = code_book_size
        self.code_book = torch.nn.Parameter(torch.randn(code_book_size, latent_dimension, requires_grad=True), requires_grad=True)

    def forward(self, x, count_low_usage = False):
        # Flatten input
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_dimension)
        
        # Compute distances between input vectors and embeddings
        distances = torch.sum(flat_x**2, dim=1, keepdim=True) + torch.sum(self.code_book**2, dim=1) - 2 * torch.matmul(flat_x, self.code_book.t())
        
        # Find the closest embeddings
        min_distances_indices = torch.argmin(distances, dim=1)
        
        # Quantize the input
        z_q = torch.index_select(self.code_book, 0, min_distances_indices).view(x.shape)
        
        # Use a straight-through estimator for backpropagation
        z_hat = x + (z_q - x).detach()
        if count_low_usage:
            count = 0
            counts = torch.bincount(min_distances_indices, minlength=self.code_book_size)
            low_usage_indices = torch.where(counts < 1)[0]
            count = len(low_usage_indices)
            return z_hat, z_q, count
        return z_hat, z_q

    def encode(self, x):
        # Flatten input
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_dimension)
        
        # Compute distances between input vectors and embeddings
        distances = torch.sum(flat_x**2, dim=1, keepdim=True) + torch.sum(self.code_book**2, dim=1) - 2 * torch.matmul(flat_x, self.code_book.t())
        
        # Find the closest embeddings
        min_distances_indices = torch.argmin(distances, dim=1).view(x.shape)
        
        return min_distances_indices