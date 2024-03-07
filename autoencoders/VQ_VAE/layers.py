import torch
import torch.nn
import torch.nn.functional as F

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_layer = False):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding = "same")
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 1, padding = "same")
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)
        self.last_layer = last_layer

    def forward(self, x):
        residual = x
        out = F.relu(self.batch_norm1(self.conv1(x)))
        if self.last_layer:
            out = self.conv2(out)
        else: 
            out = F.relu(self.batch_norm2(self.conv2(out)))
        out = out + residual
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
            else:
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
    def __init__(self, latent_dimension, code_book_size, decay=0.99, eps=1e-5):
        super(QuantizationLayer, self).__init__()
        self.dim = latent_dimension
        self.n_embed = code_book_size
        self.decay = decay
        self.eps = eps

        embed = torch.randn(latent_dimension, code_book_size)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(code_book_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x, count_low_usage = False):
        x = x.permute(0, 2, 3, 1).contiguous()
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            # reassign low usage entries
            small_clusters = self.cluster_size < 2.0
            n_small_clusters = small_clusters.sum().item()
            if n_small_clusters > 0:
                random_indices = torch.randint(0, flatten.size(0), (n_small_clusters,))
                random_samples = flatten[random_indices].detach()
                self.embed.data[:, small_clusters] = random_samples.T
                self.embed_avg[:, small_clusters] = random_samples.T
                self.embed.cluster_size[small_clusters] = 5.0
                
            
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = quantize.permute(0, 3, 1, 2)
        # quantize = (x + (quantize - x).detach()).permute(0, 3, 1, 2)
        quantize = (x + (quantize - x).detach()).permute(0, 3, 1, 2)
        return quantize, diff
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def encode(self, x):
        # Flatten input
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_dimension)
        
        # Compute distances between input vectors and embeddings
        distances = torch.sum(flat_x**2, dim=1, keepdim=True) + torch.sum(self.code_book**2, dim=1) - 2 * torch.matmul(flat_x, self.code_book.t())
        
        # Find the closest embeddings
        min_distances_indices = torch.argmin(distances, dim=1).view(x.shape)
        
        return min_distances_indices