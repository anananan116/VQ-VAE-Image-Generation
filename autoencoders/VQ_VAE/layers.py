import torch
import torch.nn
import torch.nn.functional as F

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = [3,1], last_layer = False):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size[0], padding = "same")
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size[1], padding = "same")
        self.last_layer = last_layer
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        if self.last_layer:
            out = self.conv2(out)
        else: 
            out = self.relu(self.conv2(out))
        out = out + residual
        return out
    
class convBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(convBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = kernel_size//2)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        #out = self.relu(self.batch_norm(self.conv(x)))
        out = self.relu(self.conv(x))
        return out

class decovBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, last_layer = False):
        super(decovBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride = stride, padding = 1)
        if kernel_size % 2 == 1:  # If kernel size is odd
            self.deconv.padding = ((kernel_size - 1) // 2,) * 2  # Typical padding formula for odd kernels
            self.deconv.output_padding = (1,) * 2  # Often needed adjustment for stride of 2
        self.last_layer = last_layer
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        if self.last_layer:
            out = self.deconv(x)
        else:
            out = self.relu(self.deconv(x))
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
                res_layers.append(ResidualBlock(latent_dimension, latent_dimension, last_layer = True))
            else:
                res_layers.append(ResidualBlock(latent_dimension, latent_dimension))
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
                decov_layers.append(decovBlock(latent_dimension, in_channels, kernel_sizes[i] + 1, 2, last_layer = True))
            else:
                decov_layers.append(decovBlock(latent_dimension, latent_dimension, kernel_sizes[i] + 1, 2))
        res_layers = []
        for i in range(n_res_layers):
            res_layers.append(ResidualBlock(latent_dimension, latent_dimension))
        self.res_layers = torch.nn.Sequential(*res_layers)
        self.decov_layers = torch.nn.Sequential(*decov_layers)
        
    def forward(self, x):
        out = self.res_layers(x)
        out = self.decov_layers(out)
        return out
    
class QuantizationLayer(torch.nn.Module):
    def __init__(self, latent_dimension, code_book_size, lower_bound_factor, decay=0.99, eps=1e-5):
        super(QuantizationLayer, self).__init__()
        self.dim = latent_dimension
        self.n_embed = code_book_size
        self.decay = decay
        self.eps = eps

        embed = torch.zeros(latent_dimension, code_book_size)
        self.embed = torch.nn.Parameter(embed, requires_grad=True)
        # self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(code_book_size))
        self.register_buffer("embed_avg", embed.clone())
        self.pixels_each_batch = None
        self.lower_bound_factor = lower_bound_factor

    def forward(self, x, count_low_usage = False):
        self.pixels_each_batch = x.size(0) * x.size(2) * x.size(3)
        x = x.permute(0, 2, 3, 1).contiguous() #(B, H, W, C)
        flatten = x.reshape(-1, self.dim)
        if torch.norm(self.embed.data) == 0:
            print("initializing codebook")
            random_indices = torch.randint(0, flatten.size(0), (self.n_embed,))
            random_samples = flatten[random_indices].detach()
            self.embed.data[:] = random_samples.T
            self.embed_avg.data[:] = random_samples.T
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
            small_clusters = self.cluster_size < self.pixels_each_batch/self.n_embed/self.lower_bound_factor
            n_small_clusters = small_clusters.sum().item()
            if n_small_clusters > 16:
                random_indices = torch.randint(0, flatten.size(0), (n_small_clusters,))
                random_samples = flatten[random_indices].detach()
                self.embed.data[:, small_clusters] = random_samples.T
                self.embed_avg.data[:, small_clusters] = random_samples.T
                self.cluster_size.data[small_clusters] = self.pixels_each_batch/self.n_embed/self.lower_bound_factor
            
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        quant_loss = torch.nn.functional.mse_loss(quantize.detach(), x)
        quantize = (x + (quantize - x).detach()).permute(0, 3, 1, 2) #back to (B, C, H, W)
        return quantize, quant_loss
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def encode_to_id(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() #(B, H, W, C)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        
        return embed_ind
    
    def decode_from_id(self, embed_idx):
        quantize = self.embed_code(embed_idx)
        return quantize.permute(0, 3, 1, 2) #back to (B, C, H, W)