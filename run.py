from data_utils.prepare_data import load_images
from autoencoders.VQ_VAE.VQ_VAE import VQ_VAE, VQ_VAE2
from trainers.autoencoder_trainer import VQVAE_Trainer
from yaml import safe_load
import argparse
import torch

def main(config):
    train_dataloader, validation_dataloader, test_dataloader = load_images(
            config['img_size'], 
            config['validation_ratio'], 
            config['test_ratio'], 
            config['batch_size'],
            config['dataset']
        )
    vqvae_config = config['VQ-VAE']
    if 'version' in vqvae_config.keys() and vqvae_config['version'] == 2:
        vqvae = VQ_VAE2(
            3, 
            vqvae_config['latent_dimension'], 
            vqvae_config['kernel_sizes'], 
            vqvae_config['res_layers'], 
            vqvae_config['code_book_size']
        )
    else:
        vqvae_config['version'] = 1
        vqvae = VQ_VAE(
            3, 
            vqvae_config['latent_dimension'], 
            vqvae_config['kernel_sizes'], 
            vqvae_config['res_layers'], 
            vqvae_config['code_book_size']
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae_config['device'] = device
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Wrap the model with DataParallel
        vqvae = torch.nn.DataParallel(vqvae)
    trainer = VQVAE_Trainer(vqvae, vqvae_config)
    trainer.train(train_dataloader, validation_dataloader)
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="./configs/default.yaml", help="Path to the config file")
    args = argparser.parse_args()
    config = safe_load(open(args.config))
    main(config)