from data_utils.prepare_data import load_images
from autoencoders.VQ_VAE.VQ_VAE import VQ_VAE, VQ_VAE2
from trainers.autoencoder_trainer import VQVAE_Trainer
from yaml import safe_load
import argparse
import torch

def main(config, device="cuda:0"):
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
            vqvae_config['code_book_size'],
            vqvae_config['lower_bound_factor']
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
    device = torch.device(device)
    vqvae_config['device'] = device
    trainer = VQVAE_Trainer(vqvae, vqvae_config)
    trainer.train(train_dataloader, validation_dataloader)
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="./configs/default.yaml", help="Path to the config file")
    argparser.add_argument('--device', type=str, default="cuda:0", help="device to train on")
    args = argparser.parse_args()
    config = safe_load(open(args.config))
    main(config, device=args.device)