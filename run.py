from data_utils.prepare_data import load_celebA
from autoencoders.VQ_VAE.VQ_VAE import VQ_VAE
from autoencoders.VQ_VAE.VQ_VAE_2 import VQ_VAE2
from trainers.autoencoder_trainer import VQVAE_Trainer
from yaml import safe_load
import argparse
import torch

def main(config):
    train_dataloader, validation_dataloader, test_dataloader = load_celebA(
        config['img_size'], 
        config['validation_ratio'], 
        config['test_ratio'], 
        config['batch_size']
    )
    vqvae_config = config['VQ-VAE']
    vqvae = VQ_VAE(
        3, 
        vqvae_config['latent_dimension'], 
        vqvae_config['kernel_sizes'], 
        vqvae_config['res_layers'], 
        vqvae_config['code_book_size']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae_config['device'] = device
    trainer = VQVAE_Trainer(vqvae, vqvae_config)
    trainer.train(train_dataloader, validation_dataloader)
    trainer.test(test_dataloader)
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="./configs/default.yaml", help="Path to the config file")
    args = argparser.parse_args()
    config = safe_load(open(args.config))
    main(config)