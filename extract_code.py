from data_utils.prepare_data import load_celebA
from autoencoders.VQ_VAE.VQ_VAE import VQ_VAE,VQ_VAE2
from trainers.autoencoder_trainer import VQVAE_Trainer
from yaml import safe_load
import argparse
import torch
import numpy as np

def extract(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    t_codes = []
    b_codes = []
    with torch.no_grad():
        for data in dataloader:
            b_code, t_code = model.encode_to_id(data.to(device))
            t_code = t_code.cpu().numpy()
            b_code = b_code.cpu().numpy()
            t_codes.extend(t_code)
            b_codes.extend(b_code)
    t_codes = np.array(t_codes).astype(np.uint16)
    b_codes = np.array(b_codes).astype(np.uint16)
    np.save("t_codes.npy", t_codes)
    np.save("b_codes.npy", b_codes)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="./configs/default.yaml", help="Path to the config file")
    args = argparser.parse_args()
    config = safe_load(open(args.config))
    train_dataloader, validation_dataloader, test_dataloader = load_celebA(
            config['img_size'], 
            0, 
            0, 
            config['batch_size']
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


    checkpoint = torch.load('./results/best_model.pth')
    vqvae.load_state_dict(checkpoint)
    vqvae.to(device)
    extract(vqvae, train_dataloader)