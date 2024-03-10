from trainers.pixelsnail_trainer import pixelSNAIL_Trainer
from data_utils.prepare_data import load_latent_code
from yaml import safe_load
import argparse
import torch
from prior_models.pixel_snail import PixelSNAIL


def main(config):
    train_dataloader, val_dataloader = load_latent_code(
        config['validation_ratio'],
        config['batch_size']
    )
    if config['hier'] == 'top':
        model = PixelSNAIL(
                [32, 32],
                512,
                config['channel'],
                5,
                4,
                config['n_res_block'],
                config['n_res_channel'],
                dropout=config['dropout'],
                n_out_res_block=config['n_out_res_block']
        )
    else:
        model = PixelSNAIL(
                [64, 64],
                512,
                config['channel'],
                5,
                4,
                config['n_res_block'],
                config['n_res_channel'],
                dropout=config['dropout'],
                n_out_res_block=config['n_out_res_block'],
                n_cond_res_block=config['n_cond_res_block'],
                cond_res_channel=config['n_res_channel'],
                attention=False
        )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Wrap the model with DataParallel
        model = torch.nn.DataParallel(model)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    config['device'] = device
    trainer = pixelSNAIL_Trainer(model, config)
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="./configs/default.yaml", help="Path to the config file")
    args = argparser.parse_args()
    config = safe_load(open(args.config))
    main(config)