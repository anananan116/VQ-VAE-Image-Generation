from trainers.GPT_T5_trainer import GPT_T5_Trainer
from data_utils.prepare_data import load_latent_code
from yaml import safe_load
import argparse
import torch
from transformers import GPT2Config, T5Config
from prior_models.GPT_T5 import GPTTopLayer, T5BottomLayer


def main(config):
    train_dataloader, val_dataloader = load_latent_code(
        config['validation_ratio'],
        config['batch_size']
    )
    if config['hier'] == 'top':
        GPT_config = GPT2Config(vocab_size=config['top_vocab_size'], n_embd=config['top_n_embd'])
        model = GPTTopLayer(GPT_config)
    else:
        T5_config = T5Config(vocab_size=config['bottom_vocab_size'])
        model = T5BottomLayer(T5_config, n_embd=config['bottom_n_embd'])

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Wrap the model with DataParallel
        model = torch.nn.DataParallel(model)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    config['device'] = device
    trainer = GPT_T5_Trainer(model, config)
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="./configs/default.yaml", help="Path to the config file")
    args = argparser.parse_args()
    config = safe_load(open(args.config))
    main(config)