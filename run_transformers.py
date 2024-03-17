from transformers import T5Config, T5ForConditionalGeneration, GPT2Config
from prior_models.gpt import LayerPredictionModel
from data_utils.prepare_data import load_latent_code
from yaml import safe_load
import argparse
import torch
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

def compute_accuracy(eval_pred: EvalPrediction):
    """
    Compute the accuracy of predictions.
    
    Args:
        eval_pred (EvalPrediction): An object containing the model predictions and true labels.
        
    Returns:
        dict: A dictionary with the key 'accuracy' and its computed value.
    """
    # Extract predictions and true labels from eval_pred
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = (labels==preds).sum()/(labels.shape[0]*labels.shape[1])
    # Return the accuracy in a dictionary
    return {"accuracy": acc}

def main(config):
    trainer_config = config['trainer']
    model_config = config['model']
    train_dataset, val_dataset = load_latent_code(config['validation_ratio'], 0, for_transformer=True, hier=config['hier'])
    print(train_dataset[0])
    num_steps_per_epoch = (len(train_dataset) // (trainer_config['per_device_train_batch_size'] * torch.cuda.device_count() * trainer_config['gradient_accumulation_steps'])) + 1
    trainer_config['eval_steps'] = num_steps_per_epoch // config['num_eval_per_epoch']
    if config['hier'] == 'top':
        model = LayerPredictionModel(GPT2Config(**model_config))
    elif config['hier'] == 'bottom':
        model = T5ForConditionalGeneration(T5Config(**model_config))
    else:
        raise ValueError("Invalid hierarchy")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params:,} total parameters.')
    trainer_arguments = Seq2SeqTrainingArguments(**trainer_config)
    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_accuracy,
        callbacks=[]
    )
    trainer.train()
    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="./configs_transformers/top.yaml", help="Path to the config file")
    args = argparser.parse_args()
    config = safe_load(open(args.config))
    main(config)