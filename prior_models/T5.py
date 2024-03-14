from transformers import T5ForConditionalGeneration, T5Config, GPT2LMHeadModel, GPT2Config
from torch import nn
import torch

    
class T5Layer(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(self, x, cond):
        input_ids = x.flatten(start_dim=1)
        dec_input_ids = cond.flatten(start_dim=1)
        outputs = super().forward(input_ids=input_ids, decoder_input_ids=dec_input_ids)
        loss = outputs.loss
        return loss
