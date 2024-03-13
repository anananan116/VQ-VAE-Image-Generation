from transformers import T5ForConditionalGeneration, T5Config, GPT2LMHeadModel, GPT2Config
from torch import nn
import torch
from pos_encodings import PositionalEncoding2D, Summer

class GPTTopLayer(nn.Module):
    def __init__(self, GPT_config):
        super(GPTTopLayer, self).__init__()
        self.gpt = GPT2LMHeadModel(GPT_config)
        # Input embedding layer
        self.embedding = nn.Embedding(GPT_config.vocab_size, GPT_config.n_embd)
        self.gpt.set_input_embeddings(self.embedding)
        # Positional encoding for position ids
        self.position = Summer(PositionalEncoding2D(GPT_config.n_embd))
    
    def forward(self, x):
        target = x[1:].clone()
        outputs = self.gpt(input_ids=x, position_ids=self.position(x))
        loss = outputs.loss
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=-1)
        correct = (predicted == target).float()  # convert to float for division 
        accuracy = correct.sum() / len(correct)
        return loss, accuracy
    
class T5BottomLayer(nn.Module):
    def __init__(self, T5_config, n_embd):
        super(T5BottomLayer, self).__init__()
        self.T5 = T5ForConditionalGeneration(T5_config)
        # Input embedding layer
        self.embedding = nn.Embedding(T5_config.vocab_size, n_embd)
        self.T5.set_input_embeddings(self.embedding)
        # Output embedding layer
        self.lm_head = nn.Linear(n_embd, T5_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie weights
        self.T5.set_output_embeddings(self.lm_head)
        
    def forward(self, x, condition):
        target = x[1:].clone()
        outputs = self.T5(input_ids=condition, decoder_input_ids=x, labels=target)
        loss = outputs.loss
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=-1)
        correct = (predicted == target).float()  # convert to float for division 
        accuracy = correct.sum() / len(correct)
        return loss, accuracy