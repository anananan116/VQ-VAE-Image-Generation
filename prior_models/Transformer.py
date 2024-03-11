from transformers import T5ForConditionalGeneration, T5Config, GPT2Model, GPT2Config
from torch import nn
import torch
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
class TopLayer(nn.Module):
    def __init__(self, gpt_config, vocab_size):
        super(TopLayer, self).__init__()
        self.gpt = GPT2Model(gpt_config)
        self.embedding = nn.Embedding(vocab_size, gpt_config.n_embd)
        self.positional_encoding = Summer(PositionalEncoding2D(gpt_config.n_embd))
        self.lm_head = nn.Linear(gpt_config.n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie weights
        self.crossentropy = nn.CrossEntropyLoss()
    
    def forward(self, x):
        target = x[1:].clone()
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.reshape(x.size(0), -1, x.size(-1)) # b*h*w*c -> b*seq_len*c
        x = torch.hstack([self.embedding(torch.zeros(x.size(0), 1, dtype=torch.long, device = x.device)), x])
        x = torch.hstack([x, self.embedding(torch.ones(x.size(0), 1, dtype=torch.long, device = x.device))]) # add start and end tokens
        outputs = self.gpt(inputs_embeds=x[:-1]).last_hidden_state
        logits = self.lm_head(outputs)
        loss = self.crossentropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        _, predicted = torch.max(logits, dim=-1)
        correct = (predicted == target).float()  # convert to float for division 
        accuracy = correct.sum() / len(correct)
        return loss, accuracy
    
class BottomLayer(nn.Module):
    def __init__(self, T5_config, vocab_size):
        super(BottomLayer, self).__init__()
        self.t5 = T5ForConditionalGeneration(T5_config)
        self.embedding = nn.Embedding(vocab_size, T5_config.n_embd)
        self.positional_encoding = Summer(PositionalEncoding2D(T5_config.n_embd))
        self.lm_head = nn.Linear(T5_config.n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie weights
        self.crossentropy = nn.CrossEntropyLoss()
        
    def forward(self, x, condition):
        target = x[1:].clone()
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.reshape(x.size(0), -1, x.size(-1))
        x = torch.hstack([self.embedding(torch.zeros(x.size(0), 1, dtype=torch.long, device = x.device)), x])
        x = torch.hstack([x, self.embedding(torch.ones(x.size(0), 1, dtype=torch.long, device = x.device))]) # add start and end tokens
        
        condition = self.embedding(condition)
        condition = self.positional_encoding(condition)
        condition = condition.reshape(condition.size(0), -1, condition.size(-1))
        condition = torch.hstack([condition, self.embedding(torch.ones(condition.size(0), 1, dtype=torch.long, device = condition.device))]) # add start and end tokens
        
        outputs = self.t5(input_embeds=condition, decoder_inputs_embeds=x).last_hidden_state
        logits = self.lm_head(outputs)
        loss = self.crossentropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        _, predicted = torch.max(logits, dim=-1)
        correct = (predicted == target).float()  # convert to float for division 
        accuracy = correct.sum() / len(correct)
        return loss, accuracy