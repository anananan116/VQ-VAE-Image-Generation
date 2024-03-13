from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel


class LayerPredictionModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        config.vocab_size += 2
        super().__init__(config=config)

        # NOTE: Do not use input any sequence longer than `side ** 2`` tokens.
        # The positional embedding for longer sequences do not make sense at all.
        positional_embedding = nn.Embedding(self.config.n_positions, self.config.n_embd)
        positional_embedding.weight.data[1 : 1 + config.side**2] = PositionalEncoding2D(
            self.config.n_embd
        )(torch.empty(1, config.side, config.side, config.n_embd)).view(
            self.config.side**2, self.config.n_embd
        )
        positional_embedding.requires_grad_(False)
        self.transformer.wpe = positional_embedding

    def forward(self, **kwargs):
        kwargs["input_ids"] = torch.cat(
            [torch.zeros_like(kwargs["input_ids"][:, :1]), kwargs["input_ids"]],
            dim=1,
        )
        kwargs["labels"] = torch.cat(
            [kwargs["input_ids"][:, 1:], torch.ones_like(kwargs["input_ids"][:, :1])],
            dim=1,
        )

        return super().forward(**kwargs)


class LayerDataset(Dataset):
    def __init__(self, ids: np.ndarray):
        self.ids = np.int64(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        return self.ids[index] + 2  # Skip BOS and EOS tokens


if __name__ == "__main__":
    from tqdm.auto import tqdm

    data_path = Path(__file__).parent / "data"
    top_ids = np.load(data_path / "t_codes.npy")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ds = LayerDataset(top_ids)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    model = LayerPredictionModel(
        GPT2Config(vocab_size=512, n_positions=1026, n_embd=120, side=32)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for batch in tqdm(dl):
        optimizer.zero_grad()
        batch = batch.view(8, -1)
        loss = model(input_ids=batch.to(device), labels=batch.to(device)).loss
        loss.backward()
        optimizer.step()
