from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig, PreTrainedModel, AutoModel


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):

    def __init__(self, path, linear_probe=False, num_label=2, dropout=0.1):
        config = AutoConfig.from_pretrained(path)
        super().__init__(config)
        self.transformer = AutoModel.from_pretrained(path)
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        print(hidden_size)
        self.score = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_size, num_label)
        )
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        )

    def forward(self, 
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None):

        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = torch.mean(transformer_outputs.last_hidden_state, dim=-2)
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits
