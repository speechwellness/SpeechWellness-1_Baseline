import torch

from torch import nn

class ConcatModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        self.speech_fc = nn.Linear(args.speech_input_dim, args.hidden_dim)
        self.text_fc = nn.Linear(args.text_input_dim, args.hidden_dim)
        if self.mode == "s+t":
            self.out_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(2*args.hidden_dim, 2*args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(2*args.hidden_dim, args.num_label)
            )
        else:
            self.out_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(args.hidden_dim, args.num_label)
            )

    def forward(self, speech_feature, text_feature):
        speech_feature = self.speech_fc(speech_feature)
        text_feature = self.text_fc(text_feature)
        if self.mode == "s+t":
            combine_feature = torch.cat((speech_feature, text_feature), dim=-1)
            output = self.out_layer(combine_feature)
        elif self.mode == "s":
            output = self.out_layer(speech_feature)
        elif self.mode == "t":
            output = self.out_layer(text_feature)
        return output