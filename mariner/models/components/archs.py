import torch
import torch.nn as nn
from models.components.utils import MASA_weight_init


class ArchitectureMARINER(nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        attention: torch.nn.Module,
        in_out_skip: bool = True,
        weight_init: bool = True,
    ):
        super(ArchitectureMARINER, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.in_out_skip = in_out_skip
        if weight_init:
            MASA_weight_init(self, scale=0.1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fea_r_l = self.encoder(batch["IN"])
        fea_ref_l = self.encoder(batch["REF"])
        warp_fea_ref_l = self.attention(fea_r_l, fea_ref_l)
        out = self.decoder(fea_r_l, warp_fea_ref_l)
        if self.in_out_skip:
            out = out + batch["IN"]

        return out
