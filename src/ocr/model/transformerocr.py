import torch.nn as nn

from ocr.model.backbone.cnn import CNN
from ocr.model.seqmodel.transformer import LanguageTransformer


class VietOCR(nn.Module):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args,
                 transformer_args, seq_modeling='transformer') -> None:
        super().__init__()
        
        self.cnn = CNN(backbone, **cnn_args)
        self.transformer = LanguageTransformer(vocab_size, **transformer_args)
    
    def forward(self, image, tgt_input, tgt_key_padding_mask):
        src = self.cnn(image)
        out = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        return out