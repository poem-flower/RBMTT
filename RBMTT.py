import torch
import torch.nn as nn
from torch.nn import Transformer
from torch import Tensor
import math


def data_pre(x):
    x_pre = x.clone()
    for i in range(1, x.size(0) - 1):
        x_pre[i+1] = x[i] + (x[i]-x[i-1])
    return x_pre


def loss_fn(x1: Tensor, x2: Tensor) -> Tensor:
    time_steps = x1.size(0)
    batch_size = x1.size(1)
    loss = torch.pow(torch.sum(torch.pow((x1-x2), 2))/(time_steps*batch_size), 0.5)
    return loss


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_attention_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    src_attention_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    tgt_attention_mask = generate_square_subsequent_mask(tgt_seq_len)
    return src_attention_mask, tgt_attention_mask


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):  # maxlen参数是什么？先保留
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class RBMTT(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_input: int,
                 dim_output: int,
                 dim_feedforward: int,
                 dropout: float
                 ):
        super(RBMTT, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.input_linear = nn.Linear(dim_input, emb_size)
        self.output_linear = nn.Linear(dim_output, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.linear = nn.Linear(emb_size, dim_output)  # 线性层输入[time_step, batch_size, emb_size], 输出[time_step, batch_size, dim_output]

    def forward(self,
                src: Tensor,
                tgt: Tensor):
        src_attention_mask, tgt_attention_mask = create_attention_mask(src, tgt)
        src_attention_mask, tgt_attention_mask = src_attention_mask.to('cuda'), tgt_attention_mask.to('cuda')
        input_emb = self.positional_encoding(self.input_linear(src))
        output_emb = self.positional_encoding(self.output_linear(tgt))
        outs = self.transformer(input_emb, output_emb, src_mask=src_attention_mask, tgt_mask=tgt_attention_mask)
        return self.linear(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.input_linear(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.output_linear(tgt)), memory, tgt_mask)


