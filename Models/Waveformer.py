import math
from collections import OrderedDict
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import torchaudio
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding

def mod_pad(x, chunk_size, pad):
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                      padding, groups=in_channels, dilation=dilation),
            LayerNormPermuted(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0),
            LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
    
class DilatedCausalConvEncoder(nn.Module):
    def __init__(self, channels, num_layers, kernel_size=3):
        super(DilatedCausalConvEncoder, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.buf_lengths = [(kernel_size - 1) * 2**i
                            for i in range(num_layers)]

        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(
                self.buf_indices[-1] + self.buf_lengths[i])

        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            dcc_layer = DepthwiseSeparableConv(
                channels, channels, kernel_size=3, stride=1,
                padding=0, dilation=2**i)
            _dcc_layers.update({'dcc_%d' % i: dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.channels,
                 (self.kernel_size - 1) * (2**self.num_layers - 1)),
            device=device)

    def forward(self, x, ctx_buf):
        T = x.shape[-1]

        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            dcc_in = torch.cat(
                (ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)

            ctx_buf[..., buf_start_idx:buf_end_idx] = \
                dcc_in[..., -self.buf_lengths[i]:]

            x = x + self.dcc_layers[i](dcc_in)

        return x, ctx_buf
    
class CausalTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        chunk_size: int = 1
    ) -> Tensor:
        tgt_last_tok = tgt[:, -chunk_size:, :]

        tmp_tgt, sa_map = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,
            key_padding_mask=None,
        )
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=None,
                key_padding_mask=None,
            )
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, sa_map, ca_map
    
class CausalTransformerDecoder(nn.Module):
    def __init__(self, model_dim, ctx_len, chunk_size, num_layers,
                 nhead, use_pos_enc, ff_dim):
        super(CausalTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
        self.pos_enc = PositionalEncoding(model_dim, max_len=200)
        self.tf_dec_layers = nn.ModuleList([CausalTransformerDecoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True) for _ in range(num_layers)])

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros(
            (batch_size, self.num_layers + 1, self.ctx_len, self.model_dim),
            device=device)

    def _causal_unfold(self, x):
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.

        Args:
            x: [B, ctx_len + L, C]
            ctx_len: int
        Returns:
            [B * L, ctx_len + 1, C]
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.unfold(x.unsqueeze(-1))
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size)
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, tgt, mem, ctx_buf, probe=False):
        """
        Args:
            x: [B, model_dim, T]
            ctx_buf: [B, num_layers, model_dim, ctx_len]
        """
        mem, _ = mod_pad(mem, self.chunk_size, (0, 0))
        tgt, mod = mod_pad(tgt, self.chunk_size, (0, 0))

        B, C, T = tgt.shape

        tgt = tgt.permute(0, 2, 1)
        mem = mem.permute(0, 2, 1)

        mem = torch.cat((ctx_buf[:, 0, :, :], mem), dim=1)
        ctx_buf[:, 0, :, :] = mem[:, -self.ctx_len:, :]
        mem_ctx = self._causal_unfold(mem)
        if self.use_pos_enc:
            mem_ctx = mem_ctx + self.pos_enc(mem_ctx)

        K = 1000

        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            tgt = torch.cat((ctx_buf[:, i + 1, :, :], tgt), dim=1)
            ctx_buf[:, i + 1, :, :] = tgt[:, -self.ctx_len:, :]

            tgt_ctx = self._causal_unfold(tgt)
            if self.use_pos_enc and i == 0:
                tgt_ctx = tgt_ctx + self.pos_enc(tgt_ctx)
            tgt = torch.zeros_like(tgt_ctx)[:, -self.chunk_size:, :]
            for i in range(int(math.ceil(tgt.shape[0] / K))):
                tgt[i*K:(i+1)*K], _sa_map, _ca_map = tf_dec_layer(
                    tgt_ctx[i*K:(i+1)*K], mem_ctx[i*K:(i+1)*K],
                    self.chunk_size)
            tgt = tgt.reshape(B, T, C)

        tgt = tgt.permute(0, 2, 1)
        if mod != 0:
            tgt = tgt[..., :-mod]

        return tgt, ctx_buf
    
class MaskNet(nn.Module):
    def __init__(self, enc_dim, num_enc_layers, dec_dim, dec_buf_len,
                 dec_chunk_size, num_dec_layers, use_pos_enc, skip_connection, proj):
        super(MaskNet, self).__init__()
        self.skip_connection = skip_connection
        self.proj = proj

        self.encoder = DilatedCausalConvEncoder(channels=enc_dim,
                                                num_layers=num_enc_layers)

        self.proj_e2d_e = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())
        self.proj_e2d_l = nn.Sequential(
            nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())
        self.proj_d2e = nn.Sequential(
            nn.Conv1d(dec_dim, enc_dim, kernel_size=1, stride=1, padding=0,
                      groups=dec_dim),
            nn.ReLU())
        
        self.decoder = CausalTransformerDecoder(
            model_dim=dec_dim, ctx_len=dec_buf_len, chunk_size=dec_chunk_size,
            num_layers=num_dec_layers, nhead=8, use_pos_enc=use_pos_enc,
            ff_dim=2 * dec_dim)

    def forward(self, x, l, enc_buf, dec_buf):
        """
        Generates a mask based on encoded input `e` and the one-hot
        label `label`.

        Args:
            x: [B, C, T]
                Input audio sequence
            l: [B, C]
                Label embedding
            ctx_buf: {[B, C, <receptive field of the layer>], ...}
                List of context buffers maintained by DCC encoder
        """
        e, enc_buf = self.encoder(x, enc_buf)
    
        l = l.unsqueeze(2) * e

        if self.proj:
            e = self.proj_e2d_e(e)
            m = self.proj_e2d_l(l)
            m, dec_buf = self.decoder(m, e, dec_buf)
        else:
            m, dec_buf = self.decoder(l, e, dec_buf)
        
        if self.proj:
            m = self.proj_d2e(m)
        
        if self.skip_connection:
            m = l + m
        
        l = l.squeeze(2)

        return m, l, enc_buf, dec_buf
    
class Net(nn.Module):
    def __init__(self, label_len, device, L=32,
                 enc_dim=256, num_enc_layers=10,
                 dec_dim=128, dec_buf_len=13, num_dec_layers=1,
                 dec_chunk_size=13, out_buf_len=4,
                 use_pos_enc=True, skip_connection=True, proj=True, lookahead=True):
        super(Net, self).__init__()
        self.L = L
        self.out_buf_len = out_buf_len
        self.enc_dim = enc_dim
        self.lookahead = lookahead
        self.device = device

        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=enc_dim, kernel_size=kernel_size, stride=L,
                      padding=0, bias=False),
            nn.ReLU())

        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.ReLU())
        
        self.proj_l = nn.Sequential(
            nn.Linear(enc_dim, 512),
            nn.Linear(512, label_len),
        )

        self.mask_gen = MaskNet(
            enc_dim=enc_dim, num_enc_layers=num_enc_layers,
            dec_dim=dec_dim, dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size, num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc, skip_connection=skip_connection, proj=proj)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=enc_dim, out_channels=1,
                kernel_size=(out_buf_len + 1) * L,
                stride=L,
                padding=out_buf_len * L, bias=False),
            nn.Tanh())

    def init_buffers(self, batch_size, device):
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.enc_dim, self.out_buf_len,
                              device=device)
        return enc_buf, dec_buf, out_buf

    def forward(self, x, label, init_enc_buf=None, init_dec_buf=None,
                init_out_buf=None, pad=True):
        
        #x = apply_noise_reduction(x, self.device)
          
        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            assert init_enc_buf is None and \
                   init_dec_buf is None and \
                   init_out_buf is None, \
                "Both buffers have to initialized, or " \
                "both of them have to be None."
            enc_buf, dec_buf, out_buf = self.init_buffers(
                x.shape[0], x.device)
        else:
            enc_buf, dec_buf, out_buf = \
                init_enc_buf, init_dec_buf, init_out_buf

        x = self.in_conv(x)

        l = self.label_embedding(label) # [B, label_len] --> [B, channels]

        m, l, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)
        

        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len:]
        x = self.out_conv(x)

        if mod != 0:
            x = x[:, :, :-mod]

        if init_enc_buf is None:
            return x, m
        else:
            return x, m, enc_buf, dec_buf, out_buf
