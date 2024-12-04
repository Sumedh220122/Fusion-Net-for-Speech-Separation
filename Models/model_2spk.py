from speechbrain.lobes.models.dual_path import Encoder
from speechbrain.lobes.models.dual_path import SBTransformerBlock
from speechbrain.lobes.models.dual_path import Dual_Path_Model
from speechbrain.lobes.models.dual_path import Decoder
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper
from speechbrain.nnet.schedulers import ReduceLROnPlateau
import torch.nn as nn

intertrans = SBTransformerBlock(
                    num_layers = 8,
                    d_model = 256,
                    nhead = 8,
                    d_ffn = 1024,
                    dropout = 0,
                    use_positional_encoding = True,
                    norm_before = True
                )
        
intratrans = SBTransformerBlock(
            num_layers = 8,
            d_model = 256,
            nhead = 8,
            d_ffn = 1024,
            dropout = 0,
            use_positional_encoding = True,
            norm_before = True
        )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(kernel_size = 16, out_channels = 256)

        self.mknn = Dual_Path_Model(
                    num_spks = 2,
                    in_channels = 256,
                    out_channels = 256,
                    num_layers = 2,
                    K = 250,
                    intra_model = intratrans,
                    inter_model = intertrans,
                    norm = "ln",
                    linear_layer_after_inter_intra = False,
                    skip_around_intra = True
                )
        
        self.dec = Decoder(
                    kernel_size = 16,
                    in_channels = 256,
                    out_channels = 1,
                    stride = 8,
                )
    
    def forward(self, x):
        x = self.enc(x)
        x = self.mknn(x)
        x = x.squeeze(1)
        x = self.dec(x)
        x = x.unsqueeze(1)
        
        return x
