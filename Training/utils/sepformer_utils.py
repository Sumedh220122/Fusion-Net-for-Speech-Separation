
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

def optimizer(model, data_parallel=False, **kwargs):
    return torch.optim.Adam(model.parameters(), **kwargs)

def loss(pred, tgt):
    sisnr = ScaleInvariantSignalNoiseRatio()
    return -sisnr(pred, tgt)