import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

def optimizer(model, data_parallel=False, **kwargs):
    return optim.Adam(model.parameters(), **kwargs)

def loss(pred, tgt):
    return -0.9 * snr(pred, tgt).mean() - 0.1 * si_snr(pred, tgt).mean()

def metrics(mixed, output, gt):
    """ Function to compute metrics """
    metrics = {}

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        metrics[m_fn.__name__] = metric_i(m_fn,
                                          mixed[:, :gt.shape[1], :],
                                          output,
                                          gt)

    return metrics