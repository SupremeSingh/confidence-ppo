"""Immutable, causal online readout of an offline-trained bottleneck."""
from pathlib import Path
import torch
from math_rl.provenance import sha256
from math_rl.sparse_bottleneck import Bottleneck
from math_rl.critic_layers import sampled_block_states
from math_rl.direct_jlens import rms_coordinates


def load_filter(path, checksum):
    if not checksum or sha256(path) != checksum:
        raise ValueError('Bottleneck checkpoint checksum mismatch')
    record = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = Bottleneck(record['mean'].numel(), record['width'])
    model.load_state_dict(record['state'], strict=True)
    model.eval().requires_grad_(False)
    if (record['width'] != 768 or not torch.isfinite(record['mean']).all()
            or not torch.isfinite(record['scale']).all() or (record['scale'] <= 0).any()):
        raise ValueError('Expected a finite frozen 768-dimensional filter')
    return model, record['mean'].float(), record['scale'].float()


@torch.inference_mode()
def bottleneck_prefixes(encoder, ids, masks, response_width, fitted, layers, eps):
    model, mean, scale = fitted
    device = next(encoder.parameters()).device
    model.to(device)
    result = []
    try:
        for row, mask in zip(ids, masks, strict=True):
            prompt = int(mask[:-response_width].sum())
            response = int(mask[-response_width:].sum())
            tokens = row[mask.bool()].unsqueeze(0).to(device)
            blocks = sampled_block_states(encoder, tokens, prompt, response, list(range(response)), layers)
            x = torch.cat([rms_coordinates(blocks[l].float(), eps) for l in layers], 1)
            if x.shape[1] != mean.numel():
                raise ValueError('Offline/online feature dimensions differ')
            z = torch.cat([model.features(((v-mean)/scale).to(device)).cpu() for v in x.split(512)])
            if not torch.isfinite(z).all():
                raise ValueError('Nonfinite bottleneck features')
            result.append(z.numpy().copy())
    finally:
        model.cpu()
    return result
