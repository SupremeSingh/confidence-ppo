"""Matched causal probes at one-third, two-thirds and final model depth."""
from contextlib import contextmanager
import math
import torch


def layer_spec(n):
    if n < 3:
        raise ValueError('Layer comparison needs at least three transformer blocks')
    return {'third': math.ceil(n/3), 'two_thirds': math.ceil(2*n/3), 'final': n}


@torch.inference_mode()
def sampled_layer_states(backbone, tokens, prompt_length, response_length, positions):
    if (tokens.ndim != 2 or tokens.shape != (1, prompt_length + response_length)
            or prompt_length < 1 or response_length < 1
            or any(p < 0 or p >= response_length for p in positions)):
        raise ValueError('Invalid causal prefix indices')
    spec = layer_spec(len(backbone.layers))
    captured, hooks = {}, []
    indices = torch.tensor(positions, device=tokens.device) + prompt_length - 1
    def capture(name):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden[0, indices].detach().cpu().clone()
        return hook
    try:
        for name in ('third', 'two_thirds'):
            hooks.append(backbone.layers[spec[name]-1].register_forward_hook(capture(name)))
        hooks.append(backbone.norm.register_forward_hook(capture('final')))
        backbone(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != set(spec) or any(not torch.isfinite(v).all() for v in captured.values()):
        raise ValueError('Missing or nonfinite layer features')
    return captured


def copy_backbone_state(encoder, actor_state):
    """HF causal-LM full state dict to HF backbone; strict names/shapes, no vocab head."""
    state = {k.removeprefix('model.'): v for k, v in actor_state.items() if k.startswith('model.')}
    encoder.load_state_dict(state, strict=True)
    encoder.eval().requires_grad_(False)


@torch.inference_mode()
def sampled_block_states(backbone, tokens, prompt_length, response_length, positions, layers):
    """Causal pre-action block outputs, including the last block BEFORE final norm."""
    if (tokens.ndim != 2 or tokens.shape != (1, prompt_length+response_length)
            or prompt_length < 1 or response_length < 1 or not positions
            or any(p < 0 or p >= response_length for p in positions)
            or not layers or len(set(layers)) != len(layers)
            or any(layer < 0 or layer >= len(backbone.layers) for layer in layers)):
        raise ValueError('Invalid block/prefix selection')
    indices=torch.tensor(positions,device=tokens.device)+prompt_length-1
    captured, hooks = {}, []
    def capture(layer):
        def hook(module, inputs, output):
            hidden=output[0] if isinstance(output,tuple) else output
            captured[layer]=hidden[0,indices].detach().cpu().clone()
        return hook
    try:
        for layer in layers:
            hooks.append(backbone.layers[layer].register_forward_hook(capture(layer)))
        backbone(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,return_dict=True)
    finally:
        for hook in hooks:hook.remove()
    if set(captured)!=set(layers) or any(not torch.isfinite(v).all() for v in captured.values()):
        raise ValueError('Missing or nonfinite block features')
    return captured
