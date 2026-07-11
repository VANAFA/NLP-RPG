"""Activation steering via Contrastive Activation Addition (CAA).

Method: run pairs of short texts that differ only in whether they contain
an anachronistic/sci-fi element ("positive" = in-world fantasy phrasing,
"negative" = the same idea with a modern/sci-fi element), capture the
residual-stream hidden state at the last token of each at one decoder
layer, and take mean(positive) - mean(negative) as the steering direction.
During generation, `alpha * unit_direction` is added to that layer's output
at every token via a forward hook — nudging the model's internal
representation away from "anachronism" and toward "fantasy" without
changing any weights. Reference: Rimsky et al. 2023, "Steering Llama 2 via
Contrastive Activation Addition"; Turner et al. 2023, "Activation Addition".
"""

from typing import Dict, List, Optional

import torch


def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Locate the transformer's decoder layer stack regardless of whether
    `model` is a plain AutoModelForCausalLM or a PeftModel wrapping one —
    both shapes are used in this harness (base vs. LoRA-adapted model)."""
    for accessor in (
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
    ):
        try:
            layers = accessor(model)
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate decoder layers on this model. Inspect its module "
        "tree (print(model)) and add the right accessor to get_decoder_layers()."
    )


def pick_layer_index(model, fraction: float) -> int:
    num_layers = len(get_decoder_layers(model))
    return max(0, min(num_layers - 1, round(num_layers * fraction)))


@torch.no_grad()
def compute_steering_vector(
    model, tokenizer, layer_idx: int, pairs: List[Dict[str, str]]
) -> torch.Tensor:
    """Returns a unit-norm steering direction (magnitude is applied
    separately via alpha at generation time, see SteeringController)."""
    layers = get_decoder_layers(model)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["last_token"] = hidden[:, -1, :].detach()

    handle = layers[layer_idx].register_forward_hook(hook)
    device = next(model.parameters()).device
    try:
        pos_vecs, neg_vecs = [], []
        for pair in pairs:
            for text, bucket in ((pair["positive"], pos_vecs), (pair["negative"], neg_vecs)):
                inputs = tokenizer(text, return_tensors="pt").to(device)
                model(**inputs)
                bucket.append(captured["last_token"].squeeze(0).float().cpu())
    finally:
        handle.remove()

    pos_mean = torch.stack(pos_vecs).mean(dim=0)
    neg_mean = torch.stack(neg_vecs).mean(dim=0)
    direction = pos_mean - neg_mean
    return direction / direction.norm().clamp_min(1e-6)


class SteeringController:
    """Context manager: while active, adds `alpha * vector` to decoder layer
    `layer_idx`'s output hidden states on every forward pass (i.e. every
    generated token). A no-op if `vector` is None, so the same generation
    call site serves both steered and unsteered runs."""

    def __init__(self, model, layer_idx: int, vector: Optional[torch.Tensor], alpha: float):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = vector
        self.alpha = alpha
        self._handle = None

    def _hook(self, _module, _inputs, output):
        shift = (self.alpha * self.vector)
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + shift.to(hidden.dtype).to(hidden.device)
            return (hidden, *output[1:])
        return output + shift.to(output.dtype).to(output.device)

    def __enter__(self):
        if self.vector is not None:
            layers = get_decoder_layers(self.model)
            self._handle = layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
