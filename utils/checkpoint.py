"""Load existing experiment weights using the paper's module names.

Only key names change. Tensors and optimizer state are never converted or
written back to the source checkpoint. New paper-named state dictionaries
also pass through this loader unchanged.
"""

from collections import OrderedDict
from collections.abc import Mapping
import pickle

import numpy as np
import torch


_ROOT_NAMES = {
    "dwt_hpf": "awfe",
    "freq_down1": "hwfp.stage1",
    "freq_down2": "hwfp.stage2",
    "freq_down3": "hwfp.stage3",
    "fusion_e3": "fcsm_e3",
    "fusion_d0": "fcsm_d0",
}


def extract_state_dict(payload):
    """Accept a bare state dictionary or a net/state_dict checkpoint."""
    if not isinstance(payload, Mapping):
        raise TypeError("Expected a state dictionary or a checkpoint mapping.")
    while True:
        nested = next(
            (payload[name] for name in ("net", "state_dict")
             if name in payload and isinstance(payload[name], Mapping)),
            None,
        )
        if nested is None:
            return payload
        payload = nested


def _rename_path(path):
    parts = path.split(".") if path else []
    while parts and parts[0] == "module":
        parts.pop(0)
    if not parts:
        return ""
    parts[0:1] = _ROOT_NAMES.get(parts[0], parts[0]).split(".")
    renamed = []
    for part in parts:
        if part == "ca":
            renamed.extend(("csa", "channel_attention"))
        elif part == "sa":
            renamed.extend(("csa", "spatial_attention"))
        else:
            renamed.append(part)
    return ".".join(renamed)


def remap_state_dict(payload):
    """Translate legacy keys and DataParallel prefixes without altering tensors."""
    state = extract_state_dict(payload)
    result = OrderedDict()
    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError("State-dictionary keys must be strings.")
        renamed = _rename_path(name)
        if renamed in result:
            raise ValueError(f"Multiple checkpoint keys map to {renamed!r}.")
        result[renamed] = value

    # BatchNorm and other versioned modules use this metadata when loading.
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        result._metadata = OrderedDict()
        for name, value in metadata.items():
            wrapper_root = not name or all(part == "module" for part in name.split("."))
            inner_root = f"{name}.module" if name else "module"
            if wrapper_root and inner_root in metadata:
                continue  # Retain only the innermost model root's metadata.
            renamed = _rename_path(name)
            if renamed in result._metadata:
                raise ValueError(f"Multiple metadata keys map to {renamed!r}.")
            result._metadata[renamed] = value
    return result


def load_checkpoint(path, map_location="cpu"):
    """Load weights, including NumPy scalar metrics in the original checkpoints.

The original training code saved its IoU as a NumPy float64. The restricted
unpickler needs these specific scalar/dtype classes; arbitrary pickle globals
are not enabled.
"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        numpy_core = getattr(np, "_core", None)
        if numpy_core is None:
            numpy_core = np.core
        scalar = numpy_core.multiarray.scalar
        scalar_types = [
            np.dtype,
            type(np.dtype(np.float64)),
            type(np.dtype(np.float32)),
            (scalar, "numpy.core.multiarray.scalar"),
            (scalar, "numpy._core.multiarray.scalar"),
        ]
        with torch.serialization.safe_globals(scalar_types):
            return torch.load(path, map_location=map_location, weights_only=True)
