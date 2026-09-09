"""Resolve the installed MMH3Tools package through its public node registration."""
import importlib

import nodes


def _mmh3_module(name):
    # Resolve the package ComfyUI actually loaded, independent of folder name/order.
    sampler = nodes.NODE_CLASS_MAPPINGS.get("MMH3LoopingSampler")
    if sampler is None:
        raise RuntimeError("Install/update ComfyUI-MMH3Tools and restart to use the hybrid sampler.")
    package = sampler.__module__.rsplit(".", 1)[0]
    return importlib.import_module(f"{package}.{name}")
