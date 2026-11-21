"""
ComfyUI-Rumble2Flash
Custom nodes for Reson8/Rumble-2-Flash (Qwen3-ASR 0.5B) local PyTorch/Transformers inference.
Model page: https://huggingface.co/Reson8/Rumble-2-Flash
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
