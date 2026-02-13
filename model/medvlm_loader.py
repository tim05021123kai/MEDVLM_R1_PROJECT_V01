"""
MedVLM-R1 model loader with automatic GPU detection and dtype optimization.
"""

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch


def _detect_device_and_dtype():
    """Auto-detect the best device and dtype for inference."""
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.float16
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_map = "mps"
        dtype = torch.float16
        device_label = "Apple MPS"
    else:
        device_map = "cpu"
        dtype = torch.float32
        device_label = "CPU"
    return device_map, dtype, device_label


def load_medvlm_model():
    """
    Load MedVLM-R1 model and processor with automatic device/dtype selection.

    Returns:
        tuple: (model, processor)
    """
    MODEL_PATH = "JZPeterPan/MedVLM-R1"

    device_map, dtype, device_label = _detect_device_and_dtype()
    print(f"Detected device: {device_label} | dtype: {dtype}")
    print("Downloading / loading model, this may take a few minutes...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    return model, processor
