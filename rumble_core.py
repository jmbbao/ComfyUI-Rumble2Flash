import gc
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

try:
    from transformers import Qwen3ASRForConditionalGeneration
except ImportError:
    Qwen3ASRForConditionalGeneration = None

try:
    import folder_paths
    import comfy.model_management as mm
except ImportError:
    folder_paths = None
    mm = None

DEFAULT_MODEL_ID = "Reson8/Rumble-2-Flash"

SUPPORTED_LANGUAGES: List[str] = [
    "auto",
    "Arabic",
    "Basque",
    "Bengali",
    "Bulgarian",
    "Cantonese",
    "Catalan",
    "Chinese",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "Galician",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "German",
    "Indonesian",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Norwegian",
    "Persian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Serbian",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swedish",
    "Tagalog",
    "Tamil",
    "Telugu",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Vietnamese",
]

_MODEL_CACHE: Dict[str, "RumbleModelHandle"] = {}


@dataclass
class RumbleModelHandle:
    cache_key: str
    model_id: str
    local_dir: str
    device_str: str
    dtype: torch.dtype
    attn_implementation: str
    vram_policy: str
    local_files_only: bool
    processor: Any = None
    model: Any = None
    current_device: str = "unloaded"
    metadata: Dict[str, Any] = field(default_factory=dict)


# Esta versión no hacía caso a  extra_model_paths.yaml 
# donde podemos indicar una carpeta externa de /models/rumble_asr/ 
# 
# def get_rumble_models_root() -> Path:
#     if folder_paths is not None and hasattr(folder_paths, "models_dir"):
#         root = Path(folder_paths.models_dir) / "rumble_asr"
#     else:
#         root = Path.cwd() / "models" / "rumble_asr"
#     root.mkdir(parents=True, exist_ok=True)
#     return root

def get_rumble_models_root() -> Path:
    if folder_paths is not None:
        default_root = Path(folder_paths.models_dir) / "rumble_asr"
        try:
            paths = folder_paths.get_folder_paths("rumble_asr")
            # Prioriza la ruta externa de extra_model_paths.yaml frente a la carpeta por defecto
            for p in paths:
                p_path = Path(p)
                if p_path.resolve() != default_root.resolve() and p_path.exists():
                    p_path.mkdir(parents=True, exist_ok=True)
                    return p_path
            if paths:
                first = Path(paths[0])
                first.mkdir(parents=True, exist_ok=True)
                return first
        except Exception:
            pass
        default_root.mkdir(parents=True, exist_ok=True)
        return default_root

    root = Path.cwd() / "models" / "rumble_asr"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_choice == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_choice == "cpu":
        return torch.device("cpu")

    if mm is not None:
        return mm.get_torch_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if precision == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    if precision == "float16":
        return torch.float16
    if precision == "float32":
        return torch.float32
    # auto
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def ensure_local_model_dir(model_id: str, local_files_only: bool = False) -> str:
    candidate = Path(model_id)
    if candidate.exists() and candidate.is_dir():
        return str(candidate.resolve())

    models_root = get_rumble_models_root()
    safe_folder_name = model_id.replace("/", "--")
    target_dir = models_root / safe_folder_name

    # Also allow direct subfolder name inside ComfyUI/models/rumble_asr/<name>
    direct_sub = models_root / model_id
    if direct_sub.exists() and (direct_sub / "config.json").exists():
        return str(direct_sub.resolve())

    if target_dir.exists() and (target_dir / "config.json").exists():
        return str(target_dir.resolve())

    if local_files_only:
        raise FileNotFoundError(
            f"No se encontraron pesos locales en '{target_dir}'. "
            f"Desactiva 'local_files_only' para descargar automáticamente '{model_id}' desde Hugging Face "
            f"o copia los ficheros de https://huggingface.co/{model_id}/tree/main en esa carpeta."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return str(target_dir.resolve())


def _load_model_class(local_dir: str, dtype: torch.dtype, attn_implementation: str):
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation and attn_implementation != "eager":
        load_kwargs["attn_implementation"] = attn_implementation

    errors: List[str] = []
    for cls in (AutoModelForMultimodalLM, Qwen3ASRForConditionalGeneration):
        if cls is None:
            continue
        try:
            return cls.from_pretrained(local_dir, **load_kwargs)
        except TypeError:
            # Fallback for transformers versions using dtype= instead of torch_dtype=
            alt_kwargs = dict(load_kwargs)
            alt_kwargs["dtype"] = alt_kwargs.pop("torch_dtype")
            try:
                return cls.from_pretrained(local_dir, **alt_kwargs)
            except Exception as exc:
                errors.append(f"{cls.__name__}: {exc}")
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")

    raise RuntimeError(
        "No se pudo cargar Reson8/Rumble-2-Flash con Transformers. "
        "Asegúrate de tener transformers>=4.52.0 instalado. Detalles: " + " | ".join(errors)
    )


def get_or_load_handle(
    # model_id: str,
    model_id: str = DEFAULT_MODEL_ID, # al poner este con valor por defecto hay que ponerlo a todos
    precision: str = "bfloat16",
    device_choice: str = "auto",
    attn_implementation: str = "sdpa",
    vram_policy: str = "offload_to_cpu",
    local_files_only: bool = False,  # al poner este también
) -> RumbleModelHandle:
    device = resolve_device(device_choice)
    dtype = resolve_dtype(precision, device)
    local_dir = ensure_local_model_dir(model_id, local_files_only=local_files_only)

    cache_key = f"{local_dir}|{device}|{dtype}|{attn_implementation}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None and cached.model is not None and cached.processor is not None:
        cached.vram_policy = vram_policy
        return cached

    processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
    model = _load_model_class(local_dir, dtype=dtype, attn_implementation=attn_implementation)
    model.eval()

    if vram_policy == "keep_in_vram":
        model.to(device)
        current_device = str(device)
    else:
        current_device = "cpu"

    handle = RumbleModelHandle(
        cache_key=cache_key,
        model_id=model_id,
        local_dir=local_dir,
        device_str=str(device),
        dtype=dtype,
        attn_implementation=attn_implementation,
        vram_policy=vram_policy,
        local_files_only=local_files_only,
        processor=processor,
        model=model,
        current_device=current_device,
    )
    _MODEL_CACHE[cache_key] = handle
    return handle


def ensure_on_compute_device(handle: RumbleModelHandle) -> RumbleModelHandle:
    if handle.model is None or handle.processor is None:
        handle.processor = AutoProcessor.from_pretrained(handle.local_dir, local_files_only=True)
        handle.model = _load_model_class(
            handle.local_dir,
            dtype=handle.dtype,
            attn_implementation=handle.attn_implementation,
        )
        handle.model.eval()
        handle.current_device = "cpu"

    target = torch.device(handle.device_str)
    if handle.current_device != str(target):
        handle.model.to(target)
        handle.current_device = str(target)
    return handle


def manage_vram(handle: RumbleModelHandle, action: str, run_gc: bool = True) -> str:
    if action == "offload_to_cpu":
        if handle.model is not None and handle.current_device != "cpu":
            handle.model.to("cpu")
            handle.current_device = "cpu"
    elif action == "unload_completely":
        handle.model = None
        handle.processor = None
        handle.current_device = "unloaded"
        _MODEL_CACHE.pop(handle.cache_key, None)

    if run_gc:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if mm is not None:
        mm.soft_empty_cache()

    return json.dumps(
        {
            "action": action,
            "current_device": handle.current_device,
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            if torch.cuda.is_available()
            else 0.0,
        },
        ensure_ascii=False,
    )


def apply_post_run_policy(handle: RumbleModelHandle) -> None:
    if handle.vram_policy == "offload_to_cpu":
        manage_vram(handle, "offload_to_cpu", run_gc=False)
    elif handle.vram_policy == "unload_after_run":
        manage_vram(handle, "unload_completely", run_gc=True)


def _parse_decoded_output(decoded: Any, requested_language: str) -> Tuple[str, str, Dict[str, Any]]:
    raw_item = decoded[0] if isinstance(decoded, list) and len(decoded) > 0 else decoded

    if isinstance(raw_item, dict):
        text = str(raw_item.get("text") or raw_item.get("transcription") or "").strip()
        lang = str(raw_item.get("language") or requested_language or "unknown").strip()
        return text, lang, raw_item

    raw_str = str(raw_item).strip()
    # Handle Qwen3-ASR structured format: "language Spanish<asr_text>Hola mundo"
    match = re.search(r"language\s+([^<\n]+)\s*<asr_text>(.*)", raw_str, flags=re.IGNORECASE | re.DOTALL)
    if match:
        lang = match.group(1).strip()
        text = match.group(2).strip()
        return text, lang, {"language": lang, "text": text, "raw": raw_str}

    cleaned = re.sub(r"<\|?/?asr_text\|?>", "", raw_str).strip()
    lang = "auto" if requested_language == "auto" else requested_language
    return cleaned, lang, {"language": lang, "text": cleaned}


def transcribe_array(
    handle: RumbleModelHandle,
    audio_16k: np.ndarray,
    language: str = "auto",
    context_prompt: str = "",
    return_format: str = "parsed",
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int = 42,
    digits: bool = True,
    filler: bool = False,
    verbatim: bool = False,
) -> Tuple[str, str, Dict[str, Any]]:
    ensure_on_compute_device(handle)
    processor = handle.processor
    model = handle.model

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    req_kwargs: Dict[str, Any] = {"audio": audio_16k}
    if language and language != "auto":
        req_kwargs["language"] = language
    if context_prompt and context_prompt.strip():
        req_kwargs["prompt"] = context_prompt.strip()

    req_kwargs["digits"] = bool(digits)
    req_kwargs["filler"] = bool(filler)
    req_kwargs["verbatim"] = bool(verbatim)
    
    try:
        inputs = processor.apply_transcription_request(**req_kwargs)
    except TypeError:
        # Fallback if a processor version omits prompt kwarg
        req_kwargs.pop("prompt", None)
        inputs = processor.apply_transcription_request(**req_kwargs)

    inputs = inputs.to(model.device, dtype=model.dtype)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample and temperature > 0.0:
        gen_kwargs["temperature"] = float(temperature)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, prompt_len:]

    try:
        decoded = processor.decode(generated_ids, return_format=return_format)
    except TypeError:
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return _parse_decoded_output(decoded, language)
