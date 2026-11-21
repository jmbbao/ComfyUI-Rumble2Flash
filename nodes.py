import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import folder_paths
    from comfy.utils import ProgressBar
except ImportError:
    folder_paths = None
    ProgressBar = None

from .audio_utils import (
    comfy_audio_to_16k_mono,
    format_segments_subtitles,
    load_audio_file_or_url,
    split_audio_with_vad,
    stitch_overlapping_text,
)
from .rumble_core import (
    DEFAULT_MODEL_ID,
    SUPPORTED_LANGUAGES,
    apply_post_run_policy,
    get_or_load_handle,
    manage_vram,
    transcribe_array,
)


class Rumble2FlashLoader:
    """Loads Rumble-2-Flash from models/rumble_asr (or downloads it if not present yet)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "model_id": ("STRING", {"default": DEFAULT_MODEL_ID, "multiline": False}),
                "precision": (["auto", "bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "cuda"}),
                "attn_implementation": (["sdpa", "flash_attention_2", "eager"], {"default": "sdpa"}),
                "vram_policy": (["keep_in_vram", "offload_to_cpu", "unload_after_run"], {"default": "offload_to_cpu"}),
                # "local_files_only": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RUMBLE_MODEL",)
    RETURN_NAMES = ("rumble_model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/Rumble-2-Flash"
    DESCRIPTION = "Loads Rumble-2-Flash from models/rumble_asr (or downloads it if not present yet)"

    def load_model(
        self,
        # model_id: str,
        precision: str,
        device: str,
        attn_implementation: str,
        vram_policy: str,
        # local_files_only: bool,
    ):
        handle = get_or_load_handle(
            # model_id=model_id.strip() or DEFAULT_MODEL_ID,
            model_id=DEFAULT_MODEL_ID,
            precision=precision,
            device_choice=device,
            attn_implementation=attn_implementation,
            vram_policy=vram_policy,
            # local_files_only=local_files_only,
        )
        return (handle,)


class Rumble2FlashAudioInput:
    """Loads local audio or remote HTTP/HTTPS audio into a standard ComfyUI AUDIO tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_source": (
                    "STRING",
                    {
                        "default": "https://hf.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
                        "multiline": False,
                    },
                ),
                # "target_sample_rate": (["16000", "24000", "44100", "48000"], {"default": "16000"}),
                "force_mono": ("BOOLEAN", {"default": True}),
                "normalize_peak": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "duration_sec", "source_name")
    FUNCTION = "load_audio"
    CATEGORY = "audio/Rumble-2-Flash"
    DESCRIPTION = "Loads local audio or from internet (HTTP/HTTPS)"

    def load_audio(
        self,
        audio_source: str,
        # target_sample_rate: str,
        force_mono: bool,
        normalize_peak: bool,
    ):
        audio, duration_sec, source_name = load_audio_file_or_url(
            source=audio_source,
            # target_sample_rate=int(target_sample_rate),
            target_sample_rate=16000,
            force_mono=force_mono,
            normalize_peak=normalize_peak,
        )
        return (audio, duration_sec, source_name)


# This node is not used
class Rumble2FlashTranscribe:
    """Core single-pass transcription node mirroring Reson8/Rumble-2-Flash CLI options."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rumble_model": ("RUMBLE_MODEL",),
                "audio": ("AUDIO",),
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "context_prompt": ("STRING", {"default": "", "multiline": True}),
                "return_format": (["parsed", "transcription_only"], {"default": "parsed"}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "do_sample": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "RUMBLE_SEGMENTS")
    RETURN_NAMES = ("text", "detected_language", "parsed_json", "segments")
    FUNCTION = "transcribe"
    CATEGORY = "audio/Rumble-2-Flash"
    DESCRIPTION = "Core single-pass transcription node mirroring Reson8/Rumble-2-Flash CLI options"

    def transcribe(
        self,
        rumble_model,
        audio,
        language: str,
        context_prompt: str,
        return_format: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        seed: int,
    ):
        audio_16k, duration_sec = comfy_audio_to_16k_mono(audio)
        try:
            text, detected_lang, raw_obj = transcribe_array(
                handle=rumble_model,
                audio_16k=audio_16k,
                language=language,
                context_prompt=context_prompt,
                return_format=return_format,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        finally:
            apply_post_run_policy(rumble_model)

        segment = {
            "index": 0,
            "start": 0.0,
            "end": round(duration_sec, 3),
            "language": detected_lang,
            "text": text,
        }
        payload = {
            "model": rumble_model.model_id,
            "duration_sec": round(duration_sec, 3),
            "detected_language": detected_lang,
            "text": text,
            "raw": raw_obj,
        }
        return (text, detected_lang, json.dumps(payload, ensure_ascii=False, indent=2), [segment])


class Rumble2FlashLongAudio:
    """Transcribe using RMS/VAD-aware long audio chunker with overlap stitching and timestamped segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rumble_model": ("RUMBLE_MODEL",),
                "audio": ("AUDIO",),
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "chunk_duration_sec": ("INT", {"default": 28, "min": 8, "max": 60, "step": 2,
                   "tooltip": "Training was with audios of 30 seconds, it is recommended to use a value 28 - 45"}),
                "overlap_sec": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 8.0, "step": 0.5,
                   "tooltip": "Number of seconds to compare between chunks to prevent word repetitions"}),
                "silence_aware_split": ("BOOLEAN", {"default": True,
                   "tooltip": "The splitting in chunks is done in zones with silences (avoids splitting in zones with words)"}),
                "silence_threshold_db": ("INT", {"default": -38, "min": -60, "max": -15, "step": 1}),
                "max_new_tokens": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64,
                   "tooltip": "Maximum number of tokens per chunk, increase if the text is not whole.\nBigger number uses more VRAM"}),
                "digits": ("BOOLEAN", {"default": True,
                   "tooltip": "ON: Write numbers as digits instead of using words"}),
                "filler": ("BOOLEAN", {"default": False,
                   "tooltip": "ON: Keep filler words and hesitations in speech: eh, mmh, uh, ..."}),
                "verbatim": ("BOOLEAN", {"default": False,
                   "tooltip": "ON: Keep literal repetitions of words when speaking"}),                
                "context_prompt": ("STRING", {"default": "", "multiline": True,
                   "tooltip": "Enter the words that are not transcribed correctly,\nthey will appear in the final text as you write them here."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "RUMBLE_SEGMENTS", "STRING")
    RETURN_NAMES = ("full_text", "detected_language", "segments", "structured_json")
    FUNCTION = "transcribe_long"
    CATEGORY = "audio/Rumble-2-Flash"
    DESCRIPTION = "Transcribe using long audio chunker with overlap stitching and timestamped segments"

    def transcribe_long(
        self,
        rumble_model,
        audio,
        language: str,
        chunk_duration_sec: int,
        overlap_sec: float,
        silence_aware_split: bool,
        silence_threshold_db: int,
        max_new_tokens: int,
        digits: bool,
        filler: bool,
        verbatim: bool,
        context_prompt: str,
    ):
        audio_16k, duration_sec = comfy_audio_to_16k_mono(audio)
        chunks = split_audio_with_vad(
            audio_16k=audio_16k,
            chunk_duration_sec=float(chunk_duration_sec),
            overlap_sec=float(overlap_sec),
            silence_aware_split=silence_aware_split,
            silence_threshold_db=float(silence_threshold_db),
        )

        pbar = ProgressBar(len(chunks)) if ProgressBar is not None else None
        segments: List[Dict[str, Any]] = []
        stitched_parts: List[str] = []
        primary_lang = language

        try:
            for idx, (start_s, end_s, wave_chunk) in enumerate(chunks):
                seg_text, seg_lang, _ = transcribe_array(
                    handle=rumble_model,
                    audio_16k=wave_chunk,
                    language=language,
                    context_prompt=context_prompt,
                    return_format="parsed",
                    max_new_tokens=max_new_tokens,
                    digits=digits,
                    filler=filler,
                    verbatim=verbatim,
                )
                if primary_lang == "auto" and seg_lang and seg_lang != "auto":
                    primary_lang = seg_lang

                cleaned_text = (
                    stitch_overlapping_text(stitched_parts[-1], seg_text)
                    if stitched_parts
                    else seg_text
                )
                if cleaned_text:
                    stitched_parts.append(cleaned_text)
                    segments.append(
                        {
                            "index": idx,
                            "start": start_s,
                            "end": end_s,
                            "language": seg_lang,
                            "text": cleaned_text,
                        }
                    )
                if pbar is not None:
                    pbar.update(1)
        finally:
            apply_post_run_policy(rumble_model)

        full_text = " ".join(stitched_parts).strip()
        structured = {
            "model": rumble_model.model_id,
            "duration_sec": round(duration_sec, 3),
            "detected_language": primary_lang,
            "chunk_count": len(segments),
            "full_text": full_text,
            "segments": segments,
        }
        return (
            full_text,
            primary_lang,
            segments,
            json.dumps(structured, ensure_ascii=False, indent=2),
        )


class Rumble2FlashSubtitleExport:
    """Formats RUMBLE_SEGMENTS into SubRip (.srt) and WebVTT (.vtt), with optional disk export."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("RUMBLE_SEGMENTS",),
                "max_chars_per_line": ("INT", {"default": 50, "min": 20, "max": 84, "step": 2}),
                "max_lines_per_cue": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "export_format": (["both", "srt", "vtt"], {"default": "srt"}),
                "filename_prefix": ("STRING", {"default": "rumble_subtitles", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("srt_text", "vtt_text", "saved_path")
    FUNCTION = "export_subtitles"
    CATEGORY = "audio/Rumble-2-Flash"
    DESCRIPTION = "Formats RUMBLE_SEGMENTS into SubRip (.srt) and WebVTT (.vtt), with optional disk export"
    OUTPUT_NODE = True

    def export_subtitles(
        self,
        segments,
        max_chars_per_line: int,
        max_lines_per_cue: int,
        save_to_disk: bool,
        export_format: str,
        filename_prefix: str,
    ):
        srt_text, vtt_text = format_segments_subtitles(
            segments=segments or [],
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
        )

        saved_paths: List[str] = []
        if save_to_disk:
            if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
                out_dir = Path(folder_paths.get_output_directory())
            else:
                out_dir = Path.cwd() / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_prefix = Path(filename_prefix.strip() or "rumble_subtitles").name

            if export_format in ("both", "srt"):
                srt_path = out_dir / f"{safe_prefix}.srt"
                srt_path.write_text(srt_text, encoding="utf-8")
                saved_paths.append(str(srt_path))
            if export_format in ("both", "vtt"):
                vtt_path = out_dir / f"{safe_prefix}.vtt"
                vtt_path.write_text(vtt_text, encoding="utf-8")
                saved_paths.append(str(vtt_path))

        return (srt_text, vtt_text, ", ".join(saved_paths))


class Rumble2FlashBatchFolder:
    """Batch transcribes a folder of audio files with ComfyUI progress tracking and per-file export."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rumble_model": ("RUMBLE_MODEL",),
                "folder_path": ("STRING", {"default": "input/audio_batch", "multiline": False}),
                "extensions": ("STRING", {"default": ".wav,.mp3,.flac,.m4a,.ogg", "multiline": False}),
                "recursive": ("BOOLEAN", {"default": False,
                   "tooltip": "ON: Looks recursively in folders"}),
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "auto_chunk_over_sec": ("INT", {"default": 30, "min": 10, "max": 120, "step": 5}),
                "save_per_file": (["none", "txt", "srt", "json", "txt_and_srt", "all"], {"default": "txt_and_srt"}),
                "output_dir": ("STRING", {"default": "output/rumble_batch", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("combined_text", "batch_report_json", "file_count")
    FUNCTION = "transcribe_folder"
    CATEGORY = "audio/Rumble-2-Flash"
    DESCRIPTION = "Batch transcribes a folder of audio files with ComfyUI progress tracking and per-file export"
    OUTPUT_NODE = True

    def transcribe_folder(
        self,
        rumble_model,
        folder_path: str,
        extensions: str,
        recursive: bool,
        language: str,
        auto_chunk_over_sec: int,
        save_per_file: str,
        output_dir: str,
    ):
        root = Path(folder_path.strip())
        if not root.exists():
            raise FileNotFoundError(f"La carpeta de entrada no existe: '{root}'")

        allowed_exts = {
            e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in extensions.split(",")
            if e.strip()
        }
        iterator = root.rglob("*") if recursive else root.glob("*")
        files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in allowed_exts)

        out_root = Path(output_dir.strip() or "output/rumble_batch")
        if save_per_file != "none":
            out_root.mkdir(parents=True, exist_ok=True)

        pbar = ProgressBar(max(1, len(files))) if ProgressBar is not None else None
        report_items: List[Dict[str, Any]] = []
        combined_lines: List[str] = []

        try:
            for file_path in files:
                comfy_audio, duration_sec, source_name = load_audio_file_or_url(
                    str(file_path), target_sample_rate=16000, force_mono=True
                )
                audio_16k, _ = comfy_audio_to_16k_mono(comfy_audio)

                if duration_sec > float(auto_chunk_over_sec):
                    chunks = split_audio_with_vad(audio_16k, chunk_duration_sec=28.0, overlap_sec=2.0)
                    parts: List[str] = []
                    segments: List[Dict[str, Any]] = []
                    detected_lang = language
                    for idx, (start_s, end_s, chunk_wave) in enumerate(chunks):
                        t, l, _ = transcribe_array(rumble_model, chunk_wave, language=language)
                        if detected_lang == "auto" and l:
                            detected_lang = l
                        clean = stitch_overlapping_text(parts[-1], t) if parts else t
                        if clean:
                            parts.append(clean)
                            segments.append({"index": idx, "start": start_s, "end": end_s, "language": l, "text": clean})
                    text = " ".join(parts).strip()
                else:
                    text, detected_lang, _ = transcribe_array(rumble_model, audio_16k, language=language)
                    segments = [{"index": 0, "start": 0.0, "end": duration_sec, "language": detected_lang, "text": text}]

                item = {
                    "file": source_name,
                    "path": str(file_path),
                    "duration_sec": duration_sec,
                    "detected_language": detected_lang,
                    "text": text,
                    "segments": segments,
                }
                report_items.append(item)
                combined_lines.append(f"[{source_name}] ({detected_lang}): {text}")

                if save_per_file != "none":
                    stem = out_root / file_path.stem
                    if save_per_file in ("txt", "txt_and_srt", "all"):
                        stem.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
                    if save_per_file in ("srt", "txt_and_srt", "all"):
                        srt_text, vtt_text = format_segments_subtitles(segments)
                        stem.with_suffix(".srt").write_text(srt_text, encoding="utf-8")
                        if save_per_file == "all":
                            stem.with_suffix(".vtt").write_text(vtt_text, encoding="utf-8")
                    if save_per_file in ("json", "all"):
                        stem.with_suffix(".json").write_text(
                            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
                        )

                if pbar is not None:
                    pbar.update(1)
        finally:
            apply_post_run_policy(rumble_model)

        report_json = json.dumps(
            {"folder": str(root), "file_count": len(report_items), "items": report_items},
            ensure_ascii=False,
            indent=2,
        )
        if save_per_file != "none":
            (out_root / "_batch_report.json").write_text(report_json, encoding="utf-8")

        return ("\n".join(combined_lines), report_json, len(report_items))


# This node is not used
class Rumble2FlashVRAMManager:
    """Explicitly offloads or unloads Rumble-2-Flash from VRAM between ComfyUI pipeline stages."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rumble_model": ("RUMBLE_MODEL",),
                "action": (["offload_to_cpu", "unload_completely", "clear_cache_only"], {"default": "offload_to_cpu"}),
                "run_gc": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "passthrough_text": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("RUMBLE_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("rumble_model", "text", "vram_status")
    FUNCTION = "free_memory"
    CATEGORY = "audio/Rumble-2-Flash"
    DESCRIPTION = "Explicitly offloads or unloads Rumble-2-Flash from VRAM between ComfyUI pipeline stages"

    def free_memory(self, rumble_model, action: str, run_gc: bool, passthrough_text: str = ""):
        status = manage_vram(rumble_model, action=action, run_gc=run_gc)
        return (rumble_model, passthrough_text or "", status)


NODE_CLASS_MAPPINGS = {
    "Rumble2FlashLoader": Rumble2FlashLoader,
    "Rumble2FlashAudioInput": Rumble2FlashAudioInput,
    # Este nodo no hace falta pero no lo borro por si fuera necesario
    # "Rumble2FlashTranscribe": Rumble2FlashTranscribe,
    "Rumble2FlashLongAudio": Rumble2FlashLongAudio,
    "Rumble2FlashSubtitleExport": Rumble2FlashSubtitleExport,
    "Rumble2FlashBatchFolder": Rumble2FlashBatchFolder,
    # "Rumble2FlashVRAMManager": Rumble2FlashVRAMManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Rumble2FlashLoader": "Rumble-2-Flash Model Loader",
    "Rumble2FlashAudioInput": "Rumble-2-Flash Audio / URL Loader",
    # Este nodo no hace falta pero no lo borro por si fuera necesario
    # "Rumble2FlashTranscribe": "Rumble-2-Flash Transcribe",
    # Y renombro este para que sea ahora el Transcribe
    # "Rumble2FlashLongAudio": "Rumble-2-Flash Long Audio Chunking",
    "Rumble2FlashLongAudio": "Rumble-2-Flash Transcribe (with Chunking)",
    "Rumble2FlashSubtitleExport": "Rumble-2-Flash Subtitle Formatter (SRT/VTT)",
    "Rumble2FlashBatchFolder": "Rumble-2-Flash Batch Folder",
    # "Rumble2FlashVRAMManager": "Rumble-2-Flash VRAM Manager",
}
