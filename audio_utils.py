import io
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf
import torch

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import folder_paths
except ImportError:
    folder_paths = None


def resolve_input_audio_path(source: str) -> str:
    source = source.strip()
    if source.startswith(("http://", "https://")):
        return source

    direct = Path(source)
    if direct.exists():
        return str(direct.resolve())

    if folder_paths is not None and hasattr(folder_paths, "get_input_directory"):
        in_input = Path(folder_paths.get_input_directory()) / source
        if in_input.exists():
            return str(in_input.resolve())

    raise FileNotFoundError(f"Archivo de audio no encontrado: '{source}'")


def load_audio_file_or_url(
    source: str,
    target_sample_rate: int = 16000,
    force_mono: bool = True,
    normalize_peak: bool = False,
) -> Tuple[Dict[str, Any], float, str]:
    resolved = resolve_input_audio_path(source)
    temp_path = None

    try:
        if resolved.startswith(("http://", "https://")):
            resp = requests.get(resolved, timeout=60)
            resp.raise_for_status()
            parsed = urlparse(resolved)
            suffix = Path(parsed.path).suffix or ".audio"
            source_name = Path(parsed.path).name or "remote_audio"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resp.content)
                temp_path = tmp.name
            path_to_read = temp_path
        else:
            path_to_read = resolved
            source_name = Path(resolved).name

        waveform, sr = _read_waveform(path_to_read)

        if force_mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != target_sample_rate:
            waveform = _resample_waveform(waveform, sr, target_sample_rate)
            sr = target_sample_rate

        if normalize_peak:
            peak = float(waveform.abs().max())
            if peak > 1e-6:
                waveform = waveform * (0.89125 / peak)  # -1 dBFS

        duration_sec = round(waveform.shape[-1] / float(sr), 3)
        comfy_audio = {
            "waveform": waveform.unsqueeze(0).contiguous().float(),  # [1, C, T]
            "sample_rate": int(sr),
        }
        return comfy_audio, duration_sec, source_name
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _read_waveform(path: str) -> Tuple[torch.Tensor, int]:
    if torchaudio is not None:
        try:
            wave, sr = torchaudio.load(path)
            return wave.float(), int(sr)
        except Exception:
            pass

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.T.copy())
        return wave, int(sr)
    except Exception:
        if librosa is None:
            raise
        data, sr = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[None, :]
        return torch.from_numpy(data.astype(np.float32)), int(sr)


def _resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    if torchaudio is not None:
        return torchaudio.functional.resample(waveform, orig_sr, target_sr)
    if librosa is not None:
        channels = [
            librosa.resample(ch.cpu().numpy(), orig_sr=orig_sr, target_sr=target_sr)
            for ch in waveform
        ]
        return torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))
    # Pure PyTorch linear fallback
    new_len = int(round(waveform.shape[-1] * target_sr / float(orig_sr)))
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0)


def comfy_audio_to_16k_mono(audio: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 3:
        waveform = waveform[0]  # [C, T]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = _resample_waveform(waveform, sr, 16000)

    mono_np = waveform.squeeze(0).numpy().astype(np.float32)
    duration_sec = len(mono_np) / 16000.0
    return mono_np, duration_sec


def _rms_db(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    return 20.0 * float(np.log10(rms + 1e-12))


def split_audio_with_vad(
    audio_16k: np.ndarray,
    chunk_duration_sec: float = 28.0,
    overlap_sec: float = 2.0,
    silence_aware_split: bool = True,
    silence_threshold_db: float = -38.0,
) -> List[Tuple[float, float, np.ndarray]]:
    sr = 16000
    total_samples = len(audio_16k)
    if total_samples == 0:
        return []

    chunk_samples = max(int(chunk_duration_sec * sr), sr)
    overlap_samples = max(0, min(int(overlap_sec * sr), chunk_samples // 3))

    if total_samples <= chunk_samples:
        return [(0.0, round(total_samples / sr, 3), audio_16k)]

    chunks: List[Tuple[float, float, np.ndarray]] = []
    start = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)

        if end < total_samples and silence_aware_split and overlap_samples > 0:
            search_start = max(start + chunk_samples // 2, end - overlap_samples * 2)
            window = audio_16k[search_start:end]
            frame = int(0.05 * sr)  # 50 ms frames
            if len(window) >= frame * 2:
                energies = []
                for offset in range(0, len(window) - frame, frame // 2):
                    seg = window[offset : offset + frame]
                    energies.append((float(np.mean(seg * seg)), search_start + offset + frame // 2))
                if energies:
                    _, best_cut = min(energies, key=lambda x: x[0])
                    end = best_cut

        slice_wave = audio_16k[start:end]
        if _rms_db(slice_wave) >= silence_threshold_db or len(chunks) == 0:
            chunks.append((round(start / sr, 3), round(end / sr, 3), slice_wave))

        if end >= total_samples:
            break

        next_start = end - overlap_samples if not silence_aware_split else max(start + sr, end - overlap_samples // 2)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def stitch_overlapping_text(previous: str, current: str, max_overlap_words: int = 8) -> str:
    prev_words = previous.strip().split()
    curr_words = current.strip().split()
    if not prev_words or not curr_words:
        return current.strip()

    limit = min(max_overlap_words, len(prev_words), len(curr_words))
    for k in range(limit, 0, -1):
        tail = [w.lower().strip(".,;:!?¿¡") for w in prev_words[-k:]]
        head = [w.lower().strip(".,;:!?¿¡") for w in curr_words[:k]]
        if tail == head:
            return " ".join(curr_words[k:]).strip()
    return current.strip()


def _format_timestamp(seconds: float, vtt: bool = False) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1_000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _split_segment_into_cues(
    start: float,
    end: float,
    text: str,
    max_chars_per_line: int,
    max_lines_per_cue: int,
) -> List[Tuple[float, float, str]]:
    wrapped = textwrap.wrap(text.strip(), width=max(16, max_chars_per_line))
    if not wrapped:
        return []

    groups = [
        "\n".join(wrapped[i : i + max_lines_per_cue])
        for i in range(0, len(wrapped), max_lines_per_cue)
    ]
    if len(groups) == 1:
        return [(start, end, groups[0])]

    total_chars = sum(len(g) for g in groups) or 1
    duration = max(0.2, end - start)
    cues: List[Tuple[float, float, str]] = []
    cursor = start
    for idx, group in enumerate(groups):
        share = duration * (len(group) / total_chars)
        cue_end = end if idx == len(groups) - 1 else min(end, cursor + share)
        cues.append((cursor, cue_end, group))
        cursor = cue_end
    return cues


def format_segments_subtitles(
    segments: List[Dict[str, Any]],
    max_chars_per_line: int = 42,
    max_lines_per_cue: int = 2,
) -> Tuple[str, str]:
    srt_blocks: List[str] = []
    vtt_blocks: List[str] = ["WEBVTT\n"]
    cue_index = 1

    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 2.0))
        for cue_start, cue_end, cue_body in _split_segment_into_cues(
            start, end, seg_text, max_chars_per_line, max_lines_per_cue
        ):
            srt_blocks.append(
                f"{cue_index}\n{_format_timestamp(cue_start, vtt=False)} --> {_format_timestamp(cue_end, vtt=False)}\n{cue_body}\n"
            )
            vtt_blocks.append(
                f"{_format_timestamp(cue_start, vtt=True)} --> {_format_timestamp(cue_end, vtt=True)}\n{cue_body}\n"
            )
            cue_index += 1

    return "\n".join(srt_blocks).strip() + "\n", "\n".join(vtt_blocks).strip() + "\n"
