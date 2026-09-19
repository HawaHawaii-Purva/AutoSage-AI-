"""Voice transcription for AutoSage using faster-whisper."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from faster_whisper import WhisperModel


@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size: str = "small") -> WhisperModel:
    """Load and cache a multilingual Whisper model."""
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


WHISPER_LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
}


def transcribe_audio(audio_file: Any, language: str = "Auto-detect", model_size: str = "small") -> dict[str, Any]:
    """Transcribe a Streamlit audio upload and return text + detected language."""
    if audio_file is None:
        return {"ok": False, "text": "", "language": None, "error": "No audio was provided."}

    suffix = ".wav"
    name = getattr(audio_file, "name", "") or ""
    if "." in name:
        suffix = Path(name).suffix or suffix

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.getvalue())
            temp_path = tmp.name

        model = get_whisper_model(model_size)
        whisper_language = WHISPER_LANGUAGES.get(language)
        segments, info = model.transcribe(
            temp_path,
            language=whisper_language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        detected = getattr(info, "language", None)
        return {
            "ok": bool(text),
            "text": text,
            "language": detected,
            "language_probability": getattr(info, "language_probability", None),
            "error": "" if text else "No speech was detected. Try speaking closer to the microphone.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "language": None,
            "error": f"Voice transcription failed: {type(exc).__name__}: {exc}",
        }
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
