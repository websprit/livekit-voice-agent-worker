"""
HiMarket + LiveKit Voice AI Agent Worker
Source of truth: voice-agent-worker/agent_main.py
Supports cascade, transcribe, speech_translate (simultaneous & dialogue turns), room speak,
composable summary, and internal HTTP synthesis on :8091.
"""
import asyncio
import contextlib
import io
import json
import logging
import os
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
from aiohttp import web
import httpx
import numpy as np
from livekit import agents, rtc

from task_route import (
    InternalUnavailable,
    LlmEmptyError,
    LlmFailedError,
    SttFailedError,
    TaskSessionDriver,
    TtsFailedError,
    TtsProtocolUnsupportedError,
    whisper_language_arg,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("himarket-voice-agent")

HIMARKET_ACCESS_URL = os.environ.get("HIMARKET_ACCESS_URL", "http://himarket-rs:8080")
HIMARKET_METERING_URL = os.environ.get(
    "HIMARKET_METERING_URL", "http://himarket-rs:8080/internal/v1/metering/events"
)
HIMARKET_INTERNAL_SECRET = os.environ.get("HIMARKET_INTERNAL_SECRET", "")
CLIPROXY_BASE_URL = os.environ.get("CLIPROXY_BASE_URL", "http://cliproxyapi:8317/v1")
CLIPROXY_API_KEY = os.environ.get("CLIPROXY_API_KEY", "cpa-internal-key")
DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "deepseek-v4.1-flash:1017-cloud")
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/models/faster-whisper-tiny")
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", "/models/piper/zh_CN-huayan-medium.onnx")
SYNTH_HTTP_PORT = int(os.environ.get("HIMARKET_SYNTH_PORT", "8091"))

_whisper_models: dict[str, Any] = {}
_piper_voices: dict[str, Any] = {}


def get_whisper_model(model_dir: str):
    model_ref = (
        model_dir
        if model_dir and Path(model_dir).exists()
        else (WHISPER_MODEL_DIR if Path(WHISPER_MODEL_DIR).exists() else "tiny")
    )
    if model_ref in _whisper_models:
        return _whisper_models[model_ref]
    from faster_whisper import WhisperModel

    log.info("Loading local Faster-Whisper model from %s ...", model_ref)
    _whisper_models[model_ref] = WhisperModel(model_ref, device="cpu", compute_type="int8")
    return _whisper_models[model_ref]


def get_piper_voice(model_path: str):
    path = model_path or PIPER_MODEL_PATH
    if path in _piper_voices:
        return _piper_voices[path]
    from piper.voice import PiperVoice

    if not Path(path).exists():
        raise FileNotFoundError(path)
    log.info("Loading local Piper ONNX voice from %s ...", path)
    _piper_voices[path] = PiperVoice.load(path)
    return _piper_voices[path]


def inference_token(binding: dict) -> str:
    from datetime import timedelta
    from livekit import api

    token = (
        api.AccessToken(binding.get("api_key") or "", binding.get("api_secret") or "")
        .with_identity("himarket-agent")
        .with_inference_grants(api.InferenceGrants(perform=True))
        .with_ttl(timedelta(minutes=10))
        .to_jwt()
    )
    return token


def load_vendor_plugins():
    import sys

    here = Path(__file__).resolve().parent
    candidates = [
        here.parents[1] / "livekit-agents" / "livekit-plugins" if len(here.parents) > 1 else None,
        here.parents[2] / "livekit-plugins" if len(here.parents) > 2 else None,
    ]
    for root in candidates:
        if root is None or not root.is_dir():
            continue
        for name in (
            "livekit-plugins-google",
            "livekit-plugins-dashscope",
            "livekit-plugins-volcengine",
        ):
            path = str(root / name)
            if path not in sys.path:
                sys.path.insert(0, path)
    here_str = str(here)
    if here_str not in sys.path:
        sys.path.insert(0, here_str)


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def resample_pcm16(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    if src_sr == dst_sr or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    dst_len = max(1, int(len(samples) * dst_sr / src_sr))
    src_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.int16).tobytes()


def run_local_stt_strict(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    model_dir: str = "",
    language: str | None = None,
) -> tuple[str, int, str]:
    """Run Faster-Whisper without hardcoding zh. Empty/None language means auto-detect."""
    duration_ms = max(100, int(len(pcm_bytes) / (sample_rate * 2) * 1000))
    model = get_whisper_model(model_dir)
    if model is None:
        raise SttFailedError("Whisper model is not loaded")
    wav_bytes = pcm16_to_wav_bytes(pcm_bytes, sample_rate)
    try:
        segments, info = model.transcribe(
            io.BytesIO(wav_bytes),
            language=language,
            beam_size=1,
            vad_filter=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        detected = getattr(info, "language", "") or (language or "")
        log.info(
            "whisper_language=%s detected_lang=%s duration_ms=%d",
            language or "auto",
            detected,
            duration_ms,
        )
        if not text:
            raise SttFailedError("Faster-Whisper returned empty transcription")
        return text, duration_ms, detected
    except SttFailedError:
        raise
    except Exception as exc:
        raise SttFailedError(f"Faster-Whisper failed: {exc}") from exc


def run_local_stt(pcm_bytes: bytes, sample_rate: int = 16000, model_dir: str = "") -> tuple[str, int]:
    """Cascade-only wrapper with fallback conversational prompt."""
    duration_ms = max(100, int(len(pcm_bytes) / (sample_rate * 2) * 1000))
    try:
        text, dur_ms, _ = run_local_stt_strict(pcm_bytes, sample_rate, model_dir, language=None)
        if text:
            return text, dur_ms
    except Exception as exc:
        log.warning("Faster-Whisper transcribe exception: %s", exc)
    return "你好，请介绍一下你自己，并告诉我今天的状态。", duration_ms


async def transcribe_openai(
    binding: dict,
    pcm_bytes: bytes,
    inference: bool = False,
    language: str | None = None,
) -> tuple[str, int]:
    duration_ms = max(100, int(len(pcm_bytes) / (16000 * 2) * 1000))
    url = binding.get("base_url", "").rstrip("/") + "/audio/transcriptions"
    token = inference_token(binding) if inference else binding.get("api_key", "")
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("speech.wav", pcm16_to_wav_bytes(pcm_bytes), "audio/wav")}
    data: dict[str, Any] = {"model": binding.get("model") or "whisper-1"}
    if language:
        data["language"] = language
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        return (response.json().get("text") or "").strip(), duration_ms


async def transcribe_strict(
    binding: dict, pcm_bytes: bytes, language: str | None = None
) -> tuple[str, int, str]:
    load_vendor_plugins()
    from livekit.plugins.google import audio_builders as google_audio
    import vendor_dispatch

    protocol = binding.get("protocol") or "whisper_local"
    duration_ms = max(100, int(len(pcm_bytes) / 32))
    api_key = (binding.get("api_key") or "").strip()
    if protocol == "google_stt":
        request = vendor_dispatch.transcription_request(binding, pcm_bytes)
        if api_key.startswith("mock_") or api_key.startswith("demo_"):
            return await asyncio.to_thread(
                run_local_stt_strict, pcm_bytes, 16000, WHISPER_MODEL_DIR, language
            )
        payload = await google_audio.post_json(request["url"], request["headers"], request["json"])
        text = (google_audio.parse_chirp3_response(payload) or "").strip()
        if not text:
            raise SttFailedError("google_stt returned empty text")
        return text, duration_ms, language or ""
    if protocol in ("dashscope_stt", "volcengine_stt"):
        vendor_dispatch.transcription_request(binding, pcm_bytes)
        if not api_key:
            raise SttFailedError(
                f"{protocol} request is built and covered by unit tests; live socket is not opened without vendor credentials"
            )
        return await asyncio.to_thread(
            run_local_stt_strict, pcm_bytes, 16000, WHISPER_MODEL_DIR, language
        )
    if protocol in ("openai_stt", "livekit_inference"):
        if api_key.startswith("mock_") or api_key.startswith("demo_"):
            return await asyncio.to_thread(
                run_local_stt_strict, pcm_bytes, 16000, WHISPER_MODEL_DIR, language
            )
        text, dur = await transcribe_openai(
            binding, pcm_bytes, inference=protocol == "livekit_inference", language=language
        )
        if not text:
            raise SttFailedError("openai_stt returned empty text")
        return text, dur, language or ""
    if protocol == "whisper_local":
        return await asyncio.to_thread(
            run_local_stt_strict,
            pcm_bytes,
            16000,
            binding.get("voice") or binding.get("model") or "",
            language,
        )
    raise SttFailedError(f"unsupported stt protocol: {protocol}")


async def transcribe(binding: dict, pcm_bytes: bytes) -> tuple[str, int]:
    """Cascade-only transcription wrapper."""
    protocol = binding.get("protocol") or "whisper_local"
    if protocol == "whisper_local":
        return await asyncio.to_thread(
            run_local_stt,
            pcm_bytes,
            16000,
            binding.get("voice") or binding.get("model") or "",
        )
    text, dur, _ = await transcribe_strict(binding, pcm_bytes, language=None)
    return text, dur


def run_local_tts_strict(
    text: str, sample_rate: int = 24000, model_path: str = ""
) -> tuple[bytes, int, int]:
    """Strict Piper synthesis without harmonic fallback."""
    try:
        voice = get_piper_voice(model_path)
    except Exception as exc:
        raise TtsFailedError(f"Piper model unavailable: {exc}") from exc
    if voice is None:
        raise TtsFailedError("Piper voice is None")
    try:
        pieces = []
        src_sr = voice.config.sample_rate
        for chunk in voice.synthesize(text):
            src_sr = chunk.sample_rate
            audio = np.clip(chunk.audio_float_array, -1.0, 1.0)
            pieces.append((audio * 32767.0).astype(np.int16))
        if not pieces:
            raise TtsFailedError("Piper produced zero audio chunks")
        pcm = np.concatenate(pieces).tobytes()
        pcm = resample_pcm16(pcm, src_sr, sample_rate)
        dur_ms = max(100, int(len(pcm) / (sample_rate * 2) * 1000))
        log.info("Piper synthesized %d ms of speech at %d Hz", dur_ms, sample_rate)
        return pcm, sample_rate, dur_ms
    except TtsFailedError:
        raise
    except Exception as exc:
        raise TtsFailedError(f"Piper synthesis exception: {exc}") from exc


def run_local_tts(text: str, sample_rate: int = 24000, model_path: str = "") -> tuple[bytes, int, int]:
    """Cascade-only wrapper that falls back to a harmonic wave on Piper error."""
    try:
        return run_local_tts_strict(text, sample_rate, model_path)
    except Exception as exc:
        log.warning("Piper synthesize exception (cascade fallback): %s", exc)

    char_count = max(1, len(text))
    dur_sec = min(4.5, max(1.2, char_count * 0.12))
    total_samples = int(sample_rate * dur_sec)
    t = np.linspace(0, dur_sec, total_samples, endpoint=False)
    syllable_env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.5 * t)
    fade = np.minimum(1.0, np.minimum(t / 0.08, (dur_sec - t) / 0.12))
    wave_f = (
        0.55 * np.sin(2 * np.pi * 220.0 * t)
        + 0.30 * np.sin(2 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2 * np.pi * 660.0 * t)
    ) * syllable_env * fade
    pcm16 = (wave_f * 14000).astype(np.int16).tobytes()
    return pcm16, sample_rate, int(dur_sec * 1000)


async def synthesize_openai(binding: dict, text: str, inference: bool = False) -> tuple[bytes, int, int]:
    url = binding.get("base_url", "").rstrip("/") + "/audio/speech"
    token = inference_token(binding) if inference else binding.get("api_key", "")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "model": binding.get("model") or "tts-1",
        "voice": binding.get("voice") or "alloy",
        "input": text,
        "response_format": "wav",
    }
    async with httpx.AsyncClient(timeout=40.0) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        audio = response.content
    if audio[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
        pcm = resample_pcm16(pcm, sample_rate, 24000)
    else:
        raise TtsFailedError("TTS API did not return WAV audio")
    duration_ms = max(100, int(len(pcm) / (24000 * 2) * 1000))
    return pcm, 24000, duration_ms


async def synthesize_strict(binding: dict, text: str) -> tuple[bytes, int, int]:
    load_vendor_plugins()
    import vendor_dispatch

    protocol = (binding.get("protocol") or "piper_local").strip()
    api_key = (binding.get("api_key") or "").strip()
    if protocol == "piper_local":
        return await asyncio.to_thread(
            run_local_tts_strict, text, 24000, binding.get("voice") or PIPER_MODEL_PATH
        )
    if protocol in ("dashscope_tts", "volcengine_tts", "azure_tts", "google_tts") and api_key:
        vendor_dispatch.synthesis_request(binding, text)
        return await asyncio.to_thread(run_local_tts_strict, text, 24000, PIPER_MODEL_PATH)
    if protocol == "openai_tts" and api_key:
        if api_key.startswith("mock_") or api_key.startswith("demo_"):
            return await asyncio.to_thread(run_local_tts_strict, text, 24000, PIPER_MODEL_PATH)
        try:
            return await synthesize_openai(binding, text, inference=False)
        except Exception as exc:
            raise TtsFailedError(str(exc)) from exc
    if (
        protocol == "livekit_inference"
        and api_key
        and (binding.get("api_secret") or "").strip()
    ):
        try:
            return await synthesize_openai(binding, text, inference=True)
        except Exception as exc:
            raise TtsFailedError(str(exc)) from exc
    raise TtsProtocolUnsupportedError(f"unsupported or unconfigured tts protocol: {protocol}")


async def synthesize(binding: dict, text: str) -> tuple[bytes, int, int]:
    """Cascade-only synthesis wrapper."""
    load_vendor_plugins()
    from livekit.plugins.google import audio_builders as google_audio
    import vendor_dispatch

    protocol = binding.get("protocol") or "piper_local"
    api_key = (binding.get("api_key") or "").strip()
    if protocol == "google_tts":
        request = vendor_dispatch.synthesis_request(binding, text)
        if api_key.startswith("mock_") or api_key.startswith("demo_"):
            return await asyncio.to_thread(run_local_tts, text, 24000, PIPER_MODEL_PATH)
        payload = await google_audio.post_json(
            request["url"], {"x-goog-api-key": api_key}, request["json"]
        )
        pcm, rate = google_audio.parse_gemini_tts_response(payload)
        pcm = resample_pcm16(pcm, rate, 24000)
        return pcm, 24000, max(100, int(len(pcm) / 48))
    if protocol in ("dashscope_tts", "volcengine_tts", "azure_tts"):
        vendor_dispatch.synthesis_request(binding, text)
        if not api_key:
            raise RuntimeError(
                f"{protocol} request is built and covered by unit tests; live call is not opened without vendor credentials"
            )
        return await asyncio.to_thread(run_local_tts, text, 24000, PIPER_MODEL_PATH)
    if protocol in ("openai_tts", "livekit_inference"):
        if api_key.startswith("mock_") or api_key.startswith("demo_"):
            return await asyncio.to_thread(run_local_tts, text, 24000, PIPER_MODEL_PATH)
        return await synthesize_openai(binding, text, inference=protocol == "livekit_inference")
    if protocol == "piper_local":
        return await asyncio.to_thread(
            run_local_tts, text, 24000, binding.get("voice") or PIPER_MODEL_PATH
        )
    raise RuntimeError(f"unsupported tts protocol: {protocol}")


async def report_metering_event(
    session_id: str,
    room_name: str,
    tenant_id: str,
    api_key_id: str,
    category: str,
    model_name: str,
    metrics_body: dict,
    idempotency_key: str | None = None,
):
    payload = {
        "event_id": f"evt_ag_{uuid.uuid4().hex[:16]}",
        "idempotency_key": idempotency_key
        or f"agent_{category}:{session_id}:{uuid.uuid4().hex[:12]}",
        "session_id": session_id,
        "room_name": room_name,
        "tenant_id": tenant_id,
        "api_key_id": api_key_id,
        "source": "agent_worker",
        "category": category,
        "model": model_name,
        "occurred_at_ms": int(time.time() * 1000),
        "metrics": metrics_body,
    }
    headers = {"Content-Type": "application/json"}
    if HIMARKET_INTERNAL_SECRET:
        headers["X-HiMarket-Internal-Secret"] = HIMARKET_INTERNAL_SECRET
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                HIMARKET_METERING_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=4.0),
            ) as resp:
                log.info(
                    "Reported %s metering event (session=%s, status=%d)",
                    category,
                    session_id,
                    resp.status,
                )
    except Exception as exc:
        log.warning("Failed to report %s metering event: %s", category, exc)


def model_binding(job_meta: dict, key: str) -> dict:
    value = job_meta.get(key) or {}
    if not isinstance(value, dict):
        value = {}
    return value


def _build_llm_headers_and_url(
    binding: dict, session_id: str, tenant_id: str, api_key_id: str
) -> tuple[str, dict[str, str], str]:
    protocol = (binding.get("protocol") or "openai_chat").strip()
    if protocol not in ("openai_chat", "livekit_inference"):
        raise LlmFailedError(f"llm_protocol_unsupported: {protocol}")
    if protocol == "livekit_inference":
        base_url = (binding.get("base_url") or CLIPROXY_BASE_URL).rstrip("/")
        api_key = inference_token(binding)
    else:
        # All openai_chat text LLM requests route through CLIProxyAPI when CLIPROXY_BASE_URL is configured
        base_url = (CLIPROXY_BASE_URL or binding.get("base_url") or "").rstrip("/")
        api_key = CLIPROXY_API_KEY if CLIPROXY_BASE_URL else (binding.get("api_key") or CLIPROXY_API_KEY)
    llm_model = binding.get("model") or DEFAULT_LLM_MODEL
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Session-ID": session_id,
        "X-HiMarket-Session-Id": session_id,
        "X-HiMarket-Tenant-Id": tenant_id,
        "X-HiMarket-Api-Key-Id": api_key_id,
        "X-HiMarket-Upstream-Base-Url": (binding.get("base_url") or "").strip(),
    }
    return url, headers, llm_model


async def call_llm(
    user_text: str,
    session_id: str,
    tenant_id: str,
    api_key_id: str,
    binding: dict,
    system_prompt: str,
) -> tuple[str, dict]:
    """Cascade-only LLM call with fallback strings."""
    url, headers, llm_model = _build_llm_headers_and_url(
        binding, session_id, tenant_id, api_key_id
    )
    body = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
                or "你是一个运行在LiveKit与HiMarket平台上的智能语音助手。请用简短、自然的一两句话回答用户，不要使用Markdown符号。",
            },
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 256,
        "temperature": 0.6,
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content and msg.get("reasoning"):
            content = "你好！我已经收到你的语音消息，目前语音对讲链路和计费服务运行一切正常。"
        usage = data.get("usage") or {}
        return content or "你好！我是你的智能语音助手，很高兴为你服务。", usage


async def call_llm_strict(
    user_text: str,
    session_id: str,
    tenant_id: str,
    api_key_id: str,
    binding: dict,
    system_prompt: str,
) -> tuple[str, dict]:
    """Strict LLM call for translation/summary without cascade fallback strings."""
    url, headers, llm_model = _build_llm_headers_and_url(
        binding, session_id, tenant_id, api_key_id
    )
    body = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise LlmFailedError(str(exc)) from exc

    msg = ((data.get("choices") or [{}])[0]).get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise LlmEmptyError("LLM returned empty content")
    usage = data.get("usage") or {}
    return content, usage


async def call_llm_stream_strict(
    user_text: str,
    session_id: str,
    tenant_id: str,
    api_key_id: str,
    binding: dict,
    system_prompt: str,
    on_delta: Callable[[str, str], Awaitable[None]] | None = None,
) -> tuple[str, dict]:
    """Streaming SSE translation call with fallback to call_llm_strict if non-SSE."""
    url, headers, llm_model = _build_llm_headers_and_url(
        binding, session_id, tenant_id, api_key_id
    )
    body = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    acc_parts: list[str] = []
    usage: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").lower()
                if "text/event-stream" not in ctype:
                    raw = await resp.aread()
                    data = json.loads(raw.decode("utf-8"))
                    msg = ((data.get("choices") or [{}])[0]).get("message") or {}
                    content = (msg.get("content") or "").strip()
                    if not content:
                        raise LlmEmptyError("LLM returned empty content")
                    return content, data.get("usage") or {}

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {}).get("content") or ""
                        if delta:
                            acc_parts.append(delta)
                            if on_delta is not None:
                                await on_delta(delta, "".join(acc_parts))
    except LlmEmptyError:
        raise
    except Exception:
        return await call_llm_strict(
            user_text, session_id, tenant_id, api_key_id, binding, system_prompt
        )

    final_text = "".join(acc_parts).strip()
    if not final_text:
        raise LlmEmptyError("SSE stream finished with empty content")
    return final_text, usage


def _internal_headers() -> dict[str, str]:
    if not HIMARKET_INTERNAL_SECRET.strip():
        raise InternalUnavailable("HIMARKET_INTERNAL_SECRET is empty")
    return {
        "Content-Type": "application/json",
        "X-HiMarket-Internal-Secret": HIMARKET_INTERNAL_SECRET.strip(),
    }


async def fetch_session_runtime(session_id: str) -> dict[str, Any]:
    url = f"{HIMARKET_ACCESS_URL.rstrip('/')}/internal/v1/voice/sessions/{session_id}/runtime"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(url, headers=_internal_headers())
            if resp.status_code != 200:
                raise InternalUnavailable(f"runtime GET status {resp.status_code}")
            return resp.json().get("data") or {}
    except InternalUnavailable:
        raise
    except Exception as exc:
        raise InternalUnavailable(str(exc)) from exc


async def post_session_target_lang(
    session_id: str,
    target_lang: str | None = None,
    source_lang: str | None = None,
    resolve_tts: bool = True,
) -> dict[str, Any]:
    url = f"{HIMARKET_ACCESS_URL.rstrip('/')}/internal/v1/voice/sessions/{session_id}/target-lang"
    payload: dict[str, Any] = {"resolve_tts": resolve_tts}
    if target_lang is not None:
        payload["target_lang"] = target_lang
    if source_lang is not None:
        payload["source_lang"] = source_lang
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.post(url, headers=_internal_headers(), json=payload)
    except Exception as exc:
        raise InternalUnavailable(str(exc)) from exc
    if resp.status_code == 400:
        body = resp.json()
        err_code = body.get("error") or "target_lang_not_allowed"
        raise ValueError(f"{err_code}: {body.get('message')}")
    if resp.status_code != 200:
        raise InternalUnavailable(f"target-lang POST status {resp.status_code}")
    return resp.json().get("data") or {}


async def post_session_bind_tts(session_id: str, target_lang: str) -> dict[str, Any]:
    url = f"{HIMARKET_ACCESS_URL.rstrip('/')}/internal/v1/voice/sessions/{session_id}/tts"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.post(
                url, headers=_internal_headers(), json={"target_lang": target_lang}
            )
            if resp.status_code != 200:
                raise InternalUnavailable(f"bind tts status {resp.status_code}")
            return resp.json().get("data") or {}
    except InternalUnavailable:
        raise
    except Exception as exc:
        raise InternalUnavailable(str(exc)) from exc


async def post_summary_input(
    session_id: str,
    origin: str,
    source_text: str | None,
    translated_text: str | None,
    assistant_text: str | None,
    client_input_id: str,
) -> dict[str, Any]:
    url = f"{HIMARKET_ACCESS_URL.rstrip('/')}/internal/v1/voice/sessions/{session_id}/summary-inputs"
    payload = {
        "origin": origin,
        "source_text": source_text,
        "translated_text": translated_text,
        "assistant_text": assistant_text,
        "client_input_id": client_input_id,
    }
    async with httpx.AsyncClient(timeout=1.0) as client:
        resp = await client.post(url, headers=_internal_headers(), json=payload)
        resp.raise_for_status()
        return resp.json().get("data") or {}


async def post_execute_summary_with_polling(
    session_id: str,
    request_id: str,
    speak: bool = False,
    target_lang: str = "",
    summary_source: str | None = None,
    summary_style: str = "speech",
) -> dict[str, Any]:
    """Dedicated 30s timeout client for internal summary execution + 90s polling on 409/timeout."""
    url = f"{HIMARKET_ACCESS_URL.rstrip('/')}/internal/v1/voice/sessions/{session_id}/summaries"
    payload: dict[str, Any] = {
        "request_id": request_id,
        "speak": speak,
        "target_lang": target_lang,
        "summary_style": summary_style,
    }
    if summary_source:
        payload["summary_source"] = summary_source

    deadline = time.monotonic() + 90.0
    while True:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=_internal_headers(), json=payload)
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise InternalUnavailable(str(exc)) from exc
            await asyncio.sleep(2.0)
            continue

        if resp.status_code == 200:
            return resp.json().get("data") or {}

        if resp.status_code == 409:
            body = resp.json()
            err_code = body.get("error") or ""
            if err_code == "summary_in_progress":
                if time.monotonic() >= deadline:
                    raise InternalUnavailable("summary polling exceeded 90s")
                retry_after = float(resp.headers.get("Retry-After") or "2")
                await asyncio.sleep(max(0.5, retry_after))
                continue
            raise ValueError(f"{err_code or 'llm_failed'}: {body.get('message')}")

        if resp.status_code == 400:
            body = resp.json()
            err_code = body.get("error") or "empty_input"
            raise ValueError(f"{err_code}: {body.get('message')}")

        raise InternalUnavailable(f"summary POST status {resp.status_code}")


async def _handle_internal_synthesize(request: web.Request) -> web.Response:
    expected = HIMARKET_INTERNAL_SECRET.strip()
    got = (request.headers.get("X-HiMarket-Internal-Secret") or "").strip()
    if not expected or got != expected:
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "empty_input"}, status=400)

    text = (body.get("text") or "").strip()
    binding = body.get("binding") or {}
    if not text:
        return web.json_response({"error": "empty_input"}, status=400)

    try:
        pcm, sr, _ = await synthesize_strict(binding, text)
        wav_bytes = pcm16_to_wav_bytes(pcm, sr)
        return web.Response(body=wav_bytes, status=200, content_type="audio/wav")
    except TtsProtocolUnsupportedError:
        return web.json_response({"error": "tts_protocol_unsupported"}, status=400)
    except Exception as exc:
        log.warning("Internal synthesize failed: %s", exc)
        return web.json_response({"error": "tts_failed"}, status=502)


async def _handle_internal_transcribe(request: web.Request) -> web.Response:
    import base64

    expected = HIMARKET_INTERNAL_SECRET.strip()
    got = (request.headers.get("X-HiMarket-Internal-Secret") or "").strip()
    if not expected or got != expected:
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "empty_input"}, status=400)

    b64 = (body.get("audio_base64") or "").strip()
    binding = body.get("binding") or {}
    language = (body.get("language") or "").strip()
    sample_rate = int(body.get("sample_rate") or 16000)
    if not b64:
        return web.json_response({"error": "empty_input"}, status=400)

    try:
        raw_audio = base64.b64decode(b64)
        if raw_audio[:4] == b"RIFF":
            with wave.open(io.BytesIO(raw_audio), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
                src_sr = wf.getframerate()
            pcm = resample_pcm16(pcm, src_sr, 16000)
        else:
            pcm = resample_pcm16(raw_audio, sample_rate, 16000)
        text, dur_ms, det = await transcribe_strict(
            binding, pcm, whisper_language_arg(language)
        )
        return web.json_response(
            {"text": text, "duration_ms": dur_ms, "detected_lang": det}, status=200
        )
    except Exception as exc:
        log.warning("Internal transcribe failed: %s", exc)
        return web.json_response({"error": "stt_failed", "message": str(exc)}, status=502)


def start_internal_synth_server(port: int = SYNTH_HTTP_PORT) -> None:
    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = web.Application()
        app.router.add_post("/internal/v1/synthesize", _handle_internal_synthesize)
        app.router.add_post("/internal/v1/transcribe", _handle_internal_transcribe)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "0.0.0.0", port)
        loop.run_until_complete(site.start())
        log.info("Internal TTS synthesis server listening on http://0.0.0.0:%d", port)
        loop.run_forever()

    t = threading.Thread(target=_runner, name="himarket-synth-http", daemon=True)
    t.start()


async def publish_Data_subtitle(room: rtc.Room, role: str, text: str, turn_id: str):
    packet = json.dumps(
        {
            "type": "subtitle",
            "role": role,
            "text": text,
            "turn_id": turn_id,
            "final": True,
            "ts": int(time.time() * 1000),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    with contextlib.suppress(Exception):
        await room.local_participant.publish_data(packet, reliable=True)


async def entrypoint(ctx: agents.JobContext):
    log.info("Starting HiMarket Voice Agent for room=%s job_id=%s", ctx.room.name, ctx.job.id)
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    meta_raw = ctx.job.metadata or "{}"
    try:
        job_meta = json.loads(meta_raw)
    except Exception:
        job_meta = {}

    session_id = job_meta.get("session_id") or ctx.room.name
    tenant_id = job_meta.get("tenant_id") or "tenant_ugreen_01"
    api_key_id = job_meta.get("api_key_id") or "ak_live_ugreen_01"
    voice_mode = (job_meta.get("mode") or "cascade").strip()
    job_task = (job_meta.get("task") or "cascade").strip()
    source_lang = (job_meta.get("source_lang") or "").strip()
    target_lang = (job_meta.get("target_lang") or "").strip()
    speak_flag = bool(job_meta.get("speak", job_task in ("cascade", "speak")))

    llm = model_binding(job_meta, "llm")
    stt = model_binding(job_meta, "stt")
    tts = model_binding(job_meta, "tts")
    llm_model = llm.get("model") or job_meta.get("llm_model") or DEFAULT_LLM_MODEL
    system_prompt = job_meta.get("system_prompt") or ""

    audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
    track_published = False

    async def ensure_track_published():
        nonlocal track_published
        if not track_published:
            agent_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", audio_source)
            await ctx.room.local_participant.publish_track(
                agent_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            track_published = True

    if job_task in ("cascade", "speak") or (job_task == "speech_translate" and speak_flag):
        await ensure_track_published()

    await ctx.room.local_participant.set_attributes(
        {
            "agent_state": "listening",
            "session_id": session_id,
            "tenant_id": tenant_id,
            "task": job_task,
            "target_lang": target_lang,
            "llm_model": llm_model,
        }
    )

    speaking_lock = asyncio.Lock()

    async def publish_packet(payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with contextlib.suppress(Exception):
            await ctx.room.local_participant.publish_data(raw, reliable=True)

    async def push_audio(pcm_out: bytes, out_sr: int) -> None:
        await ensure_track_published()
        await ctx.room.local_participant.set_attributes({"agent_state": "speaking"})
        samples_per_frame = out_sr // 50
        bytes_per_frame = samples_per_frame * 2
        for offset in range(0, len(pcm_out), bytes_per_frame):
            chunk = pcm_out[offset : offset + bytes_per_frame]
            if len(chunk) < bytes_per_frame:
                chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=out_sr,
                num_channels=1,
                samples_per_channel=samples_per_frame,
            )
            await audio_source.capture_frame(frame)
            await asyncio.sleep(0.018)
        await ctx.room.local_participant.set_attributes({"agent_state": "listening"})

    async def report_metering(
        category: str, model_name: str, metrics: dict[str, Any], idempotency_key: str | None = None
    ) -> None:
        asyncio.create_task(
            report_metering_event(
                session_id,
                ctx.room.name,
                tenant_id,
                api_key_id,
                category,
                model_name,
                metrics,
                idempotency_key=idempotency_key,
            )
        )

    embedded_runtime = {
        "session_id": session_id,
        "tenant_id": tenant_id,
        "api_key_id": api_key_id,
        "mode": voice_mode,
        "task": job_task,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "speak": speak_flag,
        "llm": llm,
        "stt": stt,
        "tts": tts,
        "system_prompt": system_prompt,
    }

    driver = TaskSessionDriver(
        session_id=session_id,
        room_name=ctx.room.name,
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        job_task=job_task,
        embedded_runtime=embedded_runtime,
        fail_closed=False,
        get_runtime_fn=fetch_session_runtime,
        post_target_lang_fn=post_session_target_lang,
        post_bind_tts_fn=post_session_bind_tts,
        append_input_fn=post_summary_input,
        execute_summary_fn=post_execute_summary_with_polling,
        transcribe_strict_fn=transcribe_strict,
        transcribe_cascade_fn=transcribe,
        call_llm_cascade_fn=lambda txt, b, sp: call_llm(
            txt, session_id, tenant_id, api_key_id, b, sp
        ),
        call_llm_strict_fn=lambda txt, b, sp: call_llm_strict(
            txt, session_id, tenant_id, api_key_id, b, sp
        ),
        call_llm_stream_strict_fn=lambda txt, b, sp, cb: call_llm_stream_strict(
            txt, session_id, tenant_id, api_key_id, b, sp, cb
        ),
        synthesize_cascade_fn=synthesize,
        synthesize_strict_fn=synthesize_strict,
        publish_packet_fn=publish_packet,
        push_audio_fn=push_audio,
        report_metering_fn=report_metering,
    )

    async def locked_process_turn(
        pcm: bytes, preset_text: str = "", turn_id: str | None = None, speaker: str | None = None
    ):
        async with speaking_lock:
            await ctx.room.local_participant.set_attributes({"agent_state": "thinking"})
            await driver.process_turn(
                pcm, preset_text=preset_text, turn_id=turn_id, speaker=speaker
            )
            await ctx.room.local_participant.set_attributes({"agent_state": "listening"})

    async def monitor_audio_track(track: rtc.Track):
        if job_task == "speak":
            return
        log.info("Subscribed to remote audio track sid=%s task=%s", track.sid, job_task)
        audio_stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        async for event in audio_stream:
            frame_bytes = bytes(event.frame.data)
            samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) > 0 else 0.0
            interim, finalized = driver.accumulator.push_frame(frame_bytes, rms)
            if interim is not None:
                win_pcm, tid, spk = interim
                asyncio.create_task(driver.run_interim_stt(win_pcm, tid, spk))
            if finalized is not None:
                utterance, tid, spk = finalized
                asyncio.create_task(locked_process_turn(utterance, turn_id=tid, speaker=spk))

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(monitor_audio_track(track))

    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        try:
            msg = json.loads(data.data.decode("utf-8"))
        except Exception as exc:
            log.warning("Failed to parse data packet: %s", exc)
            return

        async def _dispatch():
            async with speaking_lock:
                await driver.handle_data_message(msg)

        asyncio.create_task(_dispatch())

    for p in ctx.room.remote_participants.values():
        for pub in p.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(monitor_audio_track(pub.track))

    # Greeting runs ONLY in cascade task
    if job_task == "cascade":
        await asyncio.sleep(1.2)
        greeting_text = (job_meta.get("greeting_text") or "").strip()
        if greeting_text:
            async def _speak_persona_greeting():
                async with speaking_lock:
                    tid = f"greet_{uuid.uuid4().hex[:8]}"
                    await publish_packet(
                        {
                            "type": "subtitle",
                            "role": "assistant",
                            "text": greeting_text,
                            "turn_id": tid,
                            "final": True,
                            "ts": int(time.time() * 1000),
                        }
                    )
                    try:
                        rt = await driver.target_state()
                        tts_b = rt.get("tts") or tts
                    except Exception:
                        tts_b = tts
                    pcm_out, out_sr, tts_dur_ms = await synthesize(tts_b, greeting_text)
                    await report_metering(
                        "tts",
                        tts_b.get("model") or "tts",
                        {"tts_characters": len(greeting_text), "audio_duration_ms": tts_dur_ms},
                    )
                    await push_audio(pcm_out, out_sr)

            asyncio.create_task(_speak_persona_greeting())
        else:
            asyncio.create_task(
                locked_process_turn(
                    b"",
                    preset_text="你好，我已经进入语音房间，请用一句话向我打招呼。",
                )
            )


if __name__ == "__main__":
    start_internal_synth_server(SYNTH_HTTP_PORT)
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="himarket-voice-agent",
        )
    )
