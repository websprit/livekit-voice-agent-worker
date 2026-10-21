"""Select a vendor request from a HiMarket model binding. Transport stays outside."""

from livekit.plugins.google import audio_builders as google_audio
from livekit.plugins.dashscope import audio as dashscope_audio
from livekit.plugins.volcengine import audio as volcengine_audio

OPENAI_COMPATIBLE = {"openai_chat", "openai_stt", "openai_tts", "openai_realtime"}
DIRECT = OPENAI_COMPATIBLE | {"azure_tts", "google_stt", "google_tts", "google_realtime", "dashscope_stt", "dashscope_tts", "dashscope_realtime", "volcengine_stt", "volcengine_tts", "volcengine_realtime", "whisper_local", "piper_local", "livekit_inference", "livekit_realtime"}


class UnsupportedProtocol(ValueError):
    pass


def require_protocol(binding: dict) -> str:
    protocol = binding.get("protocol") or ""
    if protocol not in DIRECT:
        raise UnsupportedProtocol(protocol or "missing protocol")
    return protocol


def transcription_request(binding: dict, pcm: bytes) -> dict:
    protocol = require_protocol(binding)
    if protocol == "google_stt":
        location = binding.get("voice") or "global"
        return {
            "url": google_audio.chirp3_recognize_url(binding.get("api_secret") or "", location),
            "json": google_audio.chirp3_request(pcm),
            "headers": {"x-goog-api-key": binding.get("api_key") or ""},
        }
    if protocol == "dashscope_stt":
        task = dashscope_audio.asr_run_task(binding.get("model") or "")
        return {"messages": [task, dashscope_audio.asr_audio_frame(task["header"]["task_id"], pcm)]}
    if protocol == "volcengine_stt":
        return {
            "frames": [
                volcengine_audio.sauc_start(binding.get("api_secret") or "", binding.get("api_key") or "", binding.get("model") or ""),
                volcengine_audio.sauc_audio(pcm),
            ]
        }
    if protocol in {"openai_stt", "livekit_inference"}:
        return {"url": binding.get("base_url", "").rstrip("/") + "/audio/transcriptions", "model": binding.get("model")}
    raise UnsupportedProtocol(protocol)


def synthesis_request(binding: dict, text: str) -> dict:
    protocol = require_protocol(binding)
    if protocol == "google_tts":
        model = binding.get("model") or ""
        if not google_audio.is_gemini_tts(model):
            raise UnsupportedProtocol("google cloud tts voice path is unchanged")
        return {
            "url": google_audio.gemini_tts_url(model, binding.get("base_url") or "https://generativelanguage.googleapis.com/v1beta"),
            "json": google_audio.gemini_tts_request(text, voice=binding.get("voice") or ""),
        }
    if protocol == "dashscope_tts":
        return {"json": dashscope_audio.tts_request(binding.get("model") or "", text, voice=binding.get("voice") or "")}
    if protocol == "volcengine_tts":
        return {"json": volcengine_audio.tts_request(binding.get("api_secret") or "", binding.get("api_key") or "", binding.get("model") or "", text, voice=binding.get("voice") or "")}
    if protocol == "azure_tts":
        return {"voice": binding.get("voice") or "", "region_url": binding.get("base_url") or ""}
    if protocol in {"openai_tts", "livekit_inference"}:
        return {"url": binding.get("base_url", "").rstrip("/") + "/audio/speech", "model": binding.get("model"), "voice": binding.get("voice")}
    raise UnsupportedProtocol(protocol)


def realtime_session(binding: dict) -> dict:
    protocol = require_protocol(binding)
    model = binding.get("model") or ""
    if protocol == "dashscope_realtime":
        return dashscope_audio.realtime_session(model)
    if protocol == "volcengine_realtime":
        return volcengine_audio.realtime_start(model)
    if protocol in {"openai_realtime", "google_realtime", "livekit_realtime"}:
        return {"model": model, "url": binding.get("base_url") or ""}
    raise UnsupportedProtocol(protocol)
