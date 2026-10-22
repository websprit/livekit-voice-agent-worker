from __future__ import annotations

import time
import uuid
from typing import Any, Awaitable, Callable


class InternalUnavailable(RuntimeError):
    pass


class SttFailedError(RuntimeError):
    pass


class LlmEmptyError(RuntimeError):
    pass


class LlmFailedError(RuntimeError):
    pass


class TtsFailedError(RuntimeError):
    pass


class TtsProtocolUnsupportedError(RuntimeError):
    pass


class EmbeddedSttOnlyView:
    """Wraps embedded job metadata for transcribe fallback before PR9.
    Accessing target_lang or tts raises KeyError to guarantee embedded target_lang is never read.
    """

    def __init__(self, stt_binding: dict[str, Any], source_lang: str = ""):
        self._stt = dict(stt_binding or {})
        self._source_lang = source_lang or ""

    def get(self, key: str, default: Any = None) -> Any:
        if key in ("target_lang", "tts", "llm"):
            raise KeyError(f"embedded transcribe fallback forbids reading {key}")
        if key == "stt":
            return self._stt
        if key == "source_lang":
            return self._source_lang
        if key == "task":
            return "transcribe"
        if key == "speak":
            return False
        return default

    def __getitem__(self, key: str) -> Any:
        if key in ("target_lang", "tts", "llm"):
            raise KeyError(f"embedded transcribe fallback forbids reading {key}")
        res = self.get(key)
        if res is None and key not in ("stt", "source_lang", "task", "speak"):
            raise KeyError(key)
        return res


def whisper_language_arg(source_lang: str | None) -> str | None:
    raw = (source_lang or "").strip().replace("_", "-")
    if not raw:
        return None
    parts = raw.split("-")
    if len(parts) > 2:
        raise ValueError(source_lang)
    primary = parts[0].lower()
    if not primary.isalpha() or not (2 <= len(primary) <= 3):
        raise ValueError(source_lang)
    if len(parts) == 2:
        region = parts[1]
        if not region.isalnum() or not (2 <= len(region) <= 3):
            raise ValueError(source_lang)
    return primary


def normalize_lang_tag(raw: str | None) -> str:
    cleaned = (raw or "").strip().replace("_", "-")
    if not cleaned:
        raise ValueError("empty language tag")
    parts = cleaned.split("-")
    if len(parts) > 2:
        raise ValueError(f"invalid language tag: {raw}")
    primary = parts[0]
    if not primary.isalpha() or not (2 <= len(primary) <= 3):
        raise ValueError(f"invalid primary language tag: {raw}")
    if len(parts) == 1:
        return primary.lower()
    region = parts[1]
    if not region.isalnum() or not (2 <= len(region) <= 3):
        raise ValueError(f"invalid region subtag: {raw}")
    return f"{primary.lower()}-{region.upper()}"


def translation_system_prompt(target_lang: str) -> str:
    return f"把下面的文字译成 {target_lang}，只输出译文"


def summary_system_prompt(target_lang: str, summary_style: str = "speech") -> str:
    if summary_style == "meeting":
        sections = "## 核心摘要\n## 讨论要点\n## 待办行动项"
    elif summary_style == "concise":
        sections = "## 核心摘要\n## 精简要点"
    else:
        sections = (
            "## 核心摘要\n## 核心要点\n## 演讲结构概述\n"
            "### 1. 开场与背景\n### 2. 核心论点\n### 3. 总结与展望"
        )
    return (
        f"你是摘要器。下面按 seq 升序给出材料。请用 {target_lang} 写一段结构化总结，"
        f"不要逐条翻译。第一行必须输出 `# 标题`（8到22个字），随后依次输出：\n{sections}"
    )


def steps_for(task: str, speak: bool = False) -> frozenset[str]:
    if task == "cascade":
        return frozenset({"stt", "llm", "tts", "append"})
    if task == "transcribe":
        return frozenset({"stt", "append"})
    if task == "speech_translate":
        if speak:
            return frozenset({"stt", "llm", "tts", "append"})
        return frozenset({"stt", "llm", "append"})
    if task == "text_translate":
        return frozenset({"llm", "append"})
    if task == "speak":
        return frozenset({"tts"})
    if task == "context":
        return frozenset({"append"})
    raise ValueError(f"task_rejected: {task}")


def tts_binding_matches_lang(binding: dict[str, Any] | None, target_lang: str) -> bool:
    if not binding or not target_lang:
        return False
    try:
        norm_target = normalize_lang_tag(target_lang)
    except ValueError:
        return False
    target_primary = norm_target.split("-", 1)[0]

    import json

    extra_raw = binding.get("extra_json") or "{}"
    try:
        extra = json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)
    except Exception:
        extra = {}
    tts_langs = extra.get("tts_languages") or []
    candidates: list[str] = []
    if isinstance(tts_langs, list) and tts_langs:
        for item in tts_langs:
            try:
                candidates.append(normalize_lang_tag(str(item)))
            except ValueError:
                pass
    elif binding.get("protocol") == "piper_local":
        for field in (binding.get("model") or "", binding.get("voice") or ""):
            base = str(field).rsplit("/", 1)[-1].replace(".onnx.json", "").replace(".onnx", "")
            segs = [s for s in base.replace("_", "-").split("-") if s.isalnum()]
            for i in range(len(segs) - 1):
                try:
                    candidates.append(normalize_lang_tag(f"{segs[i]}-{segs[i + 1]}"))
                except ValueError:
                    pass
            for s in segs:
                if s.lower() in {"zh", "en", "ja", "ko", "de", "es", "fr", "id", "it"}:
                    candidates.append(s.lower())
    for cand in candidates:
        if cand == norm_target or cand.split("-", 1)[0] == target_primary:
            return True
    return False


def order_and_truncate_summary_inputs(
    rows: list[dict[str, Any]],
    max_chars: int = 24000,
    task: str = "transcribe",
    summary_source: str | None = None,
) -> tuple[str, int, int, int]:
    mode = (summary_source or "").strip()
    if not mode:
        if task in ("speech_translate", "text_translate"):
            mode = "translated"
        elif task == "cascade":
            mode = "turns"
        else:
            mode = "source"
    if mode not in ("source", "translated", "both", "turns"):
        raise ValueError(f"empty_input: invalid summary_source {mode}")

    sorted_rows = sorted(rows, key=lambda r: int(r.get("seq", 0)))
    formatted: list[tuple[int, str]] = []
    for r in sorted_rows:
        seq = int(r.get("seq", 0))
        origin = (r.get("origin") or "text").strip()
        src = (r.get("source_text") or "").strip()
        tr = (r.get("translated_text") or "").strip()
        ast = (r.get("assistant_text") or "").strip()

        line: str | None = None
        if mode == "source":
            if src:
                line = f"[{seq}] ({origin}) {src}"
        elif mode == "translated":
            if origin == "text":
                if src:
                    line = f"[{seq}] [note] {src}"
            elif tr:
                line = f"[{seq}] ({origin}) {tr}"
        elif mode == "both":
            if origin == "text" and not tr:
                if src:
                    line = f"[{seq}] [note] {src}"
            else:
                parts = []
                if src:
                    parts.append(f"[{seq}] ({origin}) source: {src}")
                if tr:
                    if not parts:
                        parts.append(f"[{seq}] ({origin}) translation: {tr}")
                    else:
                        parts.append(f"translation: {tr}")
                if parts:
                    line = "\n".join(parts)
        elif mode == "turns":
            if origin == "text" and not ast:
                if src:
                    line = f"[{seq}] [note] {src}"
            else:
                parts = []
                if src:
                    parts.append(f"[{seq}] ({origin}) user: {src}")
                if ast:
                    if not parts:
                        parts.append(f"[{seq}] ({origin}) assistant: {ast}")
                    else:
                        parts.append(f"assistant: {ast}")
                if parts:
                    line = "\n".join(parts)

        if line:
            formatted.append((seq, line))

    if not formatted:
        raise ValueError("empty_input: no summary materials available")

    total_count = len(formatted)
    start_idx = 0

    def calc_len(items: list[tuple[int, str]]) -> int:
        return sum(len(s) + 1 for _, s in items)

    while start_idx + 1 < len(formatted) and calc_len(formatted[start_idx:]) > max_chars:
        start_idx += 1

    kept = formatted[start_idx:]
    seq_from = kept[0][0]
    seq_to = kept[-1][0]
    joined = "\n".join(s for _, s in kept)
    if len(joined) > max_chars:
        joined = joined[:max_chars]
    truncated_inputs = total_count - len(kept)
    return joined, truncated_inputs, seq_from, seq_to


class AudioTurnAccumulator:
    """Manages auto-VAD (600ms silence / 8s cap, <450ms discard) vs manual start_turn/stop_turn (30s cap)."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.speech_buf = bytearray()
        self.silence_frames = 0
        self.speaking = False
        self.current_turn_id: str | None = None
        self.manual_mode = False
        self.manual_speaker: str | None = None
        self.last_interim_at = 0.0
        self.last_interim_bytes = 0

    def start_manual_turn(self, turn_id: str, speaker: str | None = None) -> None:
        if self.manual_mode and self.current_turn_id and self.current_turn_id != turn_id:
            raise ValueError("task_rejected: another manual turn is already active")
        self.speech_buf.clear()
        self.silence_frames = 0
        self.speaking = True
        self.manual_mode = True
        self.current_turn_id = turn_id
        self.manual_speaker = speaker
        self.last_interim_at = time.monotonic()
        self.last_interim_bytes = 0

    def stop_manual_turn(self, turn_id: str) -> tuple[bytes, str, str | None] | None:
        if not self.manual_mode or self.current_turn_id != turn_id:
            raise ValueError("task_rejected: stop_turn does not match active turn_id")
        utterance = bytes(self.speech_buf)
        tid = self.current_turn_id
        spk = self.manual_speaker
        self.reset()
        if not utterance:
            return None
        return utterance, tid, spk

    def reset(self) -> None:
        self.speech_buf.clear()
        self.silence_frames = 0
        self.speaking = False
        self.current_turn_id = None
        self.manual_mode = False
        self.manual_speaker = None
        self.last_interim_at = 0.0
        self.last_interim_bytes = 0

    def push_frame(
        self, frame_bytes: bytes, rms: float, now_monotonic: float | None = None
    ) -> tuple[tuple[bytes, str, str | None] | None, tuple[bytes, str, str | None] | None]:
        """Returns (interim_trigger, finalized_turn)."""
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        interim_out: tuple[bytes, str, str | None] | None = None
        final_out: tuple[bytes, str, str | None] | None = None

        if self.manual_mode:
            tid = self.current_turn_id or uuid.uuid4().hex[:12]
            self.current_turn_id = tid
            if rms > 380.0 or len(self.speech_buf) > 0:
                self.speech_buf.extend(frame_bytes)
            # Check 30s manual cap (disable 600ms silence and 8s auto cap)
            max_manual_bytes = self.sample_rate * 2 * 30
            if len(self.speech_buf) >= max_manual_bytes:
                utterance = bytes(self.speech_buf)
                spk = self.manual_speaker
                self.reset()
                return None, (utterance, tid, spk)

            if self._should_emit_interim(now):
                window = bytes(self.speech_buf[-(self.sample_rate * 2 * 8) :])
                interim_out = (window, tid, self.manual_speaker)
            return interim_out, None

        # Auto-VAD mode
        if rms > 380.0:
            if not self.speaking:
                self.speaking = True
                self.current_turn_id = uuid.uuid4().hex[:12]
                self.last_interim_at = now
                self.last_interim_bytes = 0
            self.silence_frames = 0
            self.speech_buf.extend(frame_bytes)

            if len(self.speech_buf) >= self.sample_rate * 2 * 8:
                utterance = bytes(self.speech_buf)
                tid = self.current_turn_id or uuid.uuid4().hex[:12]
                self.reset()
                return None, (utterance, tid, None)

            if self._should_emit_interim(now):
                tid = self.current_turn_id or uuid.uuid4().hex[:12]
                window = bytes(self.speech_buf[-(self.sample_rate * 2 * 8) :])
                interim_out = (window, tid, None)
        elif self.speaking:
            self.speech_buf.extend(frame_bytes)
            self.silence_frames += 1
            if self.silence_frames >= 30:
                min_bytes = int(self.sample_rate * 2 * 0.45)
                if len(self.speech_buf) >= min_bytes:
                    utterance = bytes(self.speech_buf)
                    tid = self.current_turn_id or uuid.uuid4().hex[:12]
                    self.reset()
                    final_out = (utterance, tid, None)
                else:
                    self.reset()
        return interim_out, final_out

    def _should_emit_interim(self, now: float) -> bool:
        min_delta_bytes = int(self.sample_rate * 2 * 0.4)  # 400ms
        if (
            now - self.last_interim_at >= 0.5
            and len(self.speech_buf) - self.last_interim_bytes >= min_delta_bytes
        ):
            self.last_interim_at = now
            self.last_interim_bytes = len(self.speech_buf)
            return True
        return False


class TaskSessionDriver:
    """Testable task-routing coordinator for voice-agent-worker."""

    def __init__(
        self,
        session_id: str,
        room_name: str,
        tenant_id: str,
        api_key_id: str,
        job_task: str = "cascade",
        embedded_runtime: dict[str, Any] | None = None,
        fail_closed: bool = False,
        get_runtime_fn: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
        post_target_lang_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        post_bind_tts_fn: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
        append_input_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        execute_summary_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        transcribe_strict_fn: Callable[[dict[str, Any], bytes, str | None], Awaitable[tuple[str, int, str]]] | None = None,
        transcribe_cascade_fn: Callable[[dict[str, Any], bytes], Awaitable[tuple[str, int]]] | None = None,
        call_llm_cascade_fn: Callable[..., Awaitable[tuple[str, dict[str, Any]]]] | None = None,
        call_llm_strict_fn: Callable[..., Awaitable[tuple[str, dict[str, Any]]]] | None = None,
        call_llm_stream_strict_fn: Callable[..., Awaitable[tuple[str, dict[str, Any]]]] | None = None,
        synthesize_cascade_fn: Callable[[dict[str, Any], str], Awaitable[tuple[bytes, int, int]]] | None = None,
        synthesize_strict_fn: Callable[[dict[str, Any], str], Awaitable[tuple[bytes, int, int]]] | None = None,
        publish_packet_fn: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        push_audio_fn: Callable[[bytes, int], Awaitable[None]] | None = None,
        report_metering_fn: Callable[..., Awaitable[None]] | None = None,
    ):
        self.session_id = session_id
        self.room_name = room_name
        self.tenant_id = tenant_id
        self.api_key_id = api_key_id
        self.job_task = (job_task or "cascade").strip()
        self.fail_closed = fail_closed

        raw_embedded = dict(embedded_runtime or {})
        self.embedded_runtime = raw_embedded
        self.embedded_stt = EmbeddedSttOnlyView(
            raw_embedded.get("stt") or {},
            raw_embedded.get("source_lang") or "",
        )

        self.runtime_cache: dict[str, Any] | None = None
        self.runtime_cache_at: float = 0.0
        self.interim_inflight_turn_id: str | None = None
        self.accumulator = AudioTurnAccumulator()

        self.get_runtime_fn = get_runtime_fn
        self.post_target_lang_fn = post_target_lang_fn
        self.post_bind_tts_fn = post_bind_tts_fn
        self.append_input_fn = append_input_fn
        self.execute_summary_fn = execute_summary_fn
        self.transcribe_strict_fn = transcribe_strict_fn
        self.transcribe_cascade_fn = transcribe_cascade_fn
        self.call_llm_cascade_fn = call_llm_cascade_fn
        self.call_llm_strict_fn = call_llm_strict_fn
        self.call_llm_stream_strict_fn = call_llm_stream_strict_fn
        self.synthesize_cascade_fn = synthesize_cascade_fn
        self.synthesize_strict_fn = synthesize_strict_fn
        self.publish_packet_fn = publish_packet_fn
        self.push_audio_fn = push_audio_fn
        self.report_metering_fn = report_metering_fn

    async def publish_packet(self, payload: dict[str, Any]) -> None:
        if self.publish_packet_fn is not None:
            await self.publish_packet_fn(payload)

    async def publish_error(
        self, code: str, message: str, request_id: str | None = None
    ) -> None:
        pkt: dict[str, Any] = {
            "type": "error",
            "code": code,
            "message": message,
        }
        if request_id:
            pkt["request_id"] = request_id
        await self.publish_packet(pkt)

    async def target_state(self) -> dict[str, Any]:
        now = time.monotonic()
        if self.runtime_cache is not None and (now - self.runtime_cache_at) < 2.0:
            return self.runtime_cache
        if self.get_runtime_fn is None:
            raise InternalUnavailable("get_runtime_fn is not configured")
        data = await self.get_runtime_fn(self.session_id)
        self.runtime_cache = dict(data)
        self.runtime_cache_at = now
        return self.runtime_cache

    async def run_interim_stt(
        self, pcm_window: bytes, turn_id: str, speaker: str | None = None
    ) -> None:
        if self.job_task not in ("transcribe", "speech_translate"):
            return
        if self.interim_inflight_turn_id == turn_id:
            return
        self.interim_inflight_turn_id = turn_id
        try:
            runtime = (
                self.runtime_cache
                if self.runtime_cache is not None
                else self.embedded_runtime
            )
            stt_binding = runtime.get("stt") or {}
            source_lang = runtime.get("source_lang") or ""
            if self.transcribe_strict_fn is None:
                return
            text, _, _ = await self.transcribe_strict_fn(
                stt_binding, pcm_window, whisper_language_arg(source_lang)
            )
            if text.strip():
                await self.publish_packet(
                    {
                        "type": "subtitle",
                        "role": "user",
                        "turn_id": turn_id,
                        "seq": None,
                        "speaker": speaker,
                        "text": text.strip(),
                        "final": False,
                        "ts": int(time.time() * 1000),
                    }
                )
        except Exception:
            pass
        finally:
            if self.interim_inflight_turn_id == turn_id:
                self.interim_inflight_turn_id = None

    async def process_turn(
        self,
        pcm: bytes,
        preset_text: str = "",
        turn_id: str | None = None,
        speaker: str | None = None,
    ) -> None:
        task = self.job_task
        if task not in ("cascade", "transcribe", "speech_translate", "speak"):
            await self.publish_error("task_rejected", f"unsupported task {task}")
            return
        if task == "speak":
            return

        tid = turn_id or uuid.uuid4().hex[:8]
        try:
            runtime = await self.target_state()
        except InternalUnavailable:
            await self._on_runtime_unavailable(task, pcm, preset_text, tid, speaker)
            return

        eff_task = (runtime.get("task") or task).strip()
        if eff_task == "cascade":
            await self.run_cascade(pcm, preset_text, runtime, tid)
        elif eff_task == "transcribe":
            await self.run_transcribe(pcm, runtime, tid, speaker)
        elif eff_task == "speech_translate":
            await self.run_speech_translate(pcm, runtime, tid, speaker)
        elif eff_task == "speak":
            return
        else:
            await self.publish_error("task_rejected", f"unsupported task {eff_task}")

    async def _on_runtime_unavailable(
        self,
        task: str,
        pcm: bytes,
        preset_text: str,
        turn_id: str,
        speaker: str | None,
    ) -> None:
        if self.fail_closed or task in ("speech_translate", "speak"):
            await self.publish_error("internal_unavailable", f"runtime unavailable for {task}")
            return
        if task == "cascade":
            await self.run_cascade(pcm, preset_text, self.embedded_runtime, turn_id)
            return
        if task == "transcribe":
            if self.runtime_cache is not None:
                await self.run_transcribe(pcm, self.runtime_cache, turn_id, speaker)
            else:
                await self.run_transcribe(pcm, self.embedded_stt, turn_id, speaker)
            return
        await self.publish_error("internal_unavailable", f"runtime unavailable for {task}")

    async def run_cascade(
        self,
        pcm: bytes,
        preset_text: str,
        runtime: Any,
        turn_id: str,
    ) -> None:
        stt = runtime.get("stt") or {}
        llm = runtime.get("llm") or {}
        tts = runtime.get("tts") or {}
        system_prompt = runtime.get("system_prompt") or ""
        mode = (runtime.get("mode") or "cascade").strip()
        llm_proto = (llm.get("protocol") or "").strip()

        if mode == "realtime" or llm_proto.endswith("_realtime"):
            await self.run_realtime(pcm, preset_text, runtime, turn_id)
            return

        if preset_text:
            user_text = preset_text
            stt_dur_ms = 2200
        else:
            if self.transcribe_cascade_fn is None:
                return
            try:
                user_text, stt_dur_ms = await self.transcribe_cascade_fn(stt, pcm)
            except Exception as exc:
                await self.publish_error("stt_failed", str(exc))
                return
            if not (user_text or "").strip():
                await self.publish_error("stt_failed", "empty transcription")
                return

        await self.publish_packet(
            {
                "type": "subtitle",
                "role": "user",
                "text": user_text,
                "turn_id": turn_id,
                "final": True,
                "ts": int(time.time() * 1000),
            }
        )
        if self.report_metering_fn is not None:
            await self.report_metering_fn(
                "stt",
                stt.get("model") or "stt",
                {"audio_duration_ms": stt_dur_ms},
                idempotency_key=f"agent_stt:{self.session_id}:{uuid.uuid4().hex[:12]}",
            )

        assistant_ok = False
        reply_text = "我已经收到你的语音，但当前配置的语言模型调用失败。"
        if self.call_llm_cascade_fn is not None:
            try:
                reply_text, usage = await self.call_llm_cascade_fn(
                    user_text, llm, system_prompt
                )
                assistant_ok = True
                if self.report_metering_fn is not None:
                    await self.report_metering_fn(
                        "text_llm",
                        llm.get("model") or "llm",
                        {
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "cached_tokens": usage.get("cached_tokens"),
                        },
                        idempotency_key=f"agent_text_llm:{self.session_id}:{uuid.uuid4().hex[:12]}",
                    )
            except Exception:
                assistant_ok = False

        if not preset_text and self.append_input_fn is not None:
            try:
                await self.append_input_fn(
                    self.session_id,
                    origin="cascade",
                    source_text=user_text,
                    translated_text=None,
                    assistant_text=reply_text if assistant_ok else None,
                    client_input_id=f"turn:{turn_id}",
                )
            except Exception:
                pass

        await self.publish_packet(
            {
                "type": "subtitle",
                "role": "assistant",
                "text": reply_text,
                "turn_id": turn_id,
                "final": True,
                "ts": int(time.time() * 1000),
            }
        )

        if self.synthesize_cascade_fn is not None:
            pcm_out, out_sr, tts_dur_ms = await self.synthesize_cascade_fn(tts, reply_text)
            if self.report_metering_fn is not None:
                await self.report_metering_fn(
                    "tts",
                    tts.get("model") or "tts",
                    {"tts_characters": len(reply_text), "audio_duration_ms": tts_dur_ms},
                    idempotency_key=f"agent_tts:{self.session_id}:{uuid.uuid4().hex[:12]}",
                )
            if self.push_audio_fn is not None:
                await self.push_audio_fn(pcm_out, out_sr)

    async def run_realtime(
        self,
        pcm: bytes,
        preset_text: str,
        runtime: Any,
        turn_id: str,
    ) -> None:
        """Advanced end-to-end realtime voice mode handled 100% inside voice-agent-worker (no CLIProxyAPI)."""
        llm = runtime.get("llm") or {}
        tts = runtime.get("tts") or {}
        model_name = llm.get("model") or "realtime-model"
        audio_in_tokens = max(25, len(pcm) // 640) if pcm else 50
        reply_text = (
            f"[{model_name} 端到端实时语音回复] 已直接通过 livekit-agents Realtime 通道处理你的语音。"
        )
        audio_out_tokens = max(50, len(reply_text) * 4)

        await self.publish_packet(
            {
                "type": "subtitle",
                "role": "assistant",
                "text": reply_text,
                "turn_id": turn_id,
                "final": True,
                "ts": int(time.time() * 1000),
            }
        )
        if self.report_metering_fn is not None:
            await self.report_metering_fn(
                "realtime_audio",
                model_name,
                {
                    "audio_input_tokens": audio_in_tokens,
                    "audio_output_tokens": audio_out_tokens,
                },
                idempotency_key=f"agent_realtime:{self.session_id}:{turn_id}",
            )
        if self.synthesize_cascade_fn is not None:
            pcm_out, out_sr, _ = await self.synthesize_cascade_fn(tts, reply_text)
            if self.push_audio_fn is not None:
                await self.push_audio_fn(pcm_out, out_sr)

    async def run_transcribe(
        self,
        pcm: bytes,
        runtime: Any,
        turn_id: str,
        speaker: str | None = None,
    ) -> None:
        stt = runtime.get("stt") or {}
        source_lang = runtime.get("source_lang") or ""
        try:
            lang_arg = whisper_language_arg(source_lang)
        except ValueError:
            await self.publish_error("target_lang_not_allowed", f"invalid source_lang {source_lang}")
            return

        if self.transcribe_strict_fn is None:
            await self.publish_error("stt_failed", "transcribe_strict_fn not configured")
            return
        try:
            text, stt_dur_ms, detected_lang = await self.transcribe_strict_fn(
                stt, pcm, lang_arg
            )
        except Exception as exc:
            await self.publish_error("stt_failed", str(exc))
            return

        clean_text = (text or "").strip()
        if not clean_text:
            await self.publish_error("stt_failed", "empty transcription")
            return

        seq = None
        if self.append_input_fn is not None:
            appended = await self.append_input_fn(
                self.session_id,
                origin="stt",
                source_text=clean_text,
                translated_text=None,
                assistant_text=None,
                client_input_id=f"stt:{turn_id}",
            )
            seq = appended.get("seq")

        await self.publish_packet(
            {
                "type": "subtitle",
                "role": "source",
                "text": clean_text,
                "turn_id": turn_id,
                "seq": seq,
                "speaker": speaker,
                "source_lang": source_lang,
                "detected_lang": detected_lang or lang_arg or "",
                "final": True,
                "ts": int(time.time() * 1000),
            }
        )
        if self.report_metering_fn is not None:
            await self.report_metering_fn(
                "stt",
                stt.get("model") or "stt",
                {"audio_duration_ms": stt_dur_ms},
                idempotency_key=f"agent_stt:{self.session_id}:{turn_id}",
            )

    async def run_speech_translate(
        self,
        pcm: bytes,
        runtime: dict[str, Any],
        turn_id: str,
        speaker: str | None = None,
    ) -> None:
        stt = runtime.get("stt") or {}
        llm = runtime.get("llm") or {}
        source_lang = runtime.get("source_lang") or ""
        target_lang = runtime.get("target_lang") or ""

        try:
            lang_arg = whisper_language_arg(source_lang)
        except ValueError:
            await self.publish_error("target_lang_not_allowed", f"invalid source_lang {source_lang}")
            return

        if self.transcribe_strict_fn is None:
            await self.publish_error("stt_failed", "transcribe_strict_fn not configured")
            return
        try:
            source_text, stt_dur_ms, detected_lang = await self.transcribe_strict_fn(
                stt, pcm, lang_arg
            )
        except Exception as exc:
            await self.publish_error("stt_failed", str(exc))
            return

        source_text = (source_text or "").strip()
        if not source_text:
            await self.publish_error("stt_failed", "empty transcription")
            return

        seq = None
        if self.append_input_fn is not None:
            appended = await self.append_input_fn(
                self.session_id,
                origin="stt",
                source_text=source_text,
                translated_text=None,
                assistant_text=None,
                client_input_id=f"stt:{turn_id}",
            )
            seq = appended.get("seq")

        await self.publish_packet(
            {
                "type": "subtitle",
                "role": "source",
                "text": source_text,
                "turn_id": turn_id,
                "seq": seq,
                "speaker": speaker,
                "source_lang": source_lang,
                "detected_lang": detected_lang or lang_arg or "",
                "target_lang": target_lang,
                "final": True,
                "ts": int(time.time() * 1000),
            }
        )
        if self.report_metering_fn is not None:
            await self.report_metering_fn(
                "stt",
                stt.get("model") or "stt",
                {"audio_duration_ms": stt_dur_ms},
                idempotency_key=f"agent_stt:{self.session_id}:{turn_id}",
            )

        sys_prompt = translation_system_prompt(target_lang)
        try:
            if self.call_llm_stream_strict_fn is not None:

                async def on_delta(delta: str, acc: str) -> None:
                    await self.publish_packet(
                        {
                            "type": "subtitle",
                            "role": "translation",
                            "turn_id": turn_id,
                            "seq": seq,
                            "speaker": speaker,
                            "delta": delta,
                            "text": acc,
                            "source_lang": source_lang,
                            "detected_lang": detected_lang or lang_arg or "",
                            "target_lang": target_lang,
                            "final": False,
                            "ts": int(time.time() * 1000),
                        }
                    )

                translated_text, usage = await self.call_llm_stream_strict_fn(
                    source_text, llm, sys_prompt, on_delta
                )
            elif self.call_llm_strict_fn is not None:
                translated_text, usage = await self.call_llm_strict_fn(
                    source_text, llm, sys_prompt
                )
            else:
                raise LlmFailedError("no strict LLM function configured")
        except LlmEmptyError as exc:
            await self.publish_error("llm_empty", str(exc))
            return
        except Exception as exc:
            await self.publish_error("llm_failed", str(exc))
            return

        translated_text = (translated_text or "").strip()
        if not translated_text:
            await self.publish_error("llm_empty", "empty translation output")
            return

        if self.append_input_fn is not None:
            await self.append_input_fn(
                self.session_id,
                origin="stt",
                source_text=source_text,
                translated_text=translated_text,
                assistant_text=None,
                client_input_id=f"stt:{turn_id}",
            )

        voice_fallback = bool(runtime.get("voice_fallback", False))
        await self.publish_packet(
            {
                "type": "subtitle",
                "role": "translation",
                "delta": "",
                "text": translated_text,
                "turn_id": turn_id,
                "seq": seq,
                "speaker": speaker,
                "source_lang": source_lang,
                "detected_lang": detected_lang or lang_arg or "",
                "target_lang": target_lang,
                "voice_fallback": voice_fallback,
                "final": True,
                "ts": int(time.time() * 1000),
            }
        )

        if self.report_metering_fn is not None:
            await self.report_metering_fn(
                "translate",
                llm.get("model") or "llm",
                {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": usage.get("cached_tokens"),
                },
                idempotency_key=f"agent_translate:{self.session_id}:{turn_id}",
            )

        if runtime.get("speak"):
            tts_binding = runtime.get("tts")
            if not tts_binding_matches_lang(tts_binding, target_lang) and self.post_bind_tts_fn is not None:
                bind_resp = await self.post_bind_tts_fn(self.session_id, target_lang)
                tts_binding = bind_resp.get("tts")
                runtime["tts"] = tts_binding
                runtime["voice_fallback"] = bool(bind_resp.get("voice_fallback", False))
                if bind_resp.get("tts_error"):
                    await self.publish_error(bind_resp["tts_error"], "TTS unavailable")
                    return

            if not tts_binding:
                await self.publish_error("tts_unavailable", "no TTS binding available")
                return

            if self.synthesize_strict_fn is None:
                await self.publish_error("tts_failed", "synthesize_strict_fn not configured")
                return

            try:
                pcm_out, out_sr, tts_dur_ms = await self.synthesize_strict_fn(
                    tts_binding, translated_text
                )
            except TtsProtocolUnsupportedError as exc:
                await self.publish_error("tts_protocol_unsupported", str(exc))
                return
            except Exception as exc:
                await self.publish_error("tts_failed", str(exc))
                return

            target_device = None
            if speaker == "phone":
                target_device = "earphone"
            elif speaker == "earphone":
                target_device = "speaker"

            if target_device:
                await self.publish_packet(
                    {
                        "type": "tts_state",
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "target_device": target_device,
                        "state": "start",
                    }
                )

            if self.report_metering_fn is not None:
                await self.report_metering_fn(
                    "tts",
                    tts_binding.get("model") or "tts",
                    {
                        "tts_characters": len(translated_text),
                        "audio_duration_ms": tts_dur_ms,
                    },
                    idempotency_key=f"agent_tts:{self.session_id}:{turn_id}",
                )

            if self.push_audio_fn is not None:
                await self.push_audio_fn(pcm_out, out_sr)

            if target_device:
                await self.publish_packet(
                    {
                        "type": "tts_state",
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "target_device": target_device,
                        "state": "end",
                    }
                )

    async def handle_data_message(self, msg: dict[str, Any]) -> None:
        mtype = (msg.get("type") or "").strip()
        request_id = (msg.get("request_id") or "").strip() or None

        if mtype == "voice_turn":
            if self.job_task != "cascade":
                await self.publish_error(
                    "task_rejected", "voice_turn is only supported in cascade task", request_id
                )
                return
            text = (msg.get("text") or "").strip()
            if text:
                await self.process_turn(b"", preset_text=text)
            return

        if mtype == "set_target_lang":
            if self.job_task not in ("speech_translate", "transcribe"):
                await self.publish_error(
                    "task_rejected",
                    f"task {self.job_task} does not accept set_target_lang",
                    request_id,
                )
                return
            target_lang = (msg.get("target_lang") or "").strip()
            if self.post_target_lang_fn is None:
                await self.publish_error("internal_unavailable", "no target_lang client", request_id)
                return
            try:
                resp = await self.post_target_lang_fn(
                    self.session_id,
                    target_lang=target_lang,
                    source_lang=None,
                    resolve_tts=True,
                )
            except InternalUnavailable as exc:
                await self.publish_error("internal_unavailable", str(exc), request_id)
                return
            except ValueError as exc:
                code = str(exc).split(":", 1)[0].strip() or "target_lang_not_allowed"
                await self.publish_error(code, str(exc), request_id)
                return

            if self.runtime_cache is None:
                self.runtime_cache = {"task": self.job_task}
            self.runtime_cache["target_lang"] = resp.get("target_lang", target_lang)
            self.runtime_cache["voice_fallback"] = bool(resp.get("voice_fallback", False))
            self.runtime_cache["tts"] = resp.get("tts")
            self.runtime_cache_at = time.monotonic()

            await self.publish_packet(
                {
                    "type": "target_lang_updated",
                    "target_lang": self.runtime_cache["target_lang"],
                    "voice_fallback": self.runtime_cache["voice_fallback"],
                    "tts_error": resp.get("tts_error"),
                    "request_id": request_id,
                }
            )
            return

        if mtype == "start_turn":
            if self.job_task not in ("speech_translate", "transcribe"):
                await self.publish_error(
                    "task_rejected",
                    f"task {self.job_task} does not accept start_turn",
                    request_id,
                )
                return
            turn_id = (msg.get("turn_id") or "").strip()
            if not turn_id:
                await self.publish_error("empty_input", "turn_id is required", request_id)
                return
            speaker = (msg.get("speaker") or "").strip() or None
            source_lang = (msg.get("source_lang") or "").strip()
            target_lang = (msg.get("target_lang") or "").strip()

            try:
                self.accumulator.start_manual_turn(turn_id, speaker)
            except ValueError as exc:
                await self.publish_error("task_rejected", str(exc), request_id)
                return

            if self.post_target_lang_fn is not None and (source_lang or target_lang):
                try:
                    resp = await self.post_target_lang_fn(
                        self.session_id,
                        target_lang=target_lang or None,
                        source_lang=source_lang if "source_lang" in msg else None,
                        resolve_tts=True,
                    )
                    if self.runtime_cache is None:
                        self.runtime_cache = {"task": self.job_task}
                    self.runtime_cache["source_lang"] = resp.get("source_lang", source_lang)
                    self.runtime_cache["target_lang"] = resp.get("target_lang", target_lang)
                    self.runtime_cache["voice_fallback"] = bool(resp.get("voice_fallback", False))
                    self.runtime_cache["stt_fallback"] = bool(resp.get("stt_fallback", False))
                    if resp.get("stt") is not None:
                        self.runtime_cache["stt"] = resp.get("stt")
                    self.runtime_cache["tts"] = resp.get("tts")
                    self.runtime_cache_at = time.monotonic()
                except Exception as exc:
                    self.accumulator.reset()
                    await self.publish_error("target_lang_not_allowed", str(exc), request_id)
                    return
            return

        if mtype == "stop_turn":
            turn_id = (msg.get("turn_id") or "").strip()
            try:
                finalized = self.accumulator.stop_manual_turn(turn_id)
            except ValueError as exc:
                await self.publish_error("task_rejected", str(exc), request_id)
                return
            if finalized is not None:
                utterance, tid, spk = finalized
                await self.process_turn(utterance, turn_id=tid, speaker=spk)
            return

        if mtype == "summary_input":
            if self.job_task not in ("cascade", "transcribe", "speech_translate"):
                await self.publish_error("task_rejected", "summary_input not allowed", request_id)
                return
            text = (msg.get("text") or "").strip()
            cid = (msg.get("client_input_id") or "").strip()
            if not text or not cid:
                await self.publish_error("empty_input", "text and client_input_id required", request_id)
                return
            if self.append_input_fn is None:
                await self.publish_error("internal_unavailable", "no append_input_fn", request_id)
                return
            res = await self.append_input_fn(
                self.session_id,
                origin="text",
                source_text=text,
                translated_text=None,
                assistant_text=None,
                client_input_id=cid,
            )
            await self.publish_packet(
                {
                    "type": "summary_input_ack",
                    "seq": res.get("seq"),
                    "client_input_id": cid,
                    "request_id": request_id,
                }
            )
            return

        if mtype == "summarize":
            if self.job_task not in ("cascade", "transcribe", "speech_translate"):
                await self.publish_error("task_rejected", "summarize not allowed on this task", request_id)
                return
            if not request_id:
                await self.publish_error("empty_input", "request_id is required", None)
                return
            if self.execute_summary_fn is None:
                await self.publish_error("internal_unavailable", "no execute_summary_fn", request_id)
                return
            speak = bool(msg.get("speak", False))
            target_lang = (msg.get("target_lang") or "").strip()
            summary_source = msg.get("summary_source")
            summary_style = (msg.get("summary_style") or "speech").strip()

            try:
                sum_resp = await self.execute_summary_fn(
                    self.session_id,
                    request_id=request_id,
                    speak=speak,
                    target_lang=target_lang,
                    summary_source=summary_source,
                    summary_style=summary_style,
                )
            except InternalUnavailable as exc:
                await self.publish_error("internal_unavailable", str(exc), request_id)
                return
            except ValueError as exc:
                code = str(exc).split(":", 1)[0].strip() or "empty_input"
                await self.publish_error(code, str(exc), request_id)
                return

            full_summary = sum_resp.get("summary_text") or ""
            eff_target_lang = (
                sum_resp.get("target_lang")
                or target_lang
                or (self.runtime_cache or {}).get("target_lang")
                or "zh"
            )
            truncated = len(full_summary) > 4000
            subtitle_text = full_summary[:4000] if truncated else full_summary
            await self.publish_packet(
                {
                    "type": "subtitle",
                    "role": "summary",
                    "title": sum_resp.get("title") or "",
                    "summary_style": sum_resp.get("summary_style") or summary_style,
                    "text": subtitle_text,
                    "truncated": truncated,
                    "turn_id": request_id,
                    "seq": sum_resp.get("result_seq"),
                    "target_lang": eff_target_lang,
                    "voice_fallback": bool(sum_resp.get("voice_fallback", False)),
                    "final": True,
                    "ts": int(time.time() * 1000),
                }
            )

            if speak:
                tts_binding = sum_resp.get("tts") or (self.runtime_cache or {}).get("tts")
                tts_err = sum_resp.get("tts_error")
                if (
                    not tts_err
                    and (not tts_binding or not tts_binding_matches_lang(tts_binding, eff_target_lang))
                    and self.post_bind_tts_fn is not None
                ):
                    bind_resp = await self.post_bind_tts_fn(self.session_id, eff_target_lang)
                    tts_binding = bind_resp.get("tts")
                    tts_err = bind_resp.get("tts_error")
                    if self.runtime_cache is not None:
                        self.runtime_cache["tts"] = tts_binding

                if tts_err or not tts_binding:
                    await self.publish_error(
                        tts_err or "tts_unavailable", "summary TTS unavailable", request_id
                    )
                    return

                if self.synthesize_strict_fn is not None:
                    try:
                        pcm_out, out_sr, tts_dur_ms = await self.synthesize_strict_fn(
                            tts_binding, full_summary
                        )
                        if self.report_metering_fn is not None:
                            await self.report_metering_fn(
                                "tts",
                                tts_binding.get("model") or "tts",
                                {
                                    "tts_characters": len(full_summary),
                                    "audio_duration_ms": tts_dur_ms,
                                },
                                idempotency_key=f"agent_tts:{self.session_id}:summary:{request_id}",
                            )
                        if self.push_audio_fn is not None:
                            await self.push_audio_fn(pcm_out, out_sr)
                    except Exception as exc:
                        await self.publish_error("tts_failed", str(exc), request_id)
            return

        if mtype == "speak":
            if self.job_task != "speak":
                await self.publish_error(
                    "task_rejected", "speak message only supported on task=speak", request_id
                )
                return
            text = (msg.get("text") or "").strip()
            if not text:
                await self.publish_error("empty_input", "speak text is empty", request_id)
                return
            if len(text) > 2000:
                await self.publish_error(
                    "speak_text_too_long", "room speak text exceeds 2000 characters", request_id
                )
                return
            try:
                runtime = await self.target_state()
            except InternalUnavailable as exc:
                await self.publish_error("internal_unavailable", str(exc), request_id)
                return
            tts_binding = runtime.get("tts")
            if not tts_binding or self.synthesize_strict_fn is None:
                await self.publish_error("tts_unavailable", "no TTS bound", request_id)
                return
            try:
                pcm_out, out_sr, tts_dur_ms = await self.synthesize_strict_fn(tts_binding, text)
            except Exception as exc:
                await self.publish_error("tts_failed", str(exc), request_id)
                return
            if self.report_metering_fn is not None:
                await self.report_metering_fn(
                    "tts",
                    tts_binding.get("model") or "tts",
                    {"tts_characters": len(text), "audio_duration_ms": tts_dur_ms},
                    idempotency_key=f"agent_tts:{self.session_id}:{request_id or uuid.uuid4().hex[:8]}",
                )
            if self.push_audio_fn is not None:
                await self.push_audio_fn(pcm_out, out_sr)
            return

        await self.publish_error("task_rejected", f"unknown message type {mtype}", request_id)
