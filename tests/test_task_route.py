import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_route import (
    AudioTurnAccumulator,
    InternalUnavailable,
    LlmEmptyError,
    TaskSessionDriver,
    TtsFailedError,
    order_and_truncate_summary_inputs,
    steps_for,
    translation_system_prompt,
    whisper_language_arg,
)


class TestTaskRoute(unittest.IsolatedAsyncioTestCase):
    def test_whisper_language_arg_and_prompts_and_steps(self):
        self.assertIsNone(whisper_language_arg(""))
        self.assertIsNone(whisper_language_arg(None))
        self.assertEqual(whisper_language_arg("zh"), "zh")
        self.assertEqual(whisper_language_arg("ja-JP"), "ja")
        with self.assertRaises(ValueError):
            whisper_language_arg("*")

        self.assertEqual(
            translation_system_prompt("ja"),
            "把下面的文字译成 ja，只输出译文",
        )

        self.assertEqual(steps_for("transcribe", False), frozenset({"stt", "append"}))
        self.assertEqual(
            steps_for("speech_translate", False), frozenset({"stt", "llm", "append"})
        )
        self.assertEqual(
            steps_for("speech_translate", True),
            frozenset({"stt", "llm", "tts", "append"}),
        )
        self.assertEqual(steps_for("text_translate", False), frozenset({"llm", "append"}))
        self.assertEqual(steps_for("speak", True), frozenset({"tts"}))
        self.assertEqual(steps_for("context", False), frozenset({"append"}))
        self.assertEqual(
            steps_for("cascade", True), frozenset({"stt", "llm", "tts", "append"})
        )
        with self.assertRaises(ValueError):
            steps_for("summarize", False)

    async def test_speech_translate_strict_and_cascade_isolation(self):
        packets = []
        audio_pushed = []
        appended = {}
        seq_counter = 0

        async def append_fn(
            session_id,
            origin,
            source_text,
            translated_text,
            assistant_text,
            client_input_id,
        ):
            nonlocal seq_counter
            if client_input_id in appended:
                row = appended[client_input_id]
                if source_text:
                    row["source_text"] = source_text
                if translated_text:
                    row["translated_text"] = translated_text
                return row
            seq_counter += 1
            row = {
                "seq": seq_counter,
                "origin": origin,
                "source_text": source_text,
                "translated_text": translated_text,
                "assistant_text": assistant_text,
                "client_input_id": client_input_id,
            }
            appended[client_input_id] = row
            return row

        async def stt_strict(binding, pcm, lang):
            return "今天天气很好", 1200, lang or "zh"

        async def llm_empty(text, binding, sys_prompt):
            raise LlmEmptyError("empty response")

        async def synth_strict(binding, text):
            return b"\x01\x02" * 480, 24000, 500

        async def get_rt_no_speak(sid):
            return {
                "session_id": sid,
                "task": "speech_translate",
                "source_lang": "zh",
                "target_lang": "ja",
                "speak": False,
                "stt": {"protocol": "whisper_local", "model": "faster-whisper-tiny"},
                "llm": {"protocol": "openai_chat", "model": "deepseek"},
                "tts": None,
            }

        # 1. Empty LLM emits llm_empty, no translation subtitle, no cascade fallback Chinese
        driver = TaskSessionDriver(
            session_id="sess_tr_1",
            room_name="lk_rm_1",
            tenant_id="tenant_ugreen_01",
            api_key_id="ak_1",
            job_task="speech_translate",
            get_runtime_fn=get_rt_no_speak,
            append_input_fn=append_fn,
            transcribe_strict_fn=stt_strict,
            call_llm_strict_fn=llm_empty,
            synthesize_strict_fn=synth_strict,
            publish_packet_fn=lambda p: self._record(packets, p),
            push_audio_fn=lambda pcm, sr: self._record(audio_pushed, (pcm, sr)),
        )
        await driver.process_turn(b"\x00\x01" * 8000, turn_id="t1")
        self.assertTrue(any(p.get("code") == "llm_empty" for p in packets))
        self.assertFalse(any(p.get("role") == "translation" for p in packets))
        self.assertFalse(
            any("智能语音助手" in (p.get("text") or "") for p in packets)
        )
        self.assertEqual(len(audio_pushed), 0)

        # 2. Successful translation with speak=False does not call synthesis, allocates 1 seq
        packets.clear()

        async def llm_ok(text, binding, sys_prompt):
            self.assertEqual(sys_prompt, "把下面的文字译成 ja，只输出译文")
            return "今日はいい天気ですね", {"prompt_tokens": 10, "completion_tokens": 8}

        driver.call_llm_strict_fn = llm_ok
        await driver.process_turn(b"\x00\x01" * 8000, turn_id="t1")
        tr_pkts = [p for p in packets if p.get("role") == "translation" and p.get("final")]
        self.assertEqual(len(tr_pkts), 1)
        self.assertEqual(tr_pkts[0]["text"], "今日はいい天気ですね")
        self.assertEqual(len(audio_pushed), 0)
        self.assertEqual(len(appended), 1)
        self.assertEqual(appended["stt:t1"]["source_text"], "今天天气很好")
        self.assertEqual(appended["stt:t1"]["translated_text"], "今日はいい天気ですね")

        # Add a pasted text note and verify default translated summary materials include both
        await driver.handle_data_message(
            {
                "type": "summary_input",
                "text": "会议室在三楼",
                "client_input_id": "note-1",
                "request_id": "r-note",
            }
        )
        joined, trunc, s_from, s_to = order_and_truncate_summary_inputs(
            list(appended.values()), task="speech_translate"
        )
        self.assertIn("今日はいい天気ですね", joined)
        self.assertIn("[note] 会议室在三楼", joined)
        self.assertEqual(trunc, 0)
        self.assertEqual((s_from, s_to), (1, 2))

        # 3. Piper failure in speech_translate(speak=True) emits tts_failed and does NOT push audio
        packets.clear()

        async def get_rt_speak(sid):
            return {
                "session_id": sid,
                "task": "speech_translate",
                "source_lang": "zh",
                "target_lang": "ja",
                "speak": True,
                "stt": {"protocol": "whisper_local", "model": "faster-whisper-tiny"},
                "llm": {"protocol": "openai_chat", "model": "deepseek"},
                "tts": {
                    "config_id": "cfg_ja",
                    "protocol": "piper_local",
                    "model": "ja_JP-test",
                    "voice": "/models/piper/ja_JP.onnx",
                    "extra_json": '{"tts_languages":["ja"]}',
                },
            }

        async def synth_fail(binding, text):
            raise TtsFailedError("piper failed")

        driver.get_runtime_fn = get_rt_speak
        driver.runtime_cache = None
        driver.synthesize_strict_fn = synth_fail
        await driver.process_turn(b"\x00\x01" * 8000, turn_id="t2")
        self.assertTrue(any(p.get("role") == "translation" for p in packets))
        self.assertTrue(any(p.get("code") == "tts_failed" for p in packets))
        self.assertEqual(len(audio_pushed), 0)

    async def test_runtime_get_failure_matrix(self):
        async def failing_rt(sid):
            raise InternalUnavailable("db unreachable")

        # 1. Cascade falls back to embedded bindings when fail_closed=False
        cas_steps = []
        cas_driver = TaskSessionDriver(
            session_id="sess_cas",
            room_name="rm",
            tenant_id="t1",
            api_key_id="ak1",
            job_task="cascade",
            embedded_runtime={
                "task": "cascade",
                "stt": {"model": "w"},
                "llm": {"model": "l"},
                "tts": {"model": "p"},
            },
            fail_closed=False,
            get_runtime_fn=failing_rt,
            transcribe_cascade_fn=self._step_fn(cas_steps, "stt", ("你好", 500)),
            call_llm_cascade_fn=self._step_fn(
                cas_steps,
                "llm",
                ("你好！我是你的智能语音助手，很高兴为你服务。", {}),
            ),
            synthesize_cascade_fn=self._step_fn(cas_steps, "tts", (b"\x00\x00", 24000, 200)),
        )
        await cas_driver.process_turn(b"\x01\x02" * 4000)
        self.assertEqual(cas_steps, ["stt", "llm", "tts"])

        # 2. Transcribe uses embedded STT (and never reads target_lang) when cache is empty,
        # and uses newer cached source_lang when cache was populated by start_turn
        tr_langs = []

        async def tr_stt(binding, pcm, lang):
            tr_langs.append(lang)
            return "测试文本", 400, lang or "auto"

        tr_driver = TaskSessionDriver(
            session_id="sess_tr",
            room_name="rm",
            tenant_id="t1",
            api_key_id="ak1",
            job_task="transcribe",
            embedded_runtime={
                "task": "transcribe",
                "source_lang": "zh",
                "target_lang": "ja",
                "stt": {"model": "w"},
            },
            fail_closed=False,
            get_runtime_fn=failing_rt,
            transcribe_strict_fn=tr_stt,
        )
        with self.assertRaises(KeyError):
            _ = tr_driver.embedded_stt["target_lang"]
        await tr_driver.process_turn(b"\x01\x02" * 4000)
        self.assertEqual(tr_langs, ["zh"])

        # Now populate cache via start_turn with source_lang="en", then let runtime GET fail after TTL
        tr_driver.runtime_cache = {
            "task": "transcribe",
            "source_lang": "en",
            "stt": {"model": "w-en"},
        }
        tr_driver.runtime_cache_at = 0.0
        await tr_driver.process_turn(b"\x01\x02" * 4000)
        self.assertEqual(tr_langs[-1], "en")

        # 3. speech_translate fails closed immediately without calling STT, LLM, or TTS
        st_steps = []
        st_pkts = []
        st_driver = TaskSessionDriver(
            session_id="sess_st",
            room_name="rm",
            tenant_id="t1",
            api_key_id="ak1",
            job_task="speech_translate",
            embedded_runtime={"task": "speech_translate", "target_lang": "ja"},
            fail_closed=False,
            get_runtime_fn=failing_rt,
            transcribe_strict_fn=self._step_fn(st_steps, "stt", ("a", 100, "zh")),
            call_llm_strict_fn=self._step_fn(st_steps, "llm", ("b", {})),
            synthesize_strict_fn=self._step_fn(st_steps, "tts", (b"", 24000, 100)),
            publish_packet_fn=lambda p: self._record(st_pkts, p),
        )
        await st_driver.process_turn(b"\x01\x02" * 4000)
        self.assertEqual(st_steps, [])
        self.assertTrue(any(p.get("code") == "internal_unavailable" for p in st_pkts))

    async def test_transcribe_summary_rebinds_tts_after_set_target_lang(self):
        bind_calls = []
        synth_bindings = []
        packets = []

        async def post_lang(sid, target_lang=None, source_lang=None, resolve_tts=True):
            self.assertTrue(resolve_tts)
            if target_lang == "ja":
                return {
                    "target_lang": "ja",
                    "voice_fallback": False,
                    "tts_error": None,
                    "tts": {
                        "config_id": "cfg_tts_ja",
                        "protocol": "piper_local",
                        "model": "ja_JP-voice",
                        "voice": "/models/piper/ja_JP.onnx",
                        "extra_json": '{"tts_languages":["ja"]}',
                    },
                }
            return {
                "target_lang": "en",
                "voice_fallback": True,
                "tts_error": None,
                "tts": {
                    "config_id": "cfg_tts_zh_fallback",
                    "protocol": "piper_local",
                    "model": "zh_CN-huayan-medium",
                    "voice": "/models/piper/zh_CN-huayan-medium.onnx",
                    "extra_json": "{}",
                },
            }

        async def post_bind_tts(sid, target_lang):
            bind_calls.append(target_lang)
            return {
                "target_lang": target_lang,
                "voice_fallback": True,
                "tts_error": None,
                "tts": {
                    "config_id": "cfg_tts_en_fallback",
                    "protocol": "piper_local",
                    "model": "zh_CN-huayan-medium",
                    "voice": "/models/piper/zh_CN-huayan-medium.onnx",
                    "extra_json": "{}",
                },
            }

        async def exec_sum(
            sid, request_id, speak=False, target_lang="", summary_source=None, summary_style="speech"
        ):
            eff_lang = target_lang or "en"
            return {
                "session_id": sid,
                "request_id": request_id,
                "result_seq": 1,
                "title": "总结标题",
                "summary_style": summary_style,
                "summary_text": "# 总结标题\n\n## 核心摘要\nEnglish summary text",
                "target_lang": eff_lang,
                "voice_fallback": True,
                "tts": None,
                "tts_error": None,
            }

        async def synth_strict(binding, text):
            synth_bindings.append(binding["config_id"])
            return b"\x00\x01" * 240, 24000, 200

        driver = TaskSessionDriver(
            session_id="sess_sum_rebind",
            room_name="rm",
            tenant_id="t1",
            api_key_id="ak1",
            job_task="transcribe",
            post_target_lang_fn=post_lang,
            post_bind_tts_fn=post_bind_tts,
            execute_summary_fn=exec_sum,
            synthesize_strict_fn=synth_strict,
            publish_packet_fn=lambda p: self._record(packets, p),
        )

        # First bind ja, then change to en
        await driver.handle_data_message(
            {"type": "set_target_lang", "target_lang": "ja", "request_id": "r1"}
        )
        self.assertEqual(driver.runtime_cache["tts"]["config_id"], "cfg_tts_ja")

        await driver.handle_data_message(
            {"type": "set_target_lang", "target_lang": "en", "request_id": "r2"}
        )
        self.assertEqual(driver.runtime_cache["target_lang"], "en")
        self.assertNotEqual(driver.runtime_cache["tts"]["config_id"], "cfg_tts_ja")

        # Trigger summarize with speak=True -> must call post_bind_tts and never use cfg_tts_ja
        await driver.handle_data_message(
            {"type": "summarize", "request_id": "sum1", "speak": True}
        )
        self.assertEqual(bind_calls, ["en"])
        self.assertEqual(synth_bindings, ["cfg_tts_en_fallback"])

        # Room task=speak rejects set_target_lang
        spk_pkts = []
        spk_driver = TaskSessionDriver(
            session_id="sess_spk",
            room_name="rm",
            tenant_id="t1",
            api_key_id="ak1",
            job_task="speak",
            publish_packet_fn=lambda p: self._record(spk_pkts, p),
        )
        await spk_driver.handle_data_message(
            {"type": "set_target_lang", "target_lang": "en", "request_id": "r3"}
        )
        self.assertTrue(any(p.get("code") == "task_rejected" for p in spk_pkts))

    async def test_interim_stt_and_manual_dialogue_turns(self):
        acc = AudioTurnAccumulator(sample_rate=16000)
        frame_20ms = b"\x10\x02" * 320  # 20ms at 16kHz mono PCM16

        # Auto-VAD: speech onset generates turn_id; 8s cap forces finalization and clears turn_id
        _, final_none = acc.push_frame(frame_20ms, rms=500.0, now_monotonic=1.0)
        self.assertIsNone(final_none)
        first_tid = acc.current_turn_id
        self.assertIsNotNone(first_tid)

        # Feed up to 8s -> finalizes and clears current_turn_id
        for i in range(400):
            _, fin = acc.push_frame(frame_20ms, rms=500.0, now_monotonic=1.0 + i * 0.02)
            if fin is not None:
                _, fin_tid, _ = fin
                self.assertEqual(fin_tid, first_tid)
                break
        self.assertIsNone(acc.current_turn_id)
        _, _ = acc.push_frame(frame_20ms, rms=500.0, now_monotonic=10.0)
        self.assertNotEqual(acc.current_turn_id, first_tid)

        # Manual turn mode: start_turn disables 600ms silence split and 8s auto split; stop_turn finalizes short buffer
        acc.start_manual_turn("manual-t1", speaker="phone")
        acc.push_frame(frame_20ms * 5, rms=500.0, now_monotonic=20.0)  # 100ms speech (<450ms)
        # 35 frames (700ms) of silence must NOT finalize in manual mode
        for i in range(35):
            _, fin = acc.push_frame(frame_20ms, rms=10.0, now_monotonic=20.1 + i * 0.02)
            self.assertIsNone(fin)
        # 9 seconds of speech in manual mode must NOT trigger the 8s auto-VAD split
        for i in range(450):
            _, fin = acc.push_frame(frame_20ms, rms=500.0, now_monotonic=21.0 + i * 0.02)
            self.assertIsNone(fin)

        stopped = acc.stop_manual_turn("manual-t1")
        self.assertIsNotNone(stopped)
        _, stopped_tid, stopped_spk = stopped
        self.assertEqual(stopped_tid, "manual-t1")
        self.assertEqual(stopped_spk, "phone")

        # Interim STT does not append input and does not report metering
        append_calls = []
        metering_calls = []
        interim_pkts = []

        async def stt_fn(b, pcm, lang):
            return "实时绿字", 500, "zh"

        driver = TaskSessionDriver(
            session_id="sess_int",
            room_name="rm",
            tenant_id="t1",
            api_key_id="ak1",
            job_task="transcribe",
            transcribe_strict_fn=stt_fn,
            append_input_fn=self._step_fn(append_calls, "append", {"seq": 1}),
            report_metering_fn=self._step_fn(metering_calls, "meter", None),
            publish_packet_fn=lambda p: self._record(interim_pkts, p),
        )
        await driver.run_interim_stt(frame_20ms * 25, "turn-int-1", speaker="earphone")
        self.assertEqual(append_calls, [])
        self.assertEqual(metering_calls, [])
        self.assertEqual(len(interim_pkts), 1)
        self.assertFalse(interim_pkts[0]["final"])
        self.assertIsNone(interim_pkts[0]["seq"])

    def test_summary_ordering_and_truncation(self):
        rows = [
            {"seq": 3, "origin": "stt", "source_text": "C" * 100},
            {"seq": 1, "origin": "stt", "source_text": "A" * 100},
            {"seq": 2, "origin": "text", "source_text": "B" * 100},
        ]
        joined, trunc, s_from, s_to = order_and_truncate_summary_inputs(
            rows, max_chars=250, task="transcribe"
        )
        self.assertEqual(trunc, 1)
        self.assertEqual((s_from, s_to), (2, 3))
        self.assertNotIn("[1]", joined)
        self.assertTrue(joined.index("[2]") < joined.index("[3]"))

    async def _record(self, target_list: list, item):
        target_list.append(item)

    def _step_fn(self, target_list: list, name: str, ret_val):
        async def _fn(*args, **kwargs):
            target_list.append(name)
            return ret_val

        return _fn


if __name__ == "__main__":
    unittest.main()
