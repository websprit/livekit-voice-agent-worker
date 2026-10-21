# livekit-voice-agent-worker

LiveKit Agents worker for the HiMarket voice platform. It registers as `himarket-voice-agent`, subscribes to the caller microphone, and runs the configured STT, LLM, and TTS providers.

`models/` is used at runtime for local Faster-Whisper and Piper files. Those large files are not part of this repository.

Vendor protocol helpers live in the `livekit-agents` repository on branch `feat/himarket-vendor-plugins`.
