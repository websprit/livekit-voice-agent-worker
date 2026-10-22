FROM livekit-audio-worker:dev
USER 0
ENV UV_CACHE_DIR=/tmp/uv
# faster-whisper 1.2.1 passes metadata_errors to av.open; PyAV 19 removed that argument.
RUN uv pip install --python /app/.venv/bin/python "faster-whisper==1.2.1" "piper-tts" "av>=12,<19" "websockets>=13,<16" --index-url https://pypi.tuna.tsinghua.edu.cn/simple || \
    uv pip install --python /app/.venv/bin/python "faster-whisper==1.2.1" "piper-tts" "av>=12,<19" "websockets>=13,<16"
WORKDIR /app
COPY agent_main.py vendor_dispatch.py task_route.py /app/
RUN mkdir -p /models && chown -R 10001:10001 /app /models /tmp/uv
USER 10001:10001
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/app/agent_main.py", "start"]
