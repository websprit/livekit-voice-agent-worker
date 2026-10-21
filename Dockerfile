FROM livekit-audio-worker:dev
USER 0
ENV UV_CACHE_DIR=/tmp/uv
RUN uv pip install --python /app/.venv/bin/python faster-whisper piper-tts --index-url https://pypi.tuna.tsinghua.edu.cn/simple || \
    uv pip install --python /app/.venv/bin/python faster-whisper piper-tts
WORKDIR /app
COPY agent_main.py vendor_dispatch.py /app/
RUN mkdir -p /models && chown -R 10001:10001 /app /models /tmp/uv
USER 10001:10001
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/app/agent_main.py", "start"]
