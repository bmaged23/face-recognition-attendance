FROM python:3.12.15-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt && python -m pip check

FROM runtime AS assets
COPY face_recognition ./face_recognition
COPY scripts/download_model.py ./scripts/download_model.py
COPY scripts/setup/export_inference_checkpoint.py ./scripts/setup/export_inference_checkpoint.py
RUN python scripts/download_model.py && python scripts/setup/export_inference_checkpoint.py \
    && python -c "from scripts.download_model import digest; from pathlib import Path; assert digest(Path('vendor/adaface/pretrained/adaface_ir100_webface12m.backbone.pt')) == '2bafbe30750677b7165236dbe05054a179c716f29fcbd2f47475aaeaa61acd95'"

FROM runtime AS application
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/bmaged23/face-recognition-attendance" \
      org.opencontainers.image.title="Face Recognition and Attendance" \
      org.opencontainers.image.revision=$VCS_REF
COPY face_recognition ./face_recognition
COPY vendor ./vendor
COPY scripts ./scripts
COPY tests ./tests
COPY enroll.py video_recognition.py live_camera_recognition.py ./
COPY --from=assets /app/vendor/adaface/pretrained/adaface_ir100_webface12m.backbone.pt ./vendor/adaface/pretrained/
COPY --from=assets /app/vendor/adaface/pretrained/adaface_ir100_webface12m.backbone.json ./vendor/adaface/pretrained/
ENV DEVICE=cuda QDRANT_PATH=/app/data/gallery OUTPUT_DIR=/app/data/output \
    ADAFACE_MODEL_PATH=/app/vendor/adaface/pretrained/adaface_ir100_webface12m.backbone.pt \
    HOME=/app/data/home XDG_CACHE_HOME=/app/data/cache
RUN mkdir -p /app/data/home && chown -R 1000:1000 /app/data && chmod 700 /app/data
USER 1000:1000
EXPOSE 7323
HEALTHCHECK --interval=30s --start-period=120s --timeout=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7323/ready', timeout=35)"
CMD ["python", "-m", "uvicorn", "face_recognition.api.app:app", "--host", "0.0.0.0", "--port", "7323", "--workers", "1"]
