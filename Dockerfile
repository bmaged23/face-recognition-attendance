FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY face_recognition ./face_recognition
COPY vendor ./vendor
COPY enroll.py video_recognition.py live_camera_recognition.py ./
EXPOSE 7323
CMD ["uvicorn", "face_recognition.api.app:app", "--host", "0.0.0.0", "--port", "7323"]
