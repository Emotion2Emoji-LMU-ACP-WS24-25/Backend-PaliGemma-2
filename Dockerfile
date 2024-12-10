# Basis-Image mit Python und CUDA-Unterstützung
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Umgebungsvariablen setzen
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends

# Arbeitsverzeichnis setzen
WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Skript kopieren (falls nötig)
COPY app.py /app/app.py

# Standardkommando
CMD ["python", "app.py"]