FROM python:3.11-slim

RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static/ ./static/

# Fractus source cloned from the PUBLIC fractus-cte repo.
# Checkpoint (palier0) downloaded from thefinalboss/fractus-cte by app.py.
# KB persists to /data (HF Space persistent storage).
ENV FRACTUS_PATH=/app/fractus_src \
    FRACTUS_KB_PATH=/data/fractus_kb.pkl \
    PORT=7860

RUN printf '#!/bin/sh\n\
if [ ! -d "$FRACTUS_PATH" ]; then\n\
  echo "[Fractus] Cloning fractus-cte from GitHub..."\n\
  git clone --depth 1 https://github.com/AFKmoney/fractus-cte.git "$FRACTUS_PATH"\n\
fi\n\
echo "[Fractus] Starting server..."\n\
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}\n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
