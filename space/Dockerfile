FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY fractus_pkg/ ./fractus/
COPY static/ ./static/

# Fractus source is BAKED into the image (no git clone at runtime).
# Checkpoint (palier0) downloaded from thefinalboss/fractus-cte by app.py.
ENV FRACTUS_PATH=/app \
    FRACTUS_KB_PATH=/data/fractus_kb.pkl \
    PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
