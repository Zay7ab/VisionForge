FROM python:3.10-slim

WORKDIR /app

# ✅ Minimal install - no OpenGL needed
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt
RUN pip install --no-cache-dir --timeout=300 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY main.py .

RUN mkdir -p saved_models

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]