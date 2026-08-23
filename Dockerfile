FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY spoilage ./spoilage
COPY web ./web
COPY samples ./samples
COPY tests ./tests

EXPOSE 7860
CMD ["uvicorn", "spoilage.api:app", "--host", "0.0.0.0", "--port", "7860"]
