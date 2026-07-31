# ffmpeg has to be in the image, not assumed on the host — this is where most
# media pipelines break the first time they leave the developer's laptop.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# For NVDEC/NVENC builds, base on nvidia/cuda and compile ffmpeg with --enable-cuda-nvcc,
# then set HWACCEL=cuda. Decode moves off the CPU, which is what makes
# high-throughput batch rendering viable.
