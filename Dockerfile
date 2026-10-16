FROM python:3.11-slim
LABEL version=v1.0.0
LABEL app=navigation-vision

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-xcb1 \
    libxrender1 \
    libxi6 \
    libxtst6 \
    libfontconfig1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


COPY requierements2.txt .
RUN pip install --no-cache-dir -r requierements2.txt


COPY . .

RUN python setup.py build_ext --inplace

CMD ["python", "video_inference.py"]