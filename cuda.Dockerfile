FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG UID
ARG GID

#install python
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    sqlite3 \
    software-properties-common \
    zip \
    supervisor \
    ffmpeg \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libsm6 \
    libice6 \
    && apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v20.0' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd dlib/ && \
    mkdir build && cd build && \
    cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build . --config Release -- -j$(nproc) && \
    cd ../ && \
    pip install --break-system-packages .

WORKDIR /app
COPY face_requirements.txt requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

USER ${UID}:${GID}

COPY face_extract.py ./
COPY similarity_gpu.py ./

CMD ["python3", "api_service.py"]
