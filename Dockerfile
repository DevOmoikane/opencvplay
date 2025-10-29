FROM python:3.10.12-slim-bookworm

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
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
    python3-numpy \
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
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

WORKDIR /app
COPY people_definer_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY people_definer.py people_definer.py

CMD ["python3", "people_definer.py"]
