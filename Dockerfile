FROM python:3.11.14-slim-bookworm

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
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v20.0' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    pip install .

# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
# COPY people_definer_requirements.txt requirements.txt
COPY face_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# COPY uv.lock uv.lock
# COPY pyproject.toml pyproject.toml
# COPY .python-version .python-version
# RUN uv sync 
# RUN mkdir input && mkdir output 
# RUN source .venv/bin/activate

# COPY people_definer.py people_definer.py
COPY face_extract.py face_extract.py

CMD ["python3", "face_extract.py", "-i", "input/", "-o", "output/"]
