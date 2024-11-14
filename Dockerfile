FROM ubuntu:latest

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    libfftw3-dev \
    git-lfs \
    wget \
    curl \
    vim \
    numpy

WORKDIR /Weather-Forecast 

COPY data ./data/
COPY predictor ./predictor/
COPY Makefile ./
COPY . .

RUN find -name "*.pyc" -exec rm {} \;

CMD ["python3", "main.py"]

RUN make
