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
    vim 

RUN pip3 install pandas numpy scikit-learn seaborn matplotlib requests statsmodels

WORKDIR /Weather-Forecast 

COPY data ./data/
COPY predictor ./predictor/
COPY Makefile ./
COPY requirements.txt ./


COPY . .

RUN find -name "*.pyc" -exec rm {} \;

CMD ["python3", "main.py"]

RUN make
