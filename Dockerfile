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

# Set up virtual environment
WORKDIR /Weather-Forecast
RUN python3 -m venv /Weather-Forecast/venv
RUN . /Weather-Forecast/venv/bin/activate && pip install --upgrade pip && pip install pandas numpy scikit-learn seaborn matplotlib requests statsmodels

COPY data ./data/
COPY predictor ./predictor/
COPY Makefile ./
COPY requirements.txt ./


COPY . .

RUN find -name "*.pyc" -exec rm {} \;

CMD ["/Weather-Forecast/venv/bin/python", "main.py"]

RUN make
