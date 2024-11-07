FROM ubuntu:latest
RUN apt update
RUN apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    libfftw3-dev \
    git-lfs \
    wget \
    curl \
    vim
RUN pip install pandas numpy scikit-learn seaborn matplotlib requests statsmodels
RUN find -name "*.pyc" -exec rm {} \;
WORKDIR /WEather-Forecast
COPY data ./data/
