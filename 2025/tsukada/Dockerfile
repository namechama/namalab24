FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG work_dir="/workspace"

# ENV PYTHON_VERSION 3.10.9
# ENV HOME /root
# ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
# ENV PATH $PYTHON_ROOT/bin:$PATH
# ENV PYENV_ROOT $HOME/.pyenv
#gitのインストール
RUN apt-get update -y && apt-get install -y build-essential vim \
    wget curl git zip gcc make openssl \
    libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev python3-tk tk-dev python-tk \
    libfreetype6-dev libffi-dev liblzma-dev

# Install nodejs for JupyterLab extension
RUN curl -sL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv
ENV HOME  /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv --version
RUN pyenv install 3.12
RUN pyenv global 3.12
RUN python --version
RUN pyenv rehash &&  pip install --upgrade pip
# JupyterLab関連のパッケージ（いくつかの拡張機能を含む）
# 必要に応じて、JupyterLabの拡張機能などを追加してください

# COPY ./requirements.txt ./
RUN pip install lit transformers accelerate
RUN python3 -m pip install torch torchvision torchaudio
