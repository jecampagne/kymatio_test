FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libcudnn8 \
      python3-appdirs \
      python3-mako \
      python3-numpy \
      python3-pytest \
      python3-pytest-cov \
      python3-pytools \
      python3-pip \
      python3-venv \
      python3-yaml \
      curl \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install \
      'numpy>=1.16' \
      scipy \
      configparser \
      'torchvision==0.8.2' \
      torch \
      jaxlib \
      jax \
      scikit-cuda \
      cupy \
      tensorflow \
      scikit-learn
