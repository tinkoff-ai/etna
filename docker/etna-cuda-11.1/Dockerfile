ARG BASE_IMAGE=nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04
ARG CUDA_VERSION_FOR_TORCH=cu111

FROM ${BASE_IMAGE}

ARG CUDA_VERSION_FOR_TORCH

RUN apt-get -y update \
    && apt-get install --no-install-recommends  -y software-properties-common curl git openssh-client \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get -y update \
    && apt-get --no-install-recommends  -y install build-essential python3.8 python3.8-distutils python3.8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/${CUDA_VERSION_FOR_TORCH}/torch_stable.html
WORKDIR /code

CMD [ "bash" ]