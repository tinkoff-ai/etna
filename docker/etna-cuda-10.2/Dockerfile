ARG BASE_IMAGE=nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
ARG DISTRO_ARCH=ubuntu1804/x86_64
ARG CUDA_VERSION_FOR_TORCH=cu102

FROM ${BASE_IMAGE}

ARG CUDA_VERSION_FOR_TORCH
ARG DISTRO_ARCH

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get -y update \
    && apt-get install --no-install-recommends  -y software-properties-common curl git openssh-client \
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && apt-get -y update \
#     && apt-get install --no-install-recommends  -y software-properties-common curl git openssh-client \
#     && apt-key del 7fa2af80 \
#     && curl -O https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO_ARCH/cuda-keyring_1.0-1_all.deb \
#     && dpkg -i cuda-keyring_1.0-1_all. \
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