ARG BASE_IMAGE=python:3.8-slim-buster
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get -y --no-install-recommends install build-essential git openssh-client && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install --no-cache-dir prophet \
    && rm -rf ~/.cache
WORKDIR /code

CMD [ "bash" ]