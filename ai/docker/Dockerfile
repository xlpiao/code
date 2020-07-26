FROM ubuntu:bionic

RUN apt update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        software-properties-common \
        vim \
        git

RUN pip3 --no-cache-dir install \
        matplotlib \
        numpy \
        scipy \
        tensorflow

WORKDIR /root/work
