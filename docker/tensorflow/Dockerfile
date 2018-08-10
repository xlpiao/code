FROM ubuntu:16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3.5 \
        python3.5-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        rsync \
        software-properties-common \
        unzip \
        libsm6 \
        locales \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        wheel \
        opencv-python \
        && \
    python3 -m ipykernel.kernelspec

# Install TensorFlow CPU version (1.9.0)
RUN pip3 --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.9.0-cp35-cp35m-linux_x86_64.whl

# TensorBoard
EXPOSE 6006

## setting locale to print text correctly
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /root/work
