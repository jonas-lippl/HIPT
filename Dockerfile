FROM nvidia/cuda:11.7.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev \
    python3-pip \
    libffi-dev \
    build-essential \
    rsync


WORKDIR /mnt

RUN pip3 hash
RUN pip3 install numpy
RUN pip3 pandas
RUN pip3 pillow
RUN pip3 tqdm
RUN pip3 einops
RUN pip3 webdataset
RUN pip3 install matplotlib
RUN pip3 opencv-python==4.4.0.46
RUN pip3 torch
RUN pip3 torchvision
RUN pip3 lifelines
RUN pip3 install scikit-survival
RUN pip3 install scikit-learn


ENV MPLCONFIGDIR=/tmp/matplotlib
ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
