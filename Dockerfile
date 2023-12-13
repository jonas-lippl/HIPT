FROM nvidia/cuda:11.7.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev \
    python3-pip \
    libffi-dev \
    build-essential \
    rsync \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1


WORKDIR /mnt

RUN pip3 install h5py
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install pillow
RUN pip3 install tqdm
RUN pip3 install einops
RUN pip3 install webdataset
RUN pip3 install matplotlib
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install lifelines
RUN pip3 install scikit-survival
RUN pip3 install scikit-learn
RUN pip3 install opencv-python
RUN pip3 install timm
RUN pip3 install tqdm
RUN pip3 install pamly


ENV MPLCONFIGDIR=/tmp/matplotlib
ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
