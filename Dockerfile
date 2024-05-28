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
RUN pip3 install xlrd
RUN pip3 install openpyxl
RUN pip3 install lifelines
RUN pip3 install scikit-survival
RUN pip3 install numba
RUN pip3 install umap-learn
RUN pip3 install captum
RUN pip3 install transformers


ENV MPLCONFIGDIR=/tmp/matplotlib
ENV OMP_NUM_THREADS=1
ENV HF_HOME=/tmp/huggingface
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
