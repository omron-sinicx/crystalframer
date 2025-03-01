FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Setup proxies if needed
# ENV http_proxy 'hogehoge'
# ENV HTTP_PROXY $http_proxy
# ENV HTTPS_PROXY $http_proxy
# ENV FTP_PROXY $http_proxy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y software-properties-common apt-utils git wget curl ca-certificates bzip2 cmake tree htop bmon iotop g++ && \
    apt-get clean

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -y && \
    apt-get upgrade -y libstdc++6 && \
    apt-get clean

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev && \
    apt-get clean

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --upgrade pip

RUN python3.11 -m pip install --no-cache-dir "numpy>=1,<2" scipy matplotlib
RUN pip install --no-cache-dir pymatgen

ARG TORCH=2.1.2
ARG TORCH_PYG=2.1.0
ARG CUDA=cu121

RUN pip install --no-cache-dir \
    torch==${TORCH}+${CUDA} \
    --extra-index-url https://download.pytorch.org/whl/${CUDA}

RUN pip install --no-cache-dir \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    torch_geometric -f https://data.pyg.org/whl/torch-${TORCH_PYG}+${CUDA}.html

RUN pip install --no-cache-dir \
    pytorch-lightning==2.1.3 \
    cupy-cuda12x==12.3.0 pytorch-pfn-extras \
    jarvis-tools \
    tensorboard

ENV CUDA_DEVICE_ORDER PCI_BUS_ID
