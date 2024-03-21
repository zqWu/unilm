######## builder ########
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as builder

USER root
ENV env_name=wzq_dit
ENV python_version=3.8


# install miniconda
RUN \
    apt update -y && \
    apt install -y wget ffmpeg libsm6 libxext6 make gcc g++ swig && \
    wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb -O libssl.deb && dpkg -i libssl.deb && rm libssl.deb


# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda-installer.sh \
    && bash /tmp/miniconda-installer.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda-installer.sh


# create conda env
ENV PATH=/opt/miniconda/envs/${env_name}/bin:/opt/miniconda/bin:$PATH
RUN conda create -y -n ${env_name} python=${python_version}


# pip install
RUN conda init bash && . /root/.bashrc && \
    conda activate ${env_name} && \
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /ws
COPY detectron2 /ws/detectron2
RUN conda init bash && . /root/.bashrc && conda activate ${env_name} && pip install -e detectron2

COPY requirements.txt   /ws/requirements.txt
RUN conda init bash && . /root/.bashrc && conda activate ${env_name} && pip install -r /ws/requirements.txt



# in case go to the container
RUN echo "conda activate ${env_name}" >> /root/.bashrc
SHELL ["/bin/bash", "-c"]


# docker build -t dit_runtime:v1 .