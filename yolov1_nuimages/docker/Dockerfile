FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT_NAME="2d-object-detection-experiments"

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 vim -y

WORKDIR /root

RUN mkdir /root/repositories
RUN mkdir /root/repositories/${PROJECT_NAME}
RUN mkdir /root/datasets
RUN mkdir /root/datasets/${PROJECT_NAME}
RUN mkdir /root/models
RUN mkdir /root/models/${PROJECT_NAME}
RUN mkdir /root/results
RUN mkdir /root/results/${PROJECT_NAME}

ENV PHD_REPOSITORIES="/root/repositories"
ENV PHD_DATASETS="/root/datasets"
ENV PHD_MODELS="/root/models"
ENV PHD_RESULTS="/root/results"

ENV PYTHONPATH="/root/repositories/${PROJECT_NAME}"
ENV TWODOBJECTDETECTION_ROOT="/root/repositories/${PROJECT_NAME}"

ENV WANDB_DIR="/root/results/${PROJECT_NAME}"

ENV USER="lucasrdalcol"