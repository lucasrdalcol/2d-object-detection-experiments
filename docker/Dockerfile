FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ENV PROJECT_NAME="2d-object-detection-experiments"

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

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

ENV USER="lucasrdalcol"