# syntax=docker/dockerfile:1
FROM continuumio/miniconda3
RUN conda update conda
RUN conda install -y pip

WORKDIR /app

COPY ./draft ./draft
COPY ./environments ./environments
COPY ./*.py ./

ENV WORKDIR=/app
ENV ENVIRONMENT=eval
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip
RUN pip install .
