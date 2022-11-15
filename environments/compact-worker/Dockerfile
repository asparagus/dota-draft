# syntax=docker/dockerfile:1
FROM apache/beam_python3.9_sdk:2.42.0

WORKDIR /app

COPY ./draft ./draft
COPY ./environments ./environments
COPY ./*.py ./

ENV WORKDIR=/app
ENV ENVIRONMENT=compact
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip
RUN pip install .
