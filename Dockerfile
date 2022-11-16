# syntax=docker/dockerfile:1
# ===================
#  Bases
# ===================
# ----- conda -------
FROM continuumio/miniconda3 AS conda
RUN conda install -y pip

WORKDIR /app

COPY ./draft ./draft
COPY ./environments ./environments
COPY ./*.py ./

ENV WORKDIR=/app
ENV PIP_ROOT_USER_ACTION=ignore

# ----- beam --------
FROM apache/beam_python3.9_sdk:2.42.0 as beam

WORKDIR /app

COPY ./draft ./draft
COPY ./environments ./environments
COPY ./*.py ./

ENV WORKDIR=/app
ENV PIP_ROOT_USER_ACTION=ignore

# ===================
#  Targets
# ===================
# ---- collect ------
FROM conda AS collect
ENV ENVIRONMENT=collect
RUN pip install .

# ---- compact ------
FROM conda AS compact
ENV ENVIRONMENT=compact
RUN pip install .

# - compact-worker --
FROM beam as compact-worker
ENV ENVIRONMENT=compact
RUN pip install .

# ----- train -------
FROM conda AS train
ENV ENVIRONMENT=train
RUN pip install .

# ----- eval --------
FROM conda AS eval
ENV ENVIRONMENT=eval
RUN pip install .

# --- dota-draft ----
FROM conda as dota-draft
RUN pip install .
