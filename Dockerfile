FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace
ADD requirements.txt requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt