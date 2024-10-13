ARG BASE_IMAGE=pytorch/torchserve:latest
FROM ${BASE_IMAGE}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt