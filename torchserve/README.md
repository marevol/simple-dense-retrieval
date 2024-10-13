# Serving Dense Retrieval Models with TorchServe

This guide explains how to serve the query and document encoders of the Simple Dense Retrieval system using TorchServe. It covers how to build the model archives, set up the API using Docker, and make requests to the running service.

## Prerequisites

Before proceeding, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- Trained models from the **Simple Dense Retrieval** project.

## Building the Model Archives and Docker Image

The provided `build.sh` script will generate the `.mar` files for the query and document encoders, and then build a Docker image for serving the models.

### Steps to Build:

1. For CPU:
   - Run the following command to build the `.mar` files and the Docker image for CPU-based TorchServe:
     ```bash
     ./build.sh
     ```

2. For GPU:
   - To build the `.mar` files and Docker image using the GPU-based TorchServe, use the `--gpu` flag:
     ```bash
     ./build.sh --gpu
     ```

This process creates:

- The model archive files (`query_encoder.mar` and `document_encoder.mar`) in the `model_store` directory.
- A Docker image (`marevol/simple-dense-retrieval:latest` for CPU or `marevol/simple-dense-retrieval:latest-gpu` for GPU) that includes TorchServe and the necessary model files.

## Running the API Servers

Once the `.mar` files and Docker image are built, you can start the servers for both query encoder and document encoder.

### For CPU:

- Query Encoder API:
  ```bash
  docker run --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 \
    -v ./model_store:/home/model-server/model-store \
    marevol/simple-dense-retrieval:latest \
    torchserve --model-store=/home/model-server/model-store --models query_encoder=query_encoder.mar --disable-token-auth
  ```

- Document Encoder API:
  ```bash
  docker run --rm -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 \
    -v ./model_store:/home/model-server/model-store \
    marevol/simple-dense-retrieval:latest \
    torchserve --model-store=/home/model-server/model-store --models document_encoder=document_encoder.mar --disable-token-auth
  ```

### For GPU:

- Query Encoder API:
  ```bash
  docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 \
    -v ./model_store:/home/model-server/model-store \
    marevol/simple-dense-retrieval:latest-gpu \
    torchserve --model-store=/home/model-server/model-store --models query_encoder=query_encoder.mar --disable-token-auth
  ```

- Document Encoder API:
  ```bash
  docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 \
    -v ./model_store:/home/model-server/model-store \
    marevol/simple-dense-retrieval:latest-gpu \
    torchserve --model-store=/home/model-server/model-store --models document_encoder=document_encoder.mar --disable-token-auth
  ```

## Making API Requests

Once the servers are running, you can make HTTP requests to get the embeddings for queries or documents.

- Query Encoder API

To get the embeddings for a query, send a POST request to the /predictions/query_encoder endpoint:

```bash
curl -X POST http://127.0.0.1:8080/predictions/query_encoder \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Example query text"
  }'
```

The response will be the query embeddings as a JSON array.

- Document Encoder API

To get the embeddings for documents, send a POST request to the /predictions/document_encoder endpoint. The request should include the document’s title, brand, and color:

```bash
curl -X POST http://127.0.0.1:8080/predictions/document_encoder \
  -H "Content-Type: application/json" \
  -d '[
      {"title": "Product 1", "brand": "Brand A", "color": "Red"},
      {"title": "Product 2", "brand": "Brand B", "color": "Blue"},
      {"title": "Product 3", "brand": "Brand C", "color": "Green"}
    ]'
```

The response will be the document embeddings as a JSON array.

## Directory Structure

The model archives (.mar files) are stored in the ./model_store directory. Make sure to mount this directory inside the Docker container using the -v option, as shown in the examples above.

- model_store/: Contains the model archive files (query_encoder.mar, document_encoder.mar).
- build.sh: Script to build the .mar files and the Docker image.
- Dockerfile: Docker configuration for building the image with TorchServe.
