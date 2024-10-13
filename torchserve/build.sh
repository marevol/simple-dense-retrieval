#!/bin/bash

# Base directory setup
base_dir=$(cd $(dirname $0); cd ..; pwd)

model_version=1.0
create_on=$(date +"%m/%d/%Y %H:%M:%S")

model_store_dir=${base_dir}/torchserve/model_store
query_encoder_dir=${model_store_dir}/query_encoder
document_encoder_dir=${model_store_dir}/document_encoder

model_dir=${base_dir}/dr_model

# Default values
docker_image="pytorch/torchserve:latest"  # CPU version as default
gpu_flag=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) 
            docker_image="pytorch/torchserve:latest-gpu"  # Use GPU version if --gpu is passed
            gpu_flag=1
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Function to build query encoder model archive
function build_query_encoder() {
  rm -rf "${query_encoder_dir}.mar" "${query_encoder_dir}"
  mkdir -p "${query_encoder_dir}/MAR-INF" "${query_encoder_dir}/simple_dense_retrieval"

  cp "${base_dir}/torchserve/query_handler.py" "${query_encoder_dir}"
  cp -r "${model_dir}/query_encoder" "${query_encoder_dir}"
  mv "${query_encoder_dir}/query_encoder/encoder.pth" "${query_encoder_dir}"
  cp -r "${base_dir}/simple_dense_retrieval/"*.py "${query_encoder_dir}/simple_dense_retrieval"
  echo "{
    \"createdOn\": \"${create_on}\",
    \"runtime\": \"python\",
    \"model\": {
      \"modelName\": \"query_encoder\",
      \"serializedFile\": \"encoder.pth\",
      \"handler\": \"query_handler.py\",
      \"modelVersion\": \"${model_version}\"
    },
    \"archiverVersion\": \"0.12.0\"
  }" > "${query_encoder_dir}/MAR-INF/MANIFEST.json"

  cd "${query_encoder_dir}" || exit
  zip -r "${query_encoder_dir}.mar" * || { echo "Failed to create query_encoder MAR file"; exit 1; }
  cd "${base_dir}" || exit
}

# Function to build document encoder model archive
function build_document_encoder() {
  rm -rf "${document_encoder_dir}.mar" "${document_encoder_dir}"
  mkdir -p "${document_encoder_dir}/MAR-INF" "${document_encoder_dir}/simple_dense_retrieval"

  cp "${base_dir}/torchserve/document_handler.py" "${document_encoder_dir}"
  cp -r "${model_dir}/document_encoder" "${document_encoder_dir}"
  mv "${document_encoder_dir}/document_encoder/encoder.pth" "${document_encoder_dir}"
  mv "${document_encoder_dir}/document_encoder/"*.pkl "${document_encoder_dir}"
  cp -r "${base_dir}/simple_dense_retrieval/"*.py "${document_encoder_dir}/simple_dense_retrieval"
  echo "{
    \"createdOn\": \"${create_on}\",
    \"runtime\": \"python\",
    \"model\": {
      \"modelName\": \"document_encoder\",
      \"serializedFile\": \"encoder.pth\",
      \"handler\": \"document_handler.py\",
      \"modelVersion\": \"${model_version}\"
    },
    \"archiverVersion\": \"0.12.0\"
  }" > "${document_encoder_dir}/MAR-INF/MANIFEST.json"

  cd "${document_encoder_dir}" || exit
  zip -r "${document_encoder_dir}.mar" * || { echo "Failed to create document_encoder MAR file"; exit 1; }
  cd "${base_dir}" || exit
}

# Build the Docker image
function build_docker_image() {
  dockerfile="${base_dir}/torchserve/Dockerfile"

  echo "Building Docker image with ${docker_image}..."
  docker build --build-arg BASE_IMAGE=${docker_image} -t marevol/simple-dense-retrieval:latest${gpu_flag:+-gpu} -f "${dockerfile}" "${base_dir}/torchserve" || { echo "Docker build failed"; exit 1; }
}

# Main process
echo "Building MAR file for Query Encoder..."
build_query_encoder

echo "Building MAR file for Document Encoder..."
build_document_encoder

echo "Building Docker image..."
build_docker_image

echo "Build process completed successfully."
