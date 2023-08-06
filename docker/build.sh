#!/bin/bash

# Ubuntu 20.04 with CUDA 11.3
IMAGE_NAME="devel"
IMAGE_TAG="1.0.0"

cd "$(dirname "$0")"

# Percorso assoluto del Dockerfile
MAIN_DIR="$PWD"/..

# Esegui la build dell'immagine Docker
docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f "$MAIN_DIR/docker/Dockerfile" "$MAIN_DIR"


# Controlla se la build è stata completata con successo
if [ $? -eq 0 ]; then
    echo "Build dell'immagine completata con successo."
else
    echo "Si è verificato un errore durante la build dell'immagine."
fi