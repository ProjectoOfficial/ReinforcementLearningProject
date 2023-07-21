#!/bin/bash

# Nome dell'immagine Docker
IMAGE_NAME="devel"
IMAGE_TAG="1.0.0"

# Nome del container
CONTAINER_NAME="ReinforcementProject"

# Percorso assoluto del codice sorgente che desideri montare nel container
# Assicurati di modificare "/path/to/your/src/folder" con il percorso corretto
PATH_TO_SRC_FOLDER="~/github/ReinforcementLearningProject/src"

# Esegui il container
docker run  --shm-size 2GB -it --gpus all \
                --name $CONTAINER_NAME \
                --user $(id -u):$(id -g) \
                $IMAGE_NAME:$IMAGE_TAG

docker rm $CONTAINER_NAME