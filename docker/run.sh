#!/bin/bash

IMAGE_NAME="devel"
IMAGE_TAG="1.0.0"

CONTAINER_NAME="ReinforcementProject"

PATH_TO_SRC_FOLDER="~/github/ReinforcementLearningProject/src"


MOUNT_SRC_PATH="-v $(dirname $PWD)/src:/home/user/src"


docker run  --shm-size 2GB -it --gpus all \
        ${MOUNT_SRC_PATH} \
        -e http_proxy -e https_proxy \
        -e HOST_USER_ID=$(id -u) -e HOST_GROUP_ID=$(id -g) \
        --name $CONTAINER_NAME \
        --user $(id -u):$(id -g) \
        --net='host' \
        --group-add video \
        $IMAGE_NAME:$IMAGE_TAG

docker rm $CONTAINER_NAME