# OS AND ENVIRONMENT SETTINGS
# Here we set as OS Ubuntu 22.04 with CUDA version 11.3
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# the docker's user will be 'user' and its home will be '/user/home'
ARG USER_NAME=user
ARG USER_HOME=/home/$USER_NAME

# to be sure we set non interactive bash also here
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y -q \
    x11-apps mesa-utils 

# remove all the packages within Debian base configuration (not wasting time installing things that will not be used)
RUN rm -f /etc/apt/sources.list.d/*.list

# install Ubuntu Software needed for the development (DEBIAN_FRONTEND="noninteractive" needed to avoid human interaction in the process)
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" && apt-get install -y -q\
    sudo \
    bash-completion \
    build-essential \
    ffmpeg \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-tk \
&& rm -rf /var/lib/apt/lists/*

# set python update alternatives - the highest is the preferred one
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

# remove python2
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip



# USER SETTINGS

# create a new user within the Docker container
RUN adduser --disabled-password --gecos '' --shell /bin/bash $USER_NAME

# create directory /src e set user as the owner
RUN mkdir -p /src && chown -R $USER_NAME:$USER_NAME /src

# create directory /etc/sudoers.d if it does not exists
RUN mkdir -p /etc/sudoers.d

# add user to video group (useful to use webcam)
RUN usermod -a -G video $USER_NAME

# set user working directory
WORKDIR $USER_HOME

# set sudo password
RUN echo 'root:Docker!' | chpasswd

# sets the user context within the Docker container.
USER $USER_NAME

# upgrade python pip
RUN pip install --upgrade pip



# FINAL SETUPS

# install pytorch torchvision and torchaudio specific for cuda 11.3
RUN pip -v install --timeout=99999 torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# install python packages in /src/requirements.txt
COPY ./src/requirements.txt ./requirements.txt
RUN python -m pip install -r ./requirements.txt

# remove all the created/copied/moved file by the docker
RUN rm -rf *

# when the container is launched it will start a bash session
CMD ["/bin/bash"]
