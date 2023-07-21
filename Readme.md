# Reinforcement Learning Project
This project is an integral part of the material to take the Reinforcement Learning exam (9 CFU) of the University of Modena and Reggio Emilia.

### Devel Branch Instruction
the devel branch, as you can guess from the name, is the branch used to develop the code. Once satisfied with the version of the code obtained, it is possible to make a pull request against the branch master.

To start working it is necessary to follow the following steps:

1. if you don't already have one, create a workspace:
    ```
    mkdir ~/developments
    cd ~/developments
    ```
2. clone the repository from github:
    ```
    git clone https://github.com/ProjectoOfficial/ReinforcementLearningProject.git
    cd ReinforcementLearningProject
    ```
3. move to _devel_ branch:
    ```
    git checkout -b devel
    git fetch
    git pull
    ```

### Docker Instruction
Docker OS: Ubuntu 20.04 \
NVIDIA CUDA Version: 11.3 

1. Please follow docker base installation:
    ```
    https://docs.docker.com/engine/install/
    ```

2. Install nvidia-docker2 for GPU support:
    ```
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    ```

3. Build docker image:
    please note that here you are downloading ubuntu, cuda and pytorch, and it may take several minutes
    ```
    ./docker/build.sh
    ```

4. run docker image:
    ```
    ./docker/run.sh
    ```