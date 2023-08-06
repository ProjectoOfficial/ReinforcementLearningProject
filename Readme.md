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
4. when you will need to push you have to:
    ```
    git push --set-upstream origin devel
    ```

### Docker Instruction
Docker OS: Ubuntu 20.04 \
NVIDIA CUDA Version: 11.3 \
Python version: 3.9

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
5. (Just in Case) docker permission error:
    ```
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    reboot
    ```

### Introduction
here we want to present you a possible implementation of the Reinforcement Learning algorithm called PPO (Proximal Policy Optimization), which tries to be as genuine as possible, without exploiting the _torchrl library_. The project was created for the Reinforcement Learning exam held at the University of Modena and Reggio Emilia. The project structure is as follows:
```
src
├── actors
│   └── actor_utils.py
│   └── actor.py
├── configs
│   └── CartPole.json
│   └── LunarLander.json
│   └── ...
├── utils
│   └── utils.py
├── customlogger.py
├── main.sh
├── ppo.py
├── requirements.txt
├── train.py
└── test.py
```

Listed here you will be able to better understand each individual component:

1. **actor_utils.py:** this file contains the functions necessary to build, starting from the configuration, the neural network that will carry out the predictions
2. **actor.py:** here is contained the actor class, which integrates the neural network that predicts the policies and the neural network that criticizes these policies. the functions for carrying out the predictions and for updating and evaluating the predictions of the policy network are also implemented here
3. **configs ... :** inside the configs folder are the configurations to make it all work. a configuration usually contains the name of the environment and the parameters of the environment itself (understood as a _gym_ environment). It also contains all the PPO configuration parameters and the configuration of the neural network layers.
4. **utils.py :** all the boundary functions are contained here, useful for making prints, reading the configuration, setting the random seed, performing the sanity check, etc...
5. **customlogger.py:** this logger is a class that saves logging data with a frequency set in the environment configuration file and this data is saved both in CSV format and in tensorboard.
6. **ppo.py:**: here is the scratch implementation of PPO
7. **train.py:** train script
8. **test.py:** test script

### Execution
Train script:
1. configuration absolute path (mandatory)
2. experiments output path (mandatory)
3. project name output folder (mandatory)
```
python /home/user/src/train.py /home/user/src/configs/CartPole.json /home/user/src/experiments CartPole --verbose
```

Test script:
1. configuration absolute path (mandatory)
2. experiments output path (mandatory)
3. project name output folder (mandatory)
```
python /home/user/src/train.py /home/user/src/configs/CartPole.json /home/user/src/experiments CartPole --ckpt-number 499 --test-episodes 10 --verbose
```