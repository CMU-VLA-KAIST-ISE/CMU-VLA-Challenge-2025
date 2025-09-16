# CMU-VLA-Challenge

## Table of Contents
[Introduction](#introduction)  

[Setting Up](#setting-up)
- [API Key](#api-key)
- [Docker Composition](#docker-composition)
- [Simulator](#simulator)
- [Execution](#execution)


## Introduction
This is the final submission repository for [CMU-VLA Challenge](https://www.ai-meets-autonomy.com/cmu-vla-challenge), by **KAIST-ISE Team**. \
Below provides a step-by-step manual for environment setup and code execution.

## Setting Up
Clone the entire `CMU-VLA-Challenge-2025` repository to your local `/home/user` folder, so that it works as the working directory in the Docker image.

### API Key
We sent an additional `.env` file that contains private API keys, arranged as follows:
```
OPENAI_API_KEY=sk...
GEMINI_API_KEY=A...
```
Download the file and place it in the repository folder. 
```
CMU-VLA-Challenge-2025
├─ .env
├─ ai_module
├─ system
├─ ...
 
```

### Docker Composition
Please make sure the computer contains **NVIDIA GPUs**.
#### 1) Install Docker and grant user permission:
This step is referenced from the [baseline repository](https://github.com/CMU-VLA-KAIST-ISE/CMU-VLA-Challenge-2025/tree/main/docker#2-for-computers-with-nvidia-gpus).
If your computer has already gone through such a process, please feel free to skip.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to restart the computer, then install Nvidia Container Toolkit (Nvidia GPU Driver should be installed already).
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Configure Docker runtime and restart the Docker daemon.
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### 2) Run and Modify Docker Image
Allow remote X connection inside the `CMU-VLA-Challenge-2025` folder:
```
xhost +
```
Go inside the `docker` folder in the terminal:
```
cd CMU-VLA-Challenge-2025/docker/
```
Add a new environment variable for display:
```
export DISPLAY=[YOUR_IP_ADDRESS]
```
You can find your IP address by running 
```
ipconfig
```
in your local terminal. 

Pull Docker images by Docker Compose: 

⚠️ **Caution**: There are two different compose files. 

* `compose_gpu.yml`: Fork of the organizers' original compose file. Only the images are changed to provide our images. **Use this on Linux servers with GPUs.** 
* `compose_gpu_wsl.yml`: Variant tuned for WSL2 (Windows) to enable GPU. **Use this for local development in WSL environment** 

Please choose the accurate compose file for your computer.
```
docker compose -f compose_gpu.yml up -d
docker compose -f compose_gpu_wsl.yml up -d
```

### Simulator

Copy the scene files into the `system/unity/src/vehicle_simulator/mesh/unity` folder. \
Then run XLaunch on your local computer.

### Execution
#### 1) `ubuntu20_ros_system` container
Access the first container:
```
docker exec -it ubuntu20_ros_system bash
```
Move to `/system/unity`, and build:
```
cd system/unity/
catkin_make
```
Go back to the `CMU-VLA-Challenge-2025`, and run:
```
./launch_system.sh
```

_※ If Unity Simulator doesn't show up, try adding execution permission to the `Model.x86_64` file, and restart `./launch_system.sh`:_
```
chmod +x ./system/unity/src/vehicle_simulator/mesh/unity/environment/Model.x86_64
```
_※ Our mode requires at least 15fps. If your simulator shows a low rate, close the container and rearrange Dockers by:_
```
docker compose -f compose_gpu.yml up -d --no-deps --force-recreate ubuntu20_ros ubuntu20_ros_system
```

#### 2) `ubuntu20_ros` container
Open a new terminal, and access the second container:
```
docker exec -it ubuntu20_ros bash
```
Go inside the `ai_module/` folder, and compile:
```
cd ai_module/
catkin_make
```
Once the compilation is done, run the development setup:
```
source devel/setup.bash
```
Finally, move to the `CMU-VLA-Challenge-2025`, run the shell script below and enter questions.
```
./launch_module.sh
```

_※ If the LLM is not working and you're using the Linux system, please try `dos2unix` in your `.env` file_
```
dos2unix .env
```


## Acknowledgements
Thank you to [AlphaZ](https://alpha-z.ai/) for sponsoring the challenge for 2025! \
For any kind of issues and questions, please feel free to contact us.
