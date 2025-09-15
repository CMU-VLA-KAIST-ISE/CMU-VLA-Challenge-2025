FROM ros:noetic-ros-core-focal

# ---------------------- 기본 도구 설치 ----------------------
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    wget \
    gnupg2 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------- rosdep 구성 ----------------------
COPY sources.cache.tar.gz /
COPY 20-default.list /etc/ros/rosdep/sources.list.d/20-default.list

RUN mkdir -p /var/lib/ros/rosdep && \
    tar xzvf /sources.cache.tar.gz -C /var/lib/ros/rosdep

RUN ln -s /usr/bin/python3 /usr/bin/python

# ---------------------- ROS 패키지 설치 ----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    ros-noetic-tf \
    ros-noetic-rviz \
    ros-noetic-pcl-ros \
    ros-noetic-angles \
    ros-noetic-sensor-msgs \
    ros-noetic-ros-numpy \
    ros-noetic-tf2-sensor-msgs \
    python3-scipy \
    ros-noetic-diagnostic-updater \
    ros-noetic-roslint \
    ros-noetic-cv-bridge \
    ros-noetic-diagnostic-aggregator \
    ros-noetic-compressed-image-transport \
    libopencv-dev \
    python3-opencv \
    libusb-0.1-4 \
    libusb-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libopengl0 \
    python3-empy \
    python3-catkin-tools \
    python3-catkin-pkg \
    python3-rospkg \
    python3-pil \ 
    && rm -rf /var/lib/apt/lists/*

# openai 설치
RUN pip3 install --no-cache-dir openai

# ---------------------- CUDA Toolkit 설치 ----------------------
RUN apt-get update && \
    apt-get install -y gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-8 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

 # ---------------------- Conda 설치 ----------------------
 ENV CONDA_DIR /opt/conda
 RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     bash ~/miniconda.sh -b -p $CONDA_DIR && \
     rm ~/miniconda.sh

 # ---------------------- Conda 초기화 및 ROS 호환 설정 ----------------------
 RUN /opt/conda/bin/conda init bash
 RUN /opt/conda/bin/conda config --set always_yes yes --set changeps1 no && \
     /opt/conda/bin/conda config --set auto_activate_base false

# ---------------------- python310 Conda 환경 생성 ----------------------
# Conda 환경 생성
RUN /opt/conda/bin/conda create -n python310 python=3.10 -y -c conda-forge --override-channels
# requirements310.txt 파일을 Docker 이미지 안으로 복사
COPY requirements310.txt .
# requirements310.txt 파일을 이용해 모든 패키지 설치
RUN /opt/conda/bin/conda run -n python310 pip install --no-cache-dir -r requirements310.txt

# ---------------------- 추가 패키지 설치 ----------------------
# 시스템 기본 Python3 환경에 필요한 패키지들 설치
RUN pip3 install --no-cache-dir netifaces defusedxml networkx pydantic-core

# python310 conda 환경에도 동일한 패키지들 설치
RUN /opt/conda/bin/conda run -n python310 pip install --no-cache-dir netifaces defusedxml networkx pydantic-core rospkg



# ---------------------- catkin/cmake가 시스템 파이썬을 쓰도록 고정 ----------------------
ENV PYTHON_EXECUTABLE=/usr/bin/python3
ENV Python3_EXECUTABLE=/usr/bin/python3
ENV CATKIN_PYTHON_VERSION=3

# ---------------------- ROS 설정 ----------------------
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc