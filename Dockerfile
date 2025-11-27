FROM ubuntu:noble AS base

ENV HOME=/root

RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    python3-html5lib \
    python3-six \
    python3-numpy \
    python3-bleach \
    python3-setuptools \
    python3-venv \  
    markdown \
    build-essential \
    cmake \
    pkg-config \
    libeigen3-dev \
    libassimp-dev \
    libccd-dev \
    libfcl-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libopenscenegraph-dev \
    fuse \
    libnlopt-dev \
    libnlopt-cxx-dev \
    coinor-libipopt-dev \
    libbullet-dev \
    libode-dev \
    liboctomap-dev \
    libflann-dev \
    libtinyxml2-dev \
    liburdfdom-dev \
    libxi-dev \
    libxmu-dev \
    freeglut3-dev \
    ssh \
    gdb \
    vim \
    git \
    sudo

# ROS2 jazzy package
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8


# Install required dependencies and repositories
RUN apt-get install -y software-properties-common curl \
&& apt-add-repository universe \
&& apt-get update \
&& curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null


RUN apt-get update && apt-get -y upgrade

# Install ROS 2 Jazzy
RUN apt-get install -y ros-jazzy-desktop

# Set up the environment
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

# Install development tools
RUN apt-get install -y ros-dev-tools



# Gazebo 
RUN apt-get install -y python3-pip lsb-release gnupg curl
RUN apt-get install -y git

RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros2-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update \
    && apt-get install -y python3-vcstool python3-colcon-common-extensions


RUN mkdir -p /opt/ws_gazebo_harmonic/src && \
    cd /opt/ws_gazebo_harmonic/src && \
    curl -O https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/collection-harmonic.yaml && \
    vcs import < collection-harmonic.yaml



RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update



RUN cd /opt/ws_gazebo_harmonic/src && \
    apt -y install $(sort -u $(find . -iname 'packages-'`lsb_release -cs`'.apt' -o -iname 'packages.apt' | grep -v '/\.git/') | sed '/gz\|sdf/d' | tr '\n' ' ')


# Change to the workspace directory
RUN cd /opt/ws_gazebo_harmonic && \
    colcon graph && \ 
    MAKEFLAGS="-j$(($(nproc)/2))" colcon build --merge-install


RUN echo "source /opt/ws_gazebo_harmonic/install/setup.bash" >> ~/.bashrc


#Install ros-gz
ENV GZ_VERSION=harmonic
# Setup the workspace
RUN mkdir -p /opt/ws_ros_gz/src && \
    cd /opt/ws_ros_gz/src && \
    git clone https://github.com/gazebosim/ros_gz.git -b ros2 && \
    sudo rosdep init && \
    rosdep update 


RUN cd /opt/ws_ros_gz/ && \
    rosdep install -r --from-paths src -i -y --rosdistro jazzy
   
RUN cd /opt/ws_ros_gz/ && \
    apt-get install -y ros-jazzy-gz-tools-vendor ros-jazzy-gz-sim-vendor ros-jazzy-ament-cmake && \
    bash -c "source /opt/ros/jazzy/setup.bash && MAKEFLAGS='-j$(($(nproc)/2))' colcon build"

RUN echo "source /opt/ws_ros_gz/install/setup.bash" >> ~/.bashrc


# Create a Python virtual environment
RUN python3 -m venv /opt/virtual_env

RUN /opt/virtual_env/bin/pip install --upgrade pip

RUN /opt/virtual_env/bin/pip install tensorflow catkin_pkg gymnasium stable-baselines3[extra] pyyaml 

RUN /opt/virtual_env/bin/pip install transforms3d

# Set the virtual environment's bin directory in PATH
ENV PATH="/opt/virtual_env/bin:$PATH"



# Set the working directory (optional)
WORKDIR /opt

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]

