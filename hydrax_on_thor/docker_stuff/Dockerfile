FROM ghcr.io/nvidia/jax:jax-2025-12-17

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

# System deps for MuJoCo's headless rendering and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglew-dev libosmesa6 \
 && rm -rf /var/lib/apt/lists/*

# MuJoCo/MJX and Hydrax's pure-Python deps
RUN pip install --break-system-packages --no-cache-dir \
        mujoco mujoco-mjx \
        "evosax>=0.2.0" "flax>=0.10.0" \
        "huggingface_hub>=0.29.3" "interpax>=0.3.7"

# --- ROS 2 Jazzy ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl gnupg lsb-release software-properties-common ca-certificates \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
 && add-apt-repository universe \
 && rm -rf /var/lib/apt/lists/*

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F "tag_name" | awk -F\" '{print $4}') \
 && curl -L -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" \
 && apt-get install -y /tmp/ros2-apt-source.deb \
 && rm /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        python3-colcon-common-extensions \
        python3-rosdep \
 && rosdep init \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=jazzy

RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

# Hydrax fork — snapshot for editable registration; live code comes from the runtime mount
COPY hydrax /workspace/hydrax
RUN pip install --break-system-packages --no-deps -e /workspace/hydrax

WORKDIR /workspace/hydrax
CMD ["/bin/bash"]
