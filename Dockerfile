FROM ros:humble-ros-base

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    ros-humble-demo-nodes-cpp \
    ros-humble-demo-nodes-py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/flow-ros2
RUN mkdir -p /workspaces/flow-ros2/src