FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN \
  --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt-get update && \
  apt-get install -y --no-install-recommends locales && \
  locale-gen ja_JP.UTF-8 && \
  update-locale LANG=ja_JP.UTF-8

RUN \
  --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  x11-apps \  
  mesa-utils \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1

RUN \
  --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  python3-pip \
  cmake \
  libopencv-dev \
  ninja-build \
  clangd \
  curl \
  cmake-format \
  clang-format

# Install ROS2
RUN \
  --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt update \
  && apt install -y --no-install-recommends \
  curl gnupg2 lsb-release python3-pip vim wget build-essential ca-certificates

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN \
  --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt update \
  && apt install -y --no-install-recommends \
  ros-humble-desktop

RUN \
  --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  gdb

RUN \
  --mount=type=bind,source=assets/onnxruntime-linux-x64-gpu-1.22.0.tgz,target=/tmp/onnxruntime.tgz \
  tar -xzvf /tmp/onnxruntime.tgz -C /tmp && \
  find /tmp/onnxruntime-linux-x64-gpu-1.22.0/lib/cmake -type f -exec grep -l "lib64" {} \; | xargs sed -i 's/lib64/lib/g' && \
  cp -r /tmp/onnxruntime-linux-x64-gpu-1.22.0/include /usr/local/include/onnxruntime && \
  cp -r /tmp/onnxruntime-linux-x64-gpu-1.22.0/lib /usr/local &&\
  rm -rf /tmp/onnxruntime-linux-x64-gpu-1.22.0

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID


# ホストと同じユーザーを作成（エラー回避のため早めに実行）
RUN \
  mkdir -p /etc/sudoers.d \
  && groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo "$USERNAME:$USERNAME" | chpasswd \
  && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# ワークディレクトリを設定
WORKDIR /home/$USERNAME/workspace
RUN chown $USER_UID:$USER_GID /home/$USERNAME/workspace
USER $USERNAME

RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/onnxruntime-linux-x64-gpu-1.22.0/lib" >> ~/.bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
