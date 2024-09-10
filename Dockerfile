FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime 

# Update the package list.
RUN apt-get update

# Install dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev libjpeg-dev libpng-dev xvfb ffmpeg xorg-dev \
    xorg-dev libboost-all-dev libsdl2-dev swig \
    libblas-dev liblapack-dev \
    libopenblas-base libatlas-base-dev graphviz \
    libvips \
    && apt-get upgrade -y \
    && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev libffi-dev \
    libatlas-base-dev gfortran \
    software-properties-common \
    && apt upgrade -y

RUN apt-get update && apt-get install -y \
    nano less libgl1 libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

# Copy the requirements.txt file.
COPY install/requirements.txt /workspace/requirements.txt 

# Change the working directory.
WORKDIR /workspace

RUN pip install opencv-python

# Install the requirements.
RUN pip install -r requirements.txt


CMD ["bash"]
