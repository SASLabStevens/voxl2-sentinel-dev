# Start from an Ubuntu 18.04 base image for ARM64 architecture
FROM arm64v8/ubuntu:18.04

# Avoid prompts from apt during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system libraries for Pillow and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    libjpeg-dev zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev libtiff5-dev \
    libwebp-dev \
    software-properties-common \
    sudo \
    --no-install-recommends \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && add-apt-repository ppa:inivation-ppa/inivation-bionic -y \
    && apt-get update \
    && apt-get install -y dv-processing-python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools
# Attempt to install Pillow again, hopefully with dependencies resolved
RUN pip3 install Pillow

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the Python script into the container
COPY . .

# Command to run the script
CMD ["python3", "./event_camera.py"]


#docker run --device=/dev/bus/usb/002/003:/dev/bus/usb/002/003 -v $(pwd)/metadata/event:/usr/src/app/metadata/event -it voxlevent

