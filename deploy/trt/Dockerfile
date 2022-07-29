# tensorrt 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/tensorrt:21.06-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install nvidia-pyindex 
Run pip install nvidia-tensorrt
RUN pip install --no-cache -r requirements.txt albumentations wandb gsutil notebook \
    # torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install --no-cache -U torch torchvision

# Create working directory
# RUN mkdir -p /tensorrt
# WORKDIR /tensorrt

# Copy contents
# COPY . /tensorrt

# Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/

# Set environment variables
# ENV HOME=/usr/src/app


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=vuthehuy1997/tensorrt:21.06-py3-cuda10.1 && docker build -t $t . && docker push $t

# Pull and Run
# t=vuthehuy1997/tensorrt:21.06-py3-cuda10.1 && docker pull $t && docker run -it --rm --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=vuthehuy1997/tensorrt:21.06-py3-cuda10.1 && docker pull $t && docker run -it --rm --ipc=host --gpus all -v "$(pwd)":/hi $t

# Kill all
# docker kill $ docker ps -q)

# Kill all image-based
# docker kill $ docker ps -qa --filter ancestor=vuthehuy1997/tensorrt:21.06-py3-cuda10.1)

# Bash into running container
# docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$ docker ps -qa) && docker start $id && docker exec -it $id bash

# Clean up
# docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/vuthehuy1997/tensorrt:21.06-py3-cuda10.1
