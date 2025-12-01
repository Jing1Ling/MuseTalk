
# Use the specified base image
FROM vault.habana.ai/gaudi-docker/1.21.3/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest


# Proxy support (only set if provided)
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

# Set environment variables
ENV PATH=/root/.local/bin:${PATH}
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PT_HPU_LAZY_MODE=0

# Update system packages and install Git, Git LFS, FFmpeg, curl
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# git user config
RUN git config --global user.name "Examples"
RUN git config --global user.email "examples@intel.com"

# Set the working directory
WORKDIR /workspace

# Copy all files from local patches directory to /workspace/patches, overwriting existing files
# This ensures /workspace/patches contains the latest patches from the build context
COPY patches/. /workspace/patches/

# Setup git repo for MuseTalk
ARG MT_REPO=https://github.com/TMElyralab/MuseTalk.git
ARG MT_COMMIT=0a89dec45a0192b824e3cf4daf96c239440c5ed8
ARG MT_PATCHES_DIR=/workspace/patches
RUN git clone $MT_REPO
WORKDIR /workspace/MuseTalk
RUN git checkout $MT_COMMIT \
    && git submodule update --init --recursive \
    && git am $MT_PATCHES_DIR/*

# Install Python dependencies and openmim
RUN pip install -r requirements.txt \
    && pip install --no-cache-dir -U openmim \
    && mim install mmengine \
    && mim install "mmdet==3.1.0" \
    && mim install "mmpose==1.1.0" \
    && pip install mmcv-lite==2.0.1

# Patch mmcv and mmengine
RUN mkdir -p /usr/local/lib/python3.10/dist-packages/mmcv/_ext \
    && touch /usr/local/lib/python3.10/dist-packages/mmcv/_ext/__init__.py \
    && sed -i '14s/^/## /;15s/^/## /' /usr/local/lib/python3.10/dist-packages/mmcv/utils/ext_loader.py \
    && sed -i '347s/torch.load(filename, map_location=map_location)/torch.load(filename, map_location=map_location, weights_only=False)/' /usr/local/lib/python3.10/dist-packages/mmengine/runner/checkpoint.py

# Install gdown and download model weights
RUN pip install --no-cache-dir gdown \
    && bash download_weights.sh \
    && test -f models/musetalk/musetalk.json \
    && test -f models/musetalk/pytorch_model.bin \
    && test -f models/musetalkV15/musetalk.json \
    && test -f models/musetalkV15/unet.pth \
    && test -f models/sd-vae/config.json \
    && test -f models/sd-vae/diffusion_pytorch_model.bin \
    && test -f models/whisper/config.json \
    && test -f models/whisper/pytorch_model.bin \
    && test -f models/whisper/preprocessor_config.json \
    && test -f models/dwpose/dw-ll_ucoco_384.pth \
    && test -f models/syncnet/latentsync_syncnet.pt \
    && test -f models/face-parse-bisent/79999_iter.pth \
    && test -f models/face-parse-bisent/resnet18-5c106cde.pth

# Default command (can be modified as needed)
CMD ["/bin/bash"]
