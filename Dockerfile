# StageBuild: Zero123++ + InstantMesh in one CUDA-ready image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata git git-lfs wget unzip ffmpeg ca-certificates python3-opencv \
    build-essential cmake ninja-build pkg-config python3-dev \
    libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    colmap \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
ARG USERNAME=app
RUN useradd -ms /bin/bash ${USERNAME} \
 && mkdir -p /workspace \
 && chown -R ${USERNAME}:${USERNAME} /workspace
USER ${USERNAME}
WORKDIR /workspace

# Python venv (tidy site-packages)
RUN python -m venv /workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Keep pip modern
RUN pip install --upgrade pip wheel setuptools

# --- Code: clone repos + install deps ---
# Zero123++
RUN git clone https://github.com/SUDO-AI-3D/zero123plus.git && \
    pip install -r zero123plus/requirements.txt || true

# InstantMesh
RUN git clone https://github.com/TencentARC/InstantMesh.git && \
    pip install -r InstantMesh/requirements.txt || true

# (Optional) TripoSR as a fast baseline
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git && \
    pip install -r TripoSR/requirements.txt || true

# Diffusers-based fallback for Zero123Plus when .pth is unavailable
RUN pip install --no-cache-dir \
    diffusers==0.30.0 \
    transformers==4.44.2 \
    accelerate==0.34.2 \
    safetensors==0.4.5 \
    Pillow==10.4.0 \
    huggingface_hub==0.35.3 \
    boto3==1.35.36

# TripoSR runtime deps that require compilation (needs CUDA devel image)
RUN pip install --no-cache-dir \
    git+https://github.com/tatsy/torchmcubes.git \
    xatlas==0.0.9 \
    trimesh==4.4.9 \
    onnxruntime==1.18.1 \
    rembg==2.0.57 || true

# Open3D for view rendering (optional, large)
RUN pip install --no-cache-dir open3d==0.17.0 || true

ENV HF_HOME=/workspace/.cache/huggingface

# Copy our helper scripts into the image
USER root
RUN mkdir -p /workspace/scripts
COPY scripts/pipeline.sh /workspace/scripts/pipeline.sh
COPY scripts/pipeline_splat.sh /workspace/scripts/pipeline_splat.sh
COPY scripts/generate_views_diffusers.py /workspace/scripts/generate_views_diffusers.py
COPY scripts/generate_views_stable.py /workspace/scripts/generate_views_stable.py
COPY scripts/meshpush.py /workspace/scripts/meshpush.py
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/scripts/*.sh /workspace/scripts/*.py /workspace/entrypoint.sh

# Working dirs that we'll mount from host/cloud
RUN mkdir -p /workspace/input /workspace/output /workspace/checkpoints \
 && chown -R ${USERNAME}:${USERNAME} /workspace
USER ${USERNAME}

ENTRYPOINT ["/workspace/entrypoint.sh"]