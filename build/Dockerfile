
ARG TRITON_VERSION=2.47.0dev
ARG TRITON_CONTAINER_VERSION=24.06dev
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.05-py3-min


############################################################################
##  Production stage: Create container with just inference server executable
############################################################################
FROM ${BASE_IMAGE}

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH /opt/tritonserver/bin:${PATH}
# Remove once https://github.com/openucx/ucx/pull/9148 is available
# in the min container.
ENV UCX_MEM_EVENTS no

ENV LD_LIBRARY_PATH /opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib/:${LD_LIBRARY_PATH}

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    1

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
RUN userdel tensorrt-server > /dev/null 2>&1 || true \
      && if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
          useradd $TRITON_SERVER_USER; \
        fi \
      && [ `id -u $TRITON_SERVER_USER` -eq 1000 ] \
      && [ `id -g $TRITON_SERVER_USER` -eq 1000 ]

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
RUN apt-get update \
      && apt-get install -y --no-install-recommends \
              clang \
              curl \
              dirmngr \
              git \
              gperf \
              libb64-0d \
              libcurl4-openssl-dev \
              libgoogle-perftools-dev \
              libjemalloc-dev \
              libnuma-dev \
              libre2-9 \
              software-properties-common \
              wget \
              libgomp1 \
      && rm -rf /var/lib/apt/lists/*

# Set TCMALLOC_RELEASE_RATE for users setting LD_PRELOAD with tcmalloc
ENV TCMALLOC_RELEASE_RATE 200

ENV DCGM_VERSION 3.2.6
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN curl -o /tmp/cuda-keyring.deb \
          https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
      && apt install /tmp/cuda-keyring.deb \
      && rm /tmp/cuda-keyring.deb \
      && apt-get update \
      && apt-get install -y datacenter-gpu-manager=1:3.2.6

# Extra defensive wiring for CUDA Compat lib
RUN ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
      && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
      && ldconfig \
      && rm -f ${_CUDA_COMPAT_PATH}/lib

# python3, python3-pip and some pip installs required for the python backend
RUN apt-get update \
      && apt-get install -y --no-install-recommends \
            python3 \
            libarchive-dev \
            python3-pip \
            libpython3-dev \
      && pip3 install --upgrade pip \
      && pip3 install --upgrade \
            wheel \
            setuptools \
            "numpy<2" \
            virtualenv \
      && rm -rf /var/lib/apt/lists/*

# Update Python path to include DALI
ENV PYTHONPATH=/opt/tritonserver/backends/dali/wheel/dali:$PYTHONPATH

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
ENV NVIDIA_PRODUCT_NAME="Triton Server"
COPY docker/entrypoint.d/ /opt/nvidia/entrypoint.d/

ENV NVIDIA_BUILD_ID <unknown>
LABEL com.nvidia.build.id=<unknown>
LABEL com.nvidia.build.ref=

WORKDIR /opt
COPY --chown=1000:1000 build/install tritonserver

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 NVIDIA_Deep_Learning_Container_License.pdf .


LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
COPY --chown=1000:1000 docker/sagemaker/serve /usr/bin/.
