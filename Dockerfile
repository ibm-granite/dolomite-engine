FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

ENV HOME=/homedir \
    PYTHON_VERSION=3.11 \
    PATH=/opt/conda/envs/ai/bin:/opt/conda/bin:${PATH} \
    BITSANDBYTES_NOWELCOME=1

WORKDIR /app

RUN apt-get -y update && \
    apt-get install -y make git git-lfs curl wget unzip libaio-dev && \
    apt-get -y clean

# taken form pytorch's dockerfile
RUN curl -L -o ./miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm ./miniconda.sh

# create conda env
RUN conda create -n ai python=${PYTHON_VERSION} pip -y

FROM base as conda

# update conda
RUN conda update -n base -c defaults conda -y
# cmake
RUN conda install -c anaconda cmake -y

# necessary stuff
RUN pip install torch --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

COPY ibm-models /app/ibm-models
RUN cd ibm-models && \
    pip install . && \
    cd .. && \
    rm -rf ibm-models

RUN pip install transformers==4.37.2 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.1 \
    safetensors==0.4.1 \
    aim==3.17.5 \
    peft==0.4.0 \
    pydantic \
    jsonlines \
    datasets \
    py-cpuinfo \
    pynvml \
    einops \
    packaging \
    ninja \
    scipy \
    --no-cache-dir

# apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--fast_layer_norm" . && \
    cd .. && \
    rm -rf apex

# deepspeed
RUN git clone https://github.com/microsoft/DeepSpeed && \
    cd DeepSpeed && \
    git checkout v0.13.1 && \
    TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -v --global-option="build_ext" --global-option="-j8" --no-cache-dir .

# flash attention
RUN MAX_JOBS=4 pip install -v flash-attn==2.5.2 --no-cache-dir --no-build-isolation

# clean conda env
RUN conda clean -ya

RUN mkdir -p ~/.cache ~/.local ~/.triton && \
    chmod -R g+w /app ~/.cache ~/.local ~/.triton && \
    touch ~/.aim_profile && chmod g+w ~/.aim_profile && aim telemetry off
