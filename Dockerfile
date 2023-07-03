FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as base

ENV HOME=/homedir \
    PYTHON_VERSION=3.9 \
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
RUN pip install torch \
    transformers==4.30.1 \
    accelerate==0.20.3 \
    bitsandbytes==0.39.1 \
    aim==3.17.5 \
    peft==0.3.0 \
    pydantic \
    jsonlines \
    datasets \
    py-cpuinfo \
    pynvml \
    einops \
    packaging \
    ninja \
    --no-cache-dir

# fms
COPY foundation-model-stack/fm /app/foundation-model-stack/fm
COPY foundation-model-stack/nlp /app/foundation-model-stack/nlp
COPY foundation-model-stack/version.py /app/foundation-model-stack/version.py
COPY foundation-model-stack/README.md /app/foundation-model-stack/README.md
RUN cd foundation-model-stack && \
    pip install ./fm ./nlp && \
    cd .. && \
    rm -rf foundation-model-stack

# apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout b3e3bab && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
    cd .. && \
    rm -rf apex

# deepspeed
RUN git clone https://github.com/microsoft/DeepSpeed && \
    cd DeepSpeed && \
    git checkout v0.9.5 && \
    TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -v --global-option="build_ext" --global-option="-j8" --no-cache-dir .

# flash attention
RUN pip install flash-attn==1.0.4 --no-cache-dir

# clean conda env
RUN conda clean -ya

RUN mkdir -p ~/.cache ~/.local && \
    chmod -R g+w /app ~/.cache ~/.local && \
    touch ~/.aim_profile && chmod g+w ~/.aim_profile && aim telemetry off
