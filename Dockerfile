# GFM reproducible environment (review item 7).
# Build:  docker build -t gfm .
# Run:    docker run --gpus all -v $(pwd):/workspace/gfm -e GFM_PROJECT_ROOT=/workspace/gfm gfm \
#             python /workspace/gfm/scripts/40_pretrain.py --epochs 500
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 \
    GFM_PROJECT_ROOT=/workspace/gfm

WORKDIR /workspace

RUN pip install --upgrade pip \
 && pip install torch-geometric numpy pandas scikit-learn scipy tqdm fair-esm

COPY . /workspace/gfm

CMD ["bash"]
