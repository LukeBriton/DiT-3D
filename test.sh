# After adding '-allow-unsupported-compiler', install CUDA 12.4+,
# cuz "The CUDA Toolkit 12.4 release introduces support for GCC 13 as a host-side compiler"
# https://forums.developer.nvidia.com/t/identifier-float32-is-undefined-etc-cuda-12-2-0-gcc-13-1/258930
# wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run
# Select only 'CUDA Toolkit' during installation
# bash cuda_13.1.0_590.44.01_linux.run --toolkit --toolkitpath=/data/lhc/cuda13.1

# Specify the nvcc path
# export CUDA_HOME=/data/lhc/cuda13.1
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python test.py --dataroot ../ShapeNetCore.v2.PC15k/ \
    --category chair --num_classes 1 \
    --bs 64 \
    --model_type 'DiT-S/4' \
    --voxel_size 32 \
    --model "./DiT-S-4.pth" \
    --gpu 0