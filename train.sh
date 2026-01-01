# Symlink e.g.
# ln -s /data/lhc/DiC-3D/DiC/results/005-DiC-B/checkpoints/0200000.pt ./DiC-B-20W.pt

# Specify the nvcc path
export CUDA_HOME=/data/lhc/cuda13.1 # cuda12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# For RTX 3090 24GiB
# DiC-S

## DiffFit PEFT
NCCL_DEBUG=INFO python train.py --use_tb --model_type DiC-S --experiment_name DiC-3D-S-5W-difffit --dic_ckpt "./DiC-S-5W.pt" --peft --peft_mode difffit --peft_report --voxel_size 32 --bs 20 --nc 3 --num_classes 55 --dataroot ../ShapeNetCore.v2.PC15k/ --category chair --niter 1000 --saveIter 10 --diagIter 10 --vizIter 10

## full fine-tuning # --bs 22 if not visualize, otherwise OOM
NCCL_DEBUG=INFO python train.py --use_tb --model_type DiC-S --experiment_name DiC-3D-S-5W --dic_ckpt "./DiC-S-5W.pt" --voxel_size 32 --bs 20 --nc 3 --num_classes 55 --dataroot ../ShapeNetCore.v2.PC15k/ --category chair --niter 1000 --saveIter 10 --diagIter 10 --vizIter 10

# DiC-B

## DiffFit PEFT
NCCL_DEBUG=INFO python train.py --use_tb --model_type DiC-B --experiment_name DiC-3D-B-5W-difffit --dic_ckpt "./DiC-B-5W.pt" --peft --peft_mode difffit --peft_report --voxel_size 32 --bs 8 --nc 3 --num_classes 55 --dataroot ../ShapeNetCore.v2.PC15k/ --category chair --niter 1000 --saveIter 10 --diagIter 10 --vizIter 10

## full fine-tuning # --bs 9 if not visualize, otherwise OOM
NCCL_DEBUG=INFO python train.py --use_tb --model_type DiC-B --experiment_name DiC-3D-B-5W --dic_ckpt "./DiC-B-5W.pt" --voxel_size 32 --bs 8 --nc 3 --num_classes 55 --dataroot ../ShapeNetCore.v2.PC15k/ --category chair --niter 1000 --saveIter 10 --diagIter 10 --vizIter 10
