#!/bin/bash

# Clean GPU startup script for video analytics app
# This replaces the outdated run_with_gpu.sh

echo "Starting Video Analytics Application..."

# Set environment variables to suppress CUDA warnings
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_DISABLE_SEGMENT_REDUCTION_OP_UNIFICATION=1

# Set PyTorch to use optimal settings
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3080 Ti architecture
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Check if GPU is available
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('✗ GPU not available, running on CPU')
"

echo "Starting application..."
python3 app.py