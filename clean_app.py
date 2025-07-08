#!/usr/bin/env python3
import os
import sys

# Suppress CUDA warnings by setting environment variables before importing anything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Explicitly set GPU device
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

# Suppress specific CUDA factory warnings
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_UNIFICATION'] = '1'

# Import and run the original app
if __name__ == "__main__":
    import app