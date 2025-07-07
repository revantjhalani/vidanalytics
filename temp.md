# Uninstall existing TensorFlow
pip uninstall -y tensorflow

# Install TensorFlow with CUDA support
pip install tensorflow[and-cuda]==2.15.1

# Set environment variables for current session
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Make environment variables permanent
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc


# Test PyTorch GPU support
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Test TensorFlow GPU support
CUDA_HOME=/usr/local/cuda-11.8 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH python3 -c "import tensorflow as tf; print(f'TensorFlow GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"

# Test YOLO import
python3 -c "from ultralytics import YOLO; print('✅ ultralytics import successful!')"



# Create a script to run your app with proper environment
cat > run_with_gpu.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
python3 app.py
EOF

# Make it executable
chmod +x run_with_gpu.sh


# Option 1: Use the script
./run_with_gpu.sh

# Option 2: Run with environment variables directly
CUDA_HOME=/usr/local/cuda-11.8 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH python3 app.py


# If you need to reinstall PyTorch with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check CUDA library paths
find /usr/local -name "libcudart.so*" 2>/dev/null

# Verify environment variables are set
echo $CUDA_HOME
echo $LD_LIBRARY_PATH


pip install "numpy<2" && pip uninstall -y tensorflow && pip install tensorflow[and-cuda]==2.15.1 && export CUDA_HOME=/usr/local/cuda-11.8 && export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH && echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc