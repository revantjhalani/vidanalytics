#!/usr/bin/env python3
import subprocess
import time
import threading
import sys
import os
import json
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics using nvidia-ml-py or fallback methods"""
    try:
        # Try using nvidia-ml-py if available
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Get GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Get temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        # Get power usage
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
        
        return {
            'memory_used': mem_info.used / 1024**3,  # GB
            'memory_total': mem_info.total / 1024**3,  # GB
            'memory_percent': (mem_info.used / mem_info.total) * 100,
            'gpu_util': util.gpu,
            'memory_util': util.memory,
            'temperature': temp,
            'power_usage': power
        }
    except Exception as e:
        print(f"pynvml not available: {e}")
        return None

def monitor_gpu_continuous():
    """Monitor GPU usage continuously"""
    print("GPU Monitoring Started...")
    print("=" * 80)
    
    while True:
        stats = get_gpu_stats()
        if stats:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] GPU: {stats['gpu_util']:3d}% | "
                  f"Memory: {stats['memory_used']:.1f}/{stats['memory_total']:.1f}GB "
                  f"({stats['memory_percent']:.1f}%) | "
                  f"Temp: {stats['temperature']:2d}°C | "
                  f"Power: {stats['power_usage']:.1f}W")
        else:
            print("GPU stats not available")
        
        time.sleep(1)

def check_pytorch_gpu_usage():
    """Check if PyTorch is using GPU"""
    try:
        import torch
        print("PyTorch GPU Check:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    except ImportError:
        print("PyTorch not available")

def test_model_gpu_usage():
    """Test if YOLO model uses GPU"""
    try:
        from ultralytics import YOLO
        import torch
        import numpy as np
        
        print("\nTesting YOLO GPU Usage:")
        model = YOLO('yolo11n.pt')
        
        # Force model to GPU
        model.to('cuda')
        
        print(f"  Model device: {model.device}")
        print(f"  Model parameters on GPU: {next(model.model.parameters()).is_cuda}")
        
        # Test inference
        print("  Running inference test...")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Monitor memory before inference
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        
        results = model(dummy_image, verbose=False)
        
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory used for inference: {mem_after - mem_before:.2f} GB")
        print(f"  Total GPU memory allocated: {mem_after:.2f} GB")
        
    except Exception as e:
        print(f"Error testing YOLO: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_gpu_continuous()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        check_pytorch_gpu_usage()
        test_model_gpu_usage()
    else:
        print("Usage:")
        print("  python gpu_monitor.py test    - Test GPU usage")
        print("  python gpu_monitor.py monitor - Monitor GPU continuously")