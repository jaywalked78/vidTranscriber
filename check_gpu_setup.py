#!/usr/bin/env python3
"""
A utility script to check GPU setup and CUDA compatibility for vidTranscriber.
"""

import os
import sys
import ctypes
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(" " * 5 + title)
    print("=" * 60 + "\n")

def check_python_version():
    """Check if Python version is supported"""
    print_header("PYTHON VERSION")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    print(f"Python version: {version_str}")
    
    if v.major != 3 or v.minor < 10:
        print("❌ Python 3.10+ is required. Please upgrade your Python version.")
    else:
        print("✅ Python version OK")

def check_required_libraries():
    """Check if required libraries are installed"""
    print_header("REQUIRED LIBRARIES")
    libraries = [
        "torch", 
        "faster_whisper", 
        "fastapi", 
        "pydantic", 
        "uvicorn", 
        "python-dotenv",
        "ffmpeg-python"
    ]
    
    missing = []
    for lib in libraries:
        try:
            module = __import__(lib)
            print(f"✅ {lib} is installed: {getattr(module, '__version__', 'Unknown version')}")
        except ImportError:
            print(f"❌ {lib} is not installed")
            missing.append(lib)
    
    if missing:
        print("\nInstall missing libraries with:")
        print(f"pip install {' '.join(missing)}")

def check_cuda_libraries():
    """Check if CUDA libraries are available"""
    print_header("CUDA LIBRARIES")
    
    # Check CUDA runtime
    cuda_available = False
    try:
        ctypes.CDLL("libcudart.so")
        print("✅ CUDA runtime (libcudart.so) found")
        cuda_available = True
    except OSError:
        print("❌ CUDA runtime (libcudart.so) not found")
    
    # Check cuDNN
    cudnn_available = False
    if cuda_available:
        try:
            ctypes.CDLL("libcudnn.so")
            print("✅ cuDNN (libcudnn.so) found")
            cudnn_available = True
        except OSError:
            print("❌ cuDNN (libcudnn.so) not found")
    
    # Check specific cuDNN ops libraries (needed for faster-whisper)
    if cudnn_available:
        cudnn_ops_files = [
            "libcudnn_ops.so", 
            "libcudnn_ops.so.9", 
            "libcudnn_ops.so.9.1", 
            "libcudnn_ops.so.9.1.0"
        ]
        
        for lib in cudnn_ops_files:
            try:
                ctypes.CDLL(lib)
                print(f"✅ {lib} found")
                break
            except OSError:
                if lib == cudnn_ops_files[-1]:
                    print(f"❌ None of these cuDNN ops libraries found: {', '.join(cudnn_ops_files)}")
                    print("   This might cause 'Cannot load symbol cudnnCreateTensorDescriptor' errors")
                    print("   Consider installing libcudnn9 and libcudnn9-dev packages")

def check_torch_cuda():
    """Check if PyTorch CUDA is working properly"""
    print_header("PYTORCH CUDA SUPPORT")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA device count: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"CUDA device {i}: {device_name}")
            
            # Try a simple CUDA operation
            try:
                x = torch.rand(5, 5).cuda()
                y = torch.rand(5, 5).cuda()
                z = x + y
                z.cpu()  # Move back to CPU
                print("✅ PyTorch CUDA operations test: PASSED")
            except Exception as e:
                print(f"❌ PyTorch CUDA operations test FAILED: {str(e)}")
        else:
            print("❌ PyTorch doesn't detect CUDA. Check your installation.")
    
    except ImportError:
        print("❌ PyTorch is not installed")

def check_ffmpeg():
    """Check if FFmpeg is installed and working"""
    print_header("FFMPEG")
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        
        if result.returncode == 0:
            ffmpeg_version = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg is installed: {ffmpeg_version}")
        else:
            print("❌ FFmpeg is installed but returned an error")
            print(result.stderr)
            
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH")
        print("   Install FFmpeg using your package manager")
        print("   For example: sudo apt install ffmpeg")

def check_faster_whisper():
    """Check if faster-whisper is working properly"""
    print_header("FASTER-WHISPER")
    
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper is installed")
        
        # Check if a test model can be loaded
        try:
            print("Attempting to load a tiny model to test faster-whisper...")
            model = WhisperModel("tiny", device="cpu", compute_type="float32")
            print("✅ Successfully loaded a test model")
        except Exception as e:
            print(f"❌ Failed to load test model: {str(e)}")
            
    except ImportError:
        print("❌ faster-whisper is not installed")

def check_environment_variables():
    """Check environment variables and configuration"""
    print_header("ENVIRONMENT VARIABLES")
    
    env_file = Path('.env')
    if env_file.exists():
        print(f"✅ Found .env file at {env_file.absolute()}")
    else:
        print("❌ .env file not found. Consider creating one from .env.example")
    
    important_vars = [
        "MODEL_NAME", "DEVICE", "FORCE_CPU", "COMPUTE_TYPE", "BATCH_SIZE",
        "TEMP_DIR", "OUTPUT_DIR", "PORT"
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var} is set to '{value}'")
        else:
            print(f"ℹ️ {var} is not set, will use default value")

def print_summary():
    """Print a summary of recommendations based on the checks"""
    print_header("SUMMARY & RECOMMENDATIONS")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print("GPU MODE RECOMMENDATIONS:")
            print("✅ Use the following in your .env file:")
            print("   DEVICE=cuda")
            print("   FORCE_CPU=false")
            print("   COMPUTE_TYPE=float16")
            
            # If CUDA is available but ops libraries weren't found earlier
            try:
                ctypes.CDLL("libcudnn_ops.so")
            except OSError:
                print("")
                print("⚠️ CUDA is available but cuDNN ops libraries weren't found.")
                print("   You might encounter the 'Cannot load symbol cudnnCreateTensorDescriptor' error.")
                print("   Solutions:")
                print("   1. Install cuDNN packages: sudo apt install libcudnn9 libcudnn9-dev")
                print("   2. Or use CPU mode by setting: FORCE_CPU=true")
                
        else:
            print("CPU MODE RECOMMENDATIONS:")
            print("✅ Use the following in your .env file:")
            print("   DEVICE=cpu")
            print("   FORCE_CPU=true")
            print("   COMPUTE_TYPE=float32")
    except ImportError:
        print("⚠️ PyTorch not installed. Install it to determine the best settings.")

def main():
    """Run all checks"""
    print("\nvidTranscriber GPU/CUDA Setup Check Tool")
    print("This tool checks your environment for vidTranscriber compatibility")
    print("---------------------------------------------------------------\n")
    
    check_python_version()
    check_required_libraries()
    check_cuda_libraries()
    check_torch_cuda()
    check_ffmpeg()
    check_faster_whisper()
    check_environment_variables()
    print_summary()
    
    print("\nCheck complete! See above for any issues or recommendations.")

if __name__ == "__main__":
    main() 