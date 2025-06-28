#!/usr/bin/env python3
"""
Script to check CUDA and cuDNN versions and GPU compatibility for faster-whisper
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_package(package_name):
    """Check if a Python package is installed"""
    return importlib.util.find_spec(package_name) is not None

def check_torch_cuda():
    """Check if PyTorch with CUDA is installed and working"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_mem_gb = free_mem / (1024**3)
                total_mem_gb = total_mem / (1024**3)
                
                logger.info(f"GPU {i}: {gpu_name}")
                logger.info(f"  Memory: {free_mem_gb:.2f}GB free / {total_mem_gb:.2f}GB total")
                
            # Check compute capabilities
            try:
                compute_capability = torch.cuda.get_device_capability(0)
                logger.info(f"GPU Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
            except:
                logger.warning("Couldn't determine compute capability")
            
            return True
        else:
            logger.warning("CUDA is not available in PyTorch")
            return False
    except ImportError:
        logger.error("PyTorch is not installed")
        return False
    except Exception as e:
        logger.error(f"Error checking PyTorch CUDA: {str(e)}")
        return False

def check_faster_whisper():
    """Check if faster-whisper is installed and working"""
    try:
        from faster_whisper import WhisperModel
        logger.info("faster-whisper is installed")
        
        # Try importing other components
        from faster_whisper import __version__
        logger.info(f"faster-whisper version: {__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"faster-whisper import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error with faster-whisper: {str(e)}")
        return False

def check_cudnn():
    """Attempt to check cuDNN version"""
    try:
        # First try using ctypes to access the library
        import ctypes
        from ctypes.util import find_library
        
        cudnn_path = find_library('cudnn')
        if cudnn_path:
            try:
                cudnn = ctypes.CDLL(cudnn_path)
                # This would only work if the library exports version symbols
                logger.info(f"Found cuDNN at: {cudnn_path}")
                return True
            except:
                pass
        
        # Try alternative method using subprocess (Linux only)
        try:
            ldconfig = subprocess.check_output('ldconfig -p | grep cudnn', shell=True).decode('utf-8')
            if ldconfig:
                logger.info(f"cuDNN found via ldconfig: {ldconfig.strip()}")
                return True
        except:
            pass
            
        # If both methods fail, check if the cuDNN package is installed via pip
        if check_package('nvidia.cudnn'):
            logger.info("cuDNN is installed via pip package nvidia.cudnn")
            try:
                import nvidia.cudnn
                logger.info(f"nvidia.cudnn package found: {nvidia.cudnn}")
                return True
            except:
                pass
                
        logger.warning("Could not determine cuDNN version")
        return False
    except Exception as e:
        logger.error(f"Error checking cuDNN: {str(e)}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            logger.info(f"FFmpeg version: {version_line}")
            return True
        else:
            logger.warning("FFmpeg not found in PATH")
            return False
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {str(e)}")
        return False

def main():
    """Main function to check all dependencies"""
    logger.info("Checking system compatibility for faster-whisper...")
    logger.info("=" * 60)
    
    # Check PyTorch and CUDA
    gpu_ok = check_torch_cuda()
    
    # Check cuDNN
    cudnn_ok = check_cudnn()
    
    # Check faster-whisper
    faster_whisper_ok = check_faster_whisper()
    
    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"PyTorch with CUDA: {'✅' if gpu_ok else '❌'}")
    logger.info(f"cuDNN libraries: {'✅' if cudnn_ok else '⚠️ (might still work)'}")
    logger.info(f"faster-whisper: {'✅' if faster_whisper_ok else '❌'}")
    logger.info(f"FFmpeg: {'✅' if ffmpeg_ok else '❌'}")
    
    if gpu_ok and faster_whisper_ok and ffmpeg_ok:
        logger.info("✅ System appears to be properly configured for GPU acceleration!")
        if not cudnn_ok:
            logger.info("⚠️ cuDNN status could not be verified, but the system might still work.")
    else:
        logger.warning("⚠️ Some components are missing or not properly configured.")
        if not gpu_ok:
            logger.info("   - Please ensure NVIDIA GPU drivers and CUDA 12+ are installed")
        if not faster_whisper_ok:
            logger.info("   - Please install faster-whisper: pip install faster-whisper")
        if not ffmpeg_ok:
            logger.info("   - Please install ffmpeg")
    
    logger.info("=" * 60)
    logger.info("For detailed setup instructions, see: https://github.com/SYSTRAN/faster-whisper")

if __name__ == "__main__":
    main() 