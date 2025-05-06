#!/bin/bash
# setup.sh - Installation script for vidTranscriber

set -e  # Exit on any error

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${BOLD}vidTranscriber Setup Script${NC}"
echo "This script will set up the vidTranscriber environment"
echo "========================================================"

# Check if running as root and warn
if [ "$EUID" -eq 0 ]; then
  echo -e "${YELLOW}Warning: Running as root. It's recommended to install Python packages as a regular user.${NC}"
  read -p "Continue as root? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting setup."
    exit 1
  fi
fi

# Check if Python 3.10+ is installed
echo -e "\n${BOLD}Checking Python version...${NC}"
if command -v python3 &>/dev/null; then
  PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  echo "Found Python $PY_VERSION"
  
  if (( $(echo "$PY_VERSION < 3.10" | bc -l) )); then
    echo -e "${RED}Error: Python 3.10 or higher is required.${NC}"
    echo "Please install a compatible Python version and try again."
    exit 1
  fi
else
  echo -e "${RED}Error: Python 3 not found. Please install Python 3.10 or higher.${NC}"
  exit 1
fi

# Check for virtual environment
echo -e "\n${BOLD}Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
  echo "Virtual environment already exists."
  read -p "Recreate virtual environment? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
    python3 -m venv venv
    echo -e "${GREEN}New virtual environment created.${NC}"
  fi
else
  echo "Creating virtual environment..."
  python3 -m venv venv
  echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "\n${BOLD}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Install dependencies
echo -e "\n${BOLD}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed.${NC}"

# Check for FFmpeg
echo -e "\n${BOLD}Checking for FFmpeg...${NC}"
if command -v ffmpeg &>/dev/null; then
  echo -e "${GREEN}FFmpeg is installed.${NC}"
else
  echo -e "${YELLOW}FFmpeg not found. vidTranscriber requires FFmpeg for audio extraction.${NC}"
  echo "Install FFmpeg using your package manager, for example:"
  echo "sudo apt install ffmpeg  # Ubuntu/Debian"
  echo "sudo yum install ffmpeg  # CentOS/RHEL"
  echo "brew install ffmpeg      # macOS with Homebrew"
fi

# Create .env file if it doesn't exist
echo -e "\n${BOLD}Setting up configuration...${NC}"
if [ ! -f ".env" ]; then
  echo "Creating .env file from template..."
  cp .env.example .env
  echo -e "${GREEN}.env file created. You may want to edit this file to adjust settings.${NC}"
else
  echo ".env file already exists."
fi

# Create required directories
echo -e "\n${BOLD}Setting up directories...${NC}"
# Get directory values from .env file
TEMP_DIR=$(grep "TEMP_DIR" .env | cut -d '=' -f2)
OUTPUT_DIR=$(grep "OUTPUT_DIR" .env | cut -d '=' -f2)

# If not found in .env, use defaults
TEMP_DIR=${TEMP_DIR:-/tmp/vidTranscriber}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/vidTranscriberOutput}

# Create directories
mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Created temporary directory: $TEMP_DIR${NC}"
echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"

# Run GPU check
echo -e "\n${BOLD}Checking GPU compatibility...${NC}"
python check_gpu_setup.py

# Final instructions
echo -e "\n${BOLD}Setup Complete!${NC}"
echo -e "${GREEN}vidTranscriber has been set up successfully.${NC}"
echo
echo "To run vidTranscriber:"
echo "1. Activate the virtual environment if not already active:"
echo "   source venv/bin/activate"
echo "2. Start the server:"
echo "   python run.py"
echo
echo "The server will be available at: http://0.0.0.0:8509"
echo
echo "For more information, refer to the README.md file." 