# vidTranscriber

A high-performance API service for transcribing videos using faster-whisper with GPU acceleration.

## Overview

vidTranscriber provides a fast, reliable way to transcribe videos from URLs with:
- GPU acceleration support via CUDA
- Automatic fallback to CPU when needed
- Asynchronous processing for larger videos
- Synchronous response for quick transcriptions
- Support for all OpenAI Whisper models

The system leverages [faster-whisper](https://github.com/guillaumekln/faster-whisper), an optimized implementation of OpenAI's Whisper model that provides significantly improved performance.

## Features

- **Multiple Operation Modes**:
  - Synchronous mode returns full transcription immediately for quick jobs
  - Asynchronous mode with job polling for longer videos
  - Configurable wait times for balanced performance

- **Video Processing**:
  - Automatic download from HTTP/HTTPS URLs
  - Audio extraction and normalization
  - Voice activity detection (VAD) to filter out silence
  - Word-level timestamps

- **Rich Output**:
  - Full text transcription
  - Segmented transcriptions with timestamps
  - Word-level timestamps option
  - Markdown-formatted output
  - JSON output with full details

- **Reliability**:
  - Graceful shutdown handling
  - Error recovery and logging
  - GPU to CPU fallback for compatibility issues
  - Signal handling (SIGTERM, SIGINT)

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for acceleration)
- CUDA Toolkit and cuDNN libraries (for GPU support)
- FFmpeg (for audio extraction)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jaywalked78/vidTranscriber.git
   cd vidTranscriber
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   # venv\Scripts\activate  # On Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. If using GPU acceleration, ensure CUDA libraries are installed:
   ```bash
   # For Ubuntu/Debian:
   sudo apt install nvidia-cuda-toolkit libcudnn9 libcudnn9-dev
   ```

5. Configure environment variables (create a `.env` file based on `.env.example`):
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

## Usage

### Starting the server

```bash
# Start the server
python run.py

# The server will start on http://0.0.0.0:8509 by default
```

### API Endpoints

#### Transcribe a Video URL

```bash
# Synchronous mode (wait for result)
curl -X POST "http://localhost:8509/transcribe/url" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "https://example.com/video.mp4"}'

# Asynchronous mode (get job_id immediately)
curl -X POST "http://localhost:8509/transcribe/url?wait_for_result=false" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "https://example.com/video.mp4"}'

# Custom wait time before falling back to async
curl -X POST "http://localhost:8509/transcribe/url?max_wait_time=10.0" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "https://example.com/video.mp4"}'
```

#### Check Job Status

```bash
curl "http://localhost:8509/transcribe/{job_id}"
```

#### Get Transcription Result (when complete)

```bash
curl "http://localhost:8509/transcribe/{job_id}/result"
```

#### Get Raw JSON output

```bash
curl "http://localhost:8509/transcribe/{job_id}/json"
```

## Configuration Options

Configuration can be done via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8509` |
| `MODEL_NAME` | Whisper model to use (tiny, base, small, medium, large-v1/2/3, turbo) | `turbo` |
| `DEVICE` | Device to use for inference (`cuda` or `cpu`) | `cuda` if available |
| `FORCE_CPU` | Force CPU usage even if GPU is available | `false` |
| `COMPUTE_TYPE` | Compute precision (float16, float32, int8) | `float16` for GPU, `float32` for CPU |
| `BATCH_SIZE` | Batch size for inference | `8` |
| `MAX_CONCURRENT_JOBS` | Maximum number of concurrent transcription jobs | `2` |
| `MAX_VIDEO_SIZE_MB` | Maximum video file size in MB | `500` |
| `MAX_VIDEO_DURATION_MINUTES` | Maximum video duration in minutes | `60` |
| `TEMP_DIR` | Directory for temporary files | `/tmp/vidTranscriber` |
| `OUTPUT_DIR` | Directory for output files | `/tmp/vidTranscriberOutput` |
| `REQUEST_LIMIT_PER_MINUTE` | Rate limit for API requests | `10` |

## Performance Tips

- GPU acceleration provides 4-10x faster transcription than CPU
- For best GPU performance, use `float16` computation type
- For systems with less GPU memory, try the `int8_float16` compute type
- Adjust batch size based on your GPU memory (higher = faster, but more memory)
- Use the smallest model that meets your accuracy needs:
  - `tiny`: Fastest, lowest quality
  - `base`: Fast with reasonable quality
  - `small`: Good balance for most use cases
  - `medium`: High quality, moderate speed
  - `large-v3` or `turbo`: Highest quality, slowest

## Troubleshooting

### CUDA/GPU Issues

If you encounter CUDA-related errors:

1. Make sure you have the correct CUDA libraries installed:
   ```bash
   # For CUDA 12.x
   sudo apt install libcudnn9 libcudnn9-dev
   ```

2. Use environment variables to force CPU mode temporarily:
   ```bash
   export FORCE_CPU=true
   export CUDA_VISIBLE_DEVICES=""
   python run.py
   ```

3. Check CUDA is properly set up:
   ```bash
   python check_gpu_setup.py
   ```

## Changelog

### v1.0.0 (May 2025)

- Initial release with core transcription capabilities

### v1.1.0 (May 2025)

- Added GPU acceleration with faster-whisper
- Implemented batch processing for improved performance

### v1.2.0 (May 2025)

- Added graceful shutdown handling
- Improved error handling and recovery
- Added signal handling for proper cleanup
- Fixed CUDA compatibility issues
- Added automatic fallback to CPU when GPU fails

### v1.3.0 (May 2025)

- Implemented synchronous mode for quick transcriptions
- Added wait_for_result parameter for flexibility
- Enhanced API response with full transcription content
- Improved error reporting and status updates

## License

MIT License

## Credits

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - For the optimized Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - For the original speech recognition model
- [FastAPI](https://fastapi.tiangolo.com/) - For the web framework 