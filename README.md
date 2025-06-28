# FastAPI Video Transcription Service

A powerful API service for transcribing videos using GPU-accelerated Whisper model.

## 🚀 Quick Start

```bash
# 1. Setup
git clone <repository>
cd vidTranscriber
bash setup.sh

# 2. Configure
cp .env.example .env
nano .env  # Adjust paths and settings

# 3. Run
python run.py

# 4. Use
curl -X POST http://localhost:8509/transcribe/url \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4"}'
```

## 📋 Key Features

- **No timeouts**: Handles 12+ hour videos without timing out
- **Fast processing**: ~1 minute per hour of audio (GPU)
- **Large file support**: Up to 10GB videos (configurable)
- **Smart filtering**: VAD removes non-speech segments
- **Full configuration**: 22 environment variables for customization
- **JSON output**: Automatic saving with timestamps

## Changelog

### Version 1.1.1 (2025-06-27)

#### 📝 Documentation & Configuration
- **Comprehensive environment variables**: All 22 hardcoded values now configurable via `.env`
- **Environment reference guide**: Added `ENV_VARIABLES_REFERENCE.md` with complete variable documentation
- **Directory structure clarification**: Documented the distinction between `/models/` and `/app/models/`
- **Cleanup notes**: Identified unused `vidTranscriber/vidTranscriber/` nested directory

### Version 1.1.0 (2025-06-26)

#### 🚀 Major Improvements
- **Removed timeout limitations**: API now waits indefinitely for transcription completion by default (no more 60-second timeout)
- **Massive file support**: Increased video file size limit from 500MB to **10GB** (configurable)
- **Extended duration support**: Increased video duration limit from 60 minutes to **720 minutes (12 hours)**
- **Custom port configuration**: Added PORT environment variable support (default: 8509)

#### 🔧 Technical Enhancements
- **Synchronous API responses**: `/transcribe/url` endpoint now returns complete transcription results instead of job IDs
- **Environment-driven configuration**: All limits now configurable via `.env` file
- **Improved error handling**: Better handling of large files and long-duration content
- **Cache management**: Added bytecode cache clearing for consistent deployments

#### 📝 Documentation Updates
- **Added CLAUDE.md**: Comprehensive development guide for future maintenance
- **Updated README**: Detailed changelog and configuration documentation
- **Environment variables**: Complete list of all configurable options

#### 🐛 Bug Fixes
- Fixed timeout issues with long audio files (2+ hours)
- Resolved port binding conflicts with configurable PORT setting
- Improved service restart reliability with cache clearing

## Features

- 🎬 Accept any video URL for transcription
- 🎯 GPU acceleration with CUDA support for RTX 3090
- 🔄 Asynchronous processing with background tasks
- 📊 Real-time progress tracking
- 🔤 Language detection and selection
- ⏱ Word-level timestamps
- 📝 Markdown formatted output
- 🚀 High-performance batched processing
- 💾 JSON output with automatic saving

## Requirements

- Python 3.9+
- CUDA 12+ and cuDNN 9+ for GPU acceleration
- FFmpeg for audio extraction
- 8+ GB of GPU memory recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-transcriber.git
cd video-transcriber
```

2. Create and activate the virtual environment:
```bash
python -m venv vidtranscribeVenv
source vidtranscribeVenv/bin/activate  # On Windows: vidtranscribeVenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your machine-specific settings
nano .env

# Key variables to customize for your machine:
# - TEMP_DIR=/path/to/temp         # Where to store temporary files
# - OUTPUT_DIR=/path/to/output     # Where to save transcription JSONs
# - GPU_DEVICE_INDEX=0             # Which GPU to use (0, 1, 2, etc.)
# - CPU_THREADS=8                  # Match your CPU core count
# - MAX_VIDEO_SIZE_MB=20480        # Adjust based on available storage
```

See `ENV_VARIABLES_REFERENCE.md` for complete documentation of all 22 configurable variables.

## Environment Variables

All configuration is now handled through environment variables. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### Server Configuration
- `HOST` - Server host address (default: "0.0.0.0")
- `PORT` - Server port (default: 8509)
- `LOG_LEVEL` - Logging level: "debug", "info", "warning", "error" (default: "info")

### Model Configuration
- `MODEL_NAME` - Whisper model: "turbo", "large-v3", "large-v2", "medium", "small", "base", "tiny" (default: "turbo")
- `COMPUTE_TYPE` - Computation precision: "float16", "float32", "int8" (default: "float16")
- `GPU_DEVICE_INDEX` - GPU device index for multi-GPU systems (default: 0)
- `CPU_THREADS` - Number of CPU threads for processing (default: 4)

### Processing Limits
- `MAX_VIDEO_SIZE_MB` - Maximum video file size in MB (default: 10240 = 10GB)
- `MAX_VIDEO_DURATION_MINUTES` - Maximum video duration in minutes (default: 720 = 12 hours)
- `BATCH_SIZE` - Batch size for transcription processing (default: 8)
- `MAX_CONCURRENT_JOBS` - Maximum concurrent transcription jobs (default: 2)

### API Configuration
- `REQUEST_LIMIT_PER_MINUTE` - API rate limit requests per minute (default: 10)
- `RATE_LIMIT_WINDOW_SECONDS` - Rate limiting time window in seconds (default: 60)

### Audio Processing
- `AUDIO_SAMPLE_RATE` - Audio sample rate in Hz (default: 16000)
- `AUDIO_CODEC` - Audio codec for extraction (default: "pcm_s16le")
- `AUDIO_CHANNELS` - Number of audio channels: 1=mono, 2=stereo (default: 1)
- `DOWNLOAD_CHUNK_SIZE_MB` - Download chunk size in MB (default: 1)

### VAD (Voice Activity Detection)
- `VAD_MIN_SILENCE_MS` - Minimum silence duration in milliseconds to filter out (default: 500)

### Storage Paths
- `TEMP_DIR` - Directory for temporary files (default: "/tmp/vidTranscriber")
- `OUTPUT_DIR` - Directory for saving JSON transcription outputs (default: "/tmp/vidTranscriberOutput")

### System Settings
- `SHUTDOWN_TIMEOUT_SECONDS` - Graceful shutdown timeout in seconds (default: 10)
- `CUDA_VISIBLE_DEVICES` - CUDA device visibility (default: 0)

## Usage

1. Start the API server:
```bash
cd vidTranscriber
python run.py              # Production mode on port 8509
python run.py --reload     # Development mode with auto-reload
python run.py --port 8080  # Custom port
```

2. Access the API documentation at http://localhost:8509/docs

### API Endpoints

#### Submit URL for transcription
```http
POST /transcribe/url
```
Request body:
```json
{
  "video_url": "https://example.com/video.mp4",
  "language": "en",           # Optional
  "beam_size": 5,             # Optional
  "vad_filter": true,         # Optional
  "word_timestamps": true     # Optional
}
```

**New in v1.1.0**: By default, this endpoint now waits for transcription completion and returns the full result:

Response (completed transcription):
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "language": "en",
  "text": "Full transcription text here...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, welcome to this video...",
      "words": [...]
    }
  ],
  "markdown": "# Video Transcription\n\n**Detected Language:** en...",
  "json_path": "/tmp/vidTranscriberOutput/uuid_timestamp.json"
}
```

**For background processing**: Add query parameter `?wait_for_result=false` to get immediate job ID:
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "progress": 0.0
}
```

#### Check transcription status
```http
GET /transcribe/{job_id}
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "transcribing",
  "progress": 0.75,
  "output_file": "/path/to/transcription/results/job_id_20240101_120000.json"
}
```

#### Get transcription result
```http
GET /transcribe/{job_id}/result
```

Response includes the full transcription with timestamps, markdown formatting, and path to the JSON file.

#### Download JSON transcription
```http
GET /transcribe/{job_id}/json
```

Returns the raw JSON file with the complete transcription.

## Performance Optimization

This service uses:

- **BatchedInferencePipeline** from faster-whisper for batched processing, providing up to 4x speed improvement
- **float16** precision on GPU for optimal balance of accuracy and performance
- **int8_float16** quantization option for reduced memory usage
- **Voice Activity Detection (VAD)** to filter out non-speech segments
- **Concurrent processing** with configurable job limits

### Real-World Performance Metrics

Based on actual usage with GPU acceleration (RTX 3090):
- **12-hour audio file**: Processed in ~12 minutes (1 minute per hour of audio)
- **Transcription speed**: ~167 words per minute
- **Output size**: 120,000 words / 650,000 characters from 12-hour file
- **VAD efficiency**: Typically removes 10-15% of non-speech audio

### VAD (Voice Activity Detection) Explained

VAD automatically removes non-speech segments including:
- Silent pauses and gaps
- Background music without vocals
- Ambient noise and static
- Technical audio artifacts

For a 12-hour file, VAD might remove 1-2 hours of non-speech, significantly improving:
- Processing speed (less audio to transcribe)
- Accuracy (no phantom words from noise)
- Output quality (cleaner transcriptions)

## Large File Support

**New in v1.1.0**: Enhanced support for massive files:

- **File Size**: Up to 10GB video files (configurable via `MAX_VIDEO_SIZE_MB`)
- **Duration**: Up to 12-hour videos (configurable via `MAX_VIDEO_DURATION_MINUTES`)
- **No Timeout**: API waits indefinitely for transcription completion
- **Memory Management**: Automatic cleanup and CUDA cache management
- **Progress Tracking**: Real-time progress updates during processing

### Example: Processing Large Files
```bash
# Set custom limits for even larger files
echo "MAX_VIDEO_SIZE_MB=20480" >> .env      # 20GB
echo "MAX_VIDEO_DURATION_MINUTES=1440" >> .env  # 24 hours
python run.py
```

## Project Structure

### Directory Overview

```
vidTranscriber/
├── app/                    # Main application code
│   ├── api/
│   │   └── routes/         # API endpoints
│   │       └── transcription.py
│   ├── core/               # Core business logic
│   │   └── transcriber.py  # Main transcription service (29KB)
│   ├── models/             # Pydantic data models (NOT ML models!)
│   │   └── transcription.py # Request/response schemas
│   └── utils/              # Utility functions
│       └── rate_limiter.py
├── models/                 # Whisper AI model files (.pt files)
│   ├── large-v3-turbo.pt   # Turbo model weights
│   ├── large-v3.pt         # Large v3 model weights
│   └── medium.en.pt        # Medium English model weights
├── tests/                  # Test files
│   └── test_transcriber.py
├── tmp/                    # Temporary files directory
│   └── vidTranscriberOutput/ # JSON transcription outputs
├── vidTranscriber/         # ⚠️ UNUSED nested directory (safe to delete)
├── .env                    # Environment configuration
├── .env.example            # Environment template
├── ENV_VARIABLES_REFERENCE.md # Complete env var documentation
├── requirements.txt        # Python dependencies
├── run.py                  # Application entry point
└── setup.sh               # Initial setup script
```

### Important Directory Distinctions

#### `/models/` vs `/app/models/`
These directories serve completely different purposes:

| Directory | Purpose | Contents | File Types |
|-----------|---------|----------|------------|
| `/models/` | Whisper AI models | Neural network weights | `.pt` files (binary) |
| `/app/models/` | Data validation | Pydantic schemas | `.py` files (Python code) |

- **`/models/`**: Contains downloaded Whisper model files (large binary files)
- **`/app/models/`**: Contains Python classes for API request/response validation

This naming can be confusing but follows standard conventions where:
- ML projects use "models" for neural networks
- Web frameworks use "models" for data structures

### Git Configuration
The `.gitignore` file is configured to exclude machine-specific files:
- `vidTranscriberVenv/` - Virtual environment
- `tmp/vidTranscriberOutput/*` - Transcription outputs (keeps folder structure)
- `.env` - Environment variables (keeps `.env.example`)
- `__pycache__/` - Python cache files
- `.claude/` - Claude development files
- `models/*.pt` - Large model files (keeps folder structure)

### Cleanup Note
The nested `vidTranscriber/vidTranscriber/` directory is not used and can be safely removed:
```bash
rm -rf vidTranscriber/
```

## License

[MIT License](LICENSE) 