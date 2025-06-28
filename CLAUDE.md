# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
- Initial setup: `bash setup.sh` (creates venv, installs dependencies, checks GPU)
- Virtual environment: `source vidtranscribeVenv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Running the Application
- Development server: `python run.py --reload`
- Production server: `python run.py`
- With custom host/port: `python run.py --host 0.0.0.0 --port 8080`
- API documentation: Access http://localhost:8000/docs when server is running

### Testing
- Run tests: `python -m pytest tests/` (from project root)
- Single test file: `python -m pytest tests/test_transcriber.py`

### GPU Setup Check
- Verify GPU/CUDA: `python check_gpu_setup.py`

## Application Architecture

### Core Components
- **TranscriptionService** (`app/core/transcriber.py`): Main service handling video download, audio extraction, and transcription using faster-whisper with GPU acceleration
- **FastAPI Application** (`app/main.py`): Web API with CORS, rate limiting, and lifespan management
- **API Routes** (`app/api/routes/transcription.py`): REST endpoints for transcription jobs
- **Data Models** (`app/models/transcription.py`): Pydantic models for requests/responses

### Key Features
- GPU-accelerated transcription with CUDA support and CPU fallback
- BatchedInferencePipeline for 4x performance improvement
- Asynchronous processing with background tasks and job queuing
- Rate limiting and graceful shutdown handling
- Automatic model downloading (faster-whisper models)
- JSON output with automatic file saving to `/tmp/vidTranscriberOutput/`

### Model Configuration
- Default model: "turbo" (maps to large-v3)
- Model mapping defined in `MODEL_NAME_MAPPING` in transcriber.py
- Models stored in `models/` directory (large-v3-turbo.pt, large-v3.pt, medium.en.pt)

### Environment Variables
Key variables (no .env file present):
- `MODEL_NAME`: Model to use (default: "turbo")
- `CUDA_VISIBLE_DEVICES`: GPU device index (default: 0)
- `COMPUTE_TYPE`: Computation precision (default: "float16")
- `BATCH_SIZE`: Batch size for processing (default: 8)
- `MAX_CONCURRENT_JOBS`: Concurrent job limit (default: 2)
- `TEMP_DIR`: Temporary files directory (default: "/tmp/vidTranscriber")
- `OUTPUT_DIR`: JSON output directory (default: "/tmp/vidTranscriberOutput")

### API Endpoints
- `POST /transcribe/url`: Submit video URL for transcription
- `GET /transcribe/{job_id}`: Get job status and progress
- `GET /transcribe/{job_id}/result`: Get completed transcription result
- `GET /transcribe/{job_id}/json`: Download raw JSON file
- `GET /health`: Health check endpoint

### Job Processing Flow
1. **Download**: Video downloaded from URL with progress tracking
2. **Extract**: Audio extracted using ffmpeg (16kHz mono PCM)
3. **Transcribe**: Audio processed with faster-whisper BatchedInferencePipeline
4. **Format**: Results formatted as markdown and saved as JSON
5. **Cleanup**: Temporary files automatically removed

### Signal Handling
- Graceful shutdown on SIGTERM/SIGINT
- Active job cleanup and temp file removal
- CUDA memory cache clearing on shutdown

### Error Handling
- CUDA/GPU error fallback to CPU processing
- Comprehensive validation for video URLs (blocks private IPs)
- Rate limiting with configurable limits
- Detailed error responses with appropriate HTTP status codes