# Environment Variables Reference Documentation

This document provides a comprehensive reference of all environment variables used in the Video Transcriber API codebase, including their locations, usage, and default values.

Generated: 2025-06-26

## 📊 Summary

- **Total Environment Variables**: 22
- **Files Using Environment Variables**: 3 (`app/core/transcriber.py`, `app/main.py`, `run.py`)
- **All Variables Defined in .env**: ✅ Yes

## 🔍 Environment Variables by Category

### 1️⃣ Server Configuration

#### **HOST**
- **Default**: `"0.0.0.0"`
- **Usage**: Server host address
- **References**:
  - `app/main.py:99` - `host=os.getenv("HOST", "0.0.0.0")`
  - `run.py:14` - `default=os.getenv("HOST", "0.0.0.0")`

#### **PORT**
- **Default**: `8000` (overridden to `8509` in .env)
- **Usage**: Server port number
- **References**:
  - `app/main.py:100` - `port=int(os.getenv("PORT", 8000))`
  - `run.py:20` - `default=int(os.getenv("PORT", 8000))`

#### **LOG_LEVEL**
- **Default**: `"info"`
- **Usage**: Logging verbosity level
- **References**:
  - `app/main.py:102` - `log_level=os.getenv("LOG_LEVEL", "info")`

### 2️⃣ Model Configuration

#### **MODEL_NAME**
- **Default**: `"turbo"`
- **Usage**: Whisper model selection
- **References**:
  - `app/core/transcriber.py:76` - `model_name = os.getenv("MODEL_NAME", "turbo")`
  - `run.py:40` - `print(f"Model: {os.getenv('MODEL_NAME', 'turbo')}")`

#### **COMPUTE_TYPE**
- **Default**: `"float16"`
- **Usage**: Computation precision for model
- **References**:
  - `app/core/transcriber.py:80` - `self.compute_type = os.getenv("COMPUTE_TYPE", "float16")`
  - `run.py:39` - `print(f"Compute Type: {os.getenv('COMPUTE_TYPE', 'float16')}")`

#### **GPU_DEVICE_INDEX**
- **Default**: `0`
- **Usage**: GPU device selection for multi-GPU systems
- **References**:
  - `app/core/transcriber.py:111` - `self.gpu_device_index = int(os.getenv("GPU_DEVICE_INDEX", 0))`

#### **CPU_THREADS**
- **Default**: `4`
- **Usage**: Number of CPU threads for processing
- **References**:
  - `app/core/transcriber.py:112` - `self.cpu_threads = int(os.getenv("CPU_THREADS", 4))`

### 3️⃣ Processing Limits

#### **MAX_VIDEO_SIZE_MB**
- **Default**: `10240` (10GB)
- **Usage**: Maximum allowed video file size
- **References**:
  - `app/core/transcriber.py:98` - `self.max_video_size_mb = int(os.getenv("MAX_VIDEO_SIZE_MB", 10240))`

#### **MAX_VIDEO_DURATION_MINUTES**
- **Default**: `720` (12 hours)
- **Usage**: Maximum allowed video duration
- **References**:
  - `app/core/transcriber.py:99` - `self.max_video_duration_minutes = int(os.getenv("MAX_VIDEO_DURATION_MINUTES", 720))`

#### **MAX_CONCURRENT_JOBS**
- **Default**: `2`
- **Usage**: Maximum number of concurrent transcription jobs
- **References**:
  - `app/core/transcriber.py:97` - `self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", 2))`

#### **BATCH_SIZE**
- **Default**: `8`
- **Usage**: Batch size for transcription processing
- **References**:
  - `app/core/transcriber.py:94` - `self.batch_size = int(os.getenv("BATCH_SIZE", "8"))`

### 4️⃣ API Configuration

#### **REQUEST_LIMIT_PER_MINUTE**
- **Default**: `10`
- **Usage**: Rate limit for API requests per minute
- **References**:
  - `app/main.py:60` - `limit=int(os.getenv("REQUEST_LIMIT_PER_MINUTE", 10))`

#### **RATE_LIMIT_WINDOW_SECONDS**
- **Default**: `60`
- **Usage**: Time window for rate limiting
- **References**:
  - `app/main.py:61` - `window=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))`

### 5️⃣ Audio Processing

#### **AUDIO_SAMPLE_RATE**
- **Default**: `16000` (16kHz)
- **Usage**: Audio sample rate for extraction
- **References**:
  - `app/core/transcriber.py:102` - `self.audio_sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))`

#### **AUDIO_CODEC**
- **Default**: `"pcm_s16le"`
- **Usage**: Audio codec for ffmpeg extraction
- **References**:
  - `app/core/transcriber.py:103` - `self.audio_codec = os.getenv("AUDIO_CODEC", "pcm_s16le")`

#### **AUDIO_CHANNELS**
- **Default**: `1` (mono)
- **Usage**: Number of audio channels
- **References**:
  - `app/core/transcriber.py:104` - `self.audio_channels = int(os.getenv("AUDIO_CHANNELS", 1))`

#### **DOWNLOAD_CHUNK_SIZE_MB**
- **Default**: `1` (1MB)
- **Usage**: Chunk size for downloading videos
- **References**:
  - `app/core/transcriber.py:105` - `self.download_chunk_size = int(os.getenv("DOWNLOAD_CHUNK_SIZE_MB", 1)) * 1024 * 1024`

### 6️⃣ VAD Settings

#### **VAD_MIN_SILENCE_MS**
- **Default**: `500` (500ms)
- **Usage**: Minimum silence duration for VAD filter
- **References**:
  - `app/core/transcriber.py:108` - `self.vad_min_silence_ms = int(os.getenv("VAD_MIN_SILENCE_MS", 500))`

### 7️⃣ Storage Paths

#### **TEMP_DIR**
- **Default**: `"/tmp/vidTranscriber"`
- **Usage**: Directory for temporary files
- **References**:
  - `app/core/transcriber.py:95` - `self.temp_dir = os.getenv("TEMP_DIR", "/tmp/vidTranscriber")`

#### **OUTPUT_DIR**
- **Default**: `"/tmp/vidTranscriberOutput"`
- **Usage**: Directory for JSON output files
- **References**:
  - `app/core/transcriber.py:96` - `self.output_dir = os.getenv("OUTPUT_DIR", "/tmp/vidTranscriberOutput")`

### 8️⃣ System Settings

#### **SHUTDOWN_TIMEOUT_SECONDS**
- **Default**: `10`
- **Usage**: Graceful shutdown timeout
- **References**:
  - `app/core/transcriber.py:115` - `self.shutdown_timeout_seconds = int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", 10))`

#### **CUDA_VISIBLE_DEVICES**
- **Default**: `"0"`
- **Usage**: CUDA device visibility (standard CUDA environment variable)
- **References**:
  - `run.py:38` - `print(f"GPU Device: {os.getenv('CUDA_VISIBLE_DEVICES', '0')}")`
  - **Note**: This is used by CUDA/PyTorch directly, not read by our code

## 📋 Usage Verification

### ✅ All Environment Variables Are Used

Every variable defined in the `.env` file is actively used in the codebase:

| Variable | Defined in .env | Used in Code |
|----------|----------------|--------------|
| HOST | ✅ | ✅ |
| PORT | ✅ | ✅ |
| LOG_LEVEL | ✅ | ✅ |
| MODEL_NAME | ✅ | ✅ |
| COMPUTE_TYPE | ✅ | ✅ |
| GPU_DEVICE_INDEX | ✅ | ✅ |
| CPU_THREADS | ✅ | ✅ |
| MAX_VIDEO_SIZE_MB | ✅ | ✅ |
| MAX_VIDEO_DURATION_MINUTES | ✅ | ✅ |
| MAX_CONCURRENT_JOBS | ✅ | ✅ |
| BATCH_SIZE | ✅ | ✅ |
| REQUEST_LIMIT_PER_MINUTE | ✅ | ✅ |
| RATE_LIMIT_WINDOW_SECONDS | ✅ | ✅ |
| AUDIO_SAMPLE_RATE | ✅ | ✅ |
| AUDIO_CODEC | ✅ | ✅ |
| AUDIO_CHANNELS | ✅ | ✅ |
| DOWNLOAD_CHUNK_SIZE_MB | ✅ | ✅ |
| VAD_MIN_SILENCE_MS | ✅ | ✅ |
| TEMP_DIR | ✅ | ✅ |
| OUTPUT_DIR | ✅ | ✅ |
| SHUTDOWN_TIMEOUT_SECONDS | ✅ | ✅ |
| CUDA_VISIBLE_DEVICES | ✅ | ✅ |

## 🔧 How Environment Variables Are Loaded

1. **dotenv Loading**: Both `run.py` and `app/main.py` call `load_dotenv()` to load the `.env` file
2. **Import Order**: The transcriber service is instantiated as a global variable when the module is imported
3. **Override Priority**: Environment variables > .env file > hardcoded defaults

## 💡 Usage Examples

### Setting Custom Values

```bash
# Option 1: Using .env file (recommended)
echo "TEMP_DIR=/custom/temp/path" >> .env
echo "GPU_DEVICE_INDEX=1" >> .env

# Option 2: Environment variables (override .env)
export MAX_VIDEO_SIZE_MB=20480
export CPU_THREADS=8
python run.py

# Option 3: Both (environment variables win)
# .env has PORT=8509
PORT=9000 python run.py  # Will use port 9000
```

### Machine-Specific Configuration

For deployment on different machines, modify these key variables:
- `TEMP_DIR` and `OUTPUT_DIR` - File system paths
- `GPU_DEVICE_INDEX` - For multi-GPU systems
- `CPU_THREADS` - Match your CPU core count
- `HOST` and `PORT` - Network configuration
- Storage limits based on available disk space