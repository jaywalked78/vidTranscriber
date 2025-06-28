from unittest.mock import patch, MagicMock
from app.core.transcriber import TranscriptionService

class TestTranscriptionService:
    @patch("app.core.transcriber.BatchedInferencePipeline", autospec=True)
    @patch("app.core.transcriber.WhisperModel", autospec=True)
    @patch("app.core.transcriber.Storage")
    async def test_transcribe_audio(self, mock_storage, mock_model, mock_batched):
        # Setup
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_batched_instance = MagicMock()
        mock_batched.return_value = mock_batched_instance
        
        # Create segments and info
        segment1 = MagicMock()
        segment1.start = 0.0
        segment1.end = 1.0
        segment1.text = "Hello"
        segment1.words = [MagicMock(start=0.0, end=0.5, word="He"), MagicMock(start=0.5, end=1.0, word="llo")]
        
        segment2 = MagicMock()
        segment2.start = 1.0
        segment2.end = 2.0
        segment2.text = "World"
        segment2.words = [MagicMock(start=1.0, end=1.5, word="Wo"), MagicMock(start=1.5, end=2.0, word="rld")]
        
        mock_segments = [segment1, segment2]
        mock_info = MagicMock(language="en", language_probability=0.99)
        
        # Mock the transcribe method to return segments and info
        mock_batched_instance.transcribe.return_value = (mock_segments, mock_info)
        
        # Create instance with mocked properties
        transcriber = TranscriptionService()
        transcriber.model = mock_model_instance
        transcriber.batched_model = mock_batched_instance
        transcriber.device = "cpu"
        transcriber.batch_size = 16
        
        transcriber.update_job = MagicMock()
        job_id = "test_job"
        audio_path = "test_audio.wav"
        options = {
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "language": "en"
        }
        
        # Test
        result = await transcriber._transcribe_audio(job_id, audio_path, options)
        
        # Validate that the transcribe method was called correctly
        mock_batched_instance.transcribe.assert_called_once_with(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            batch_size=16
        ) 