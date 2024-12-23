import sounddevice as sd
import whisper
import numpy as np


class SpeechToTextService:
    def __init__(self, model="base.en"):
        """
        Initializes the SpeechToTextService class with the Whisper model.

        Args:
            model (str, optional): The model to use for speech-to-text. Defaults to "base.en".
        """
        self.model = whisper.load_model(model)

    def record_audio(self, duration: int, samplerate: int = 16000):
        """
        Records audio from the microphone.

        Args:
            duration (int): The duration of the recording in seconds.
            samplerate (int, optional): The sample rate of the recording. Defaults to 16000.

        Returns:
            np.ndarray: The recorded audio data.
        """
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
        return audio_data

    def transcribe(self, audio_data):
        """
        Transcribes the recorded audio to text.

        Args:
            audio_data (np.ndarray): The recorded audio data.

        Returns:
            str: The transcribed text.
        """
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.model.transcribe(audio_np)
        return result["text"]
