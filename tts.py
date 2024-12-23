from gtts import gTTS
import tempfile
import numpy as np
import soundfile as sf
import os
class TextToSpeechService:
    def synthesize(self, text: str):
        """
        Synthesize text into speech using Google TTS.

        Args:
            text (str): Text to convert to speech.

        Returns:
            tuple: Sample rate and audio array.
        """
        # Đảm bảo text là chuỗi
        if not isinstance(text, str):
            text = str(text)

        tts = gTTS(text=text.strip(), lang="en")  # Loại bỏ khoảng trắng thừa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            # Đọc dữ liệu âm thanh
            data, sample_rate = sf.read(temp_file.name)

        # Xóa tệp sau khi sử dụng
        os.unlink(temp_file.name)
        return sample_rate, np.array(data)
