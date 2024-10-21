from __future__ import annotations

import dataclasses
import io
import os
import wave
from dataclasses import dataclass
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer, merge_frames

from .log import logger
from .models import ProsaLanguages, ProsaSTTModels
from .prosa import Prosa

BASE_URL = "https://api.prosa.ai/v2/speech/stt"


@dataclass
class STTOptions:
    language: ProsaLanguages | str | None
    model: ProsaSTTModels
    wait: bool
    speaker_count: int
    include_filler: bool
    include_partial_results: bool
    auto_punctuation: bool
    enable_spoken_numerals: bool
    enable_speech_insights: bool
    enable_voice_insights: bool
    
class STT(stt.STT):
    def __init__(
        self,
        *,
        model: ProsaSTTModels = "stt-general",
        language: ProsaLanguages = "id-ID",
        wait: bool = True,
        speaker_count: int = 1,
        include_filler: bool = False,
        include_partial_results: bool= False,
        auto_punctuation: bool = False,
        enable_spoken_numerals: bool = False,
        enable_speech_insights: bool = False,
        enable_voice_insights: bool = False,     
        api_key: str | None = None,
        client: Prosa.STT | None = None,
    ) -> None:
        """
        Create a new instance of Prosa STT.

        ``api_key`` must be set to your Prosa API key, either using the argument or by setting
        the ``Prosa_STT_API_KEY`` environmental variable.
        """

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=include_partial_results
            )
        )

        self._api_key = api_key or os.environ.get("Prosa_STT_API_KEY")
        if self._api_key is None:
            raise ValueError("Prosa API key is required")

        if language not in ("id-ID", "id"):
            logger.warning(
                f"{model} does not support language {language}, falling back to stt-general"
            )
            model = "stt-general"

        self._opts = STTOptions(
            language=language,
            model=model,
            wait=wait,
            speaker_count=speaker_count,
            include_filler=include_filler,
            include_partial_results=include_partial_results,
            auto_punctuation=auto_punctuation,
            enable_spoken_numerals=enable_spoken_numerals,
            enable_speech_insights=enable_speech_insights,
            enable_voice_insights=enable_voice_insights,
        )

        self._client = client or Prosa.STT(self._api_key)

    async def recognize(
        self, buffer: AudioBuffer, *, language: ProsaLanguages | str | None = None
    ) -> stt.SpeechEvent:

        buffer = merge_frames(buffer)
        io_buffer = io.BytesIO()
        with wave.open(io_buffer, "wb") as wav:
            wav.setnchannels(buffer.num_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(buffer.sample_rate)
            wav.writeframes(buffer.data)

        bytes_data = io_buffer.getvalue()

        with open("prosa/output.wav", "wb") as wav_file:
            wav_file.write(bytes_data)

        print("start transcription...")
        transcription = self._client.create_transcription(
            filename="prosa/output.wav",
            model=self._opts.model,
            wait=self._opts.wait
        )
        print("finish transcription...")

        try:
            transcription_text = transcription["data"][0]["transcript"]
        except Exception as e:
            print("Error:",  e)
            transcription_text = None

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=transcription_text or "", language=language or "")
            ]
        )


    def _sanitize_options(self, *, language: str | None = None) -> STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        return config
