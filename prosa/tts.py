from __future__ import annotations

import os
from dataclasses import dataclass
from typing import AsyncContextManager

from livekit.agents import tts, utils

from .log import logger
from .models import ProsaTTSModels
from .prosa import Prosa

PROSA_TTS_SAMPLE_RATE = 24000
PROSA_TTS_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: ProsaTTSModels
    wait: bool
    pitch: float
    tempo: float
    audio_format : str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: ProsaTTSModels = "tts-1",
        wait: bool = True,
        pitch: float = 0,
        tempo: float = 1,
        audio_format : str = "mp3",
        api_key: str | None = None,
        client: Prosa.TTS | None = None,
    ) -> None:
        """
        Create a new instance of Prosa TTS.

        ``api_key`` must be set to your Prosa TTS API key, either using the argument or by setting the
        ``PROSA_TTS_API_KEY`` environmental variable.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=PROSA_TTS_SAMPLE_RATE,
            num_channels=PROSA_TTS_CHANNELS,
        )

        # throw an error on our end
        self._api_key = api_key or os.environ.get("PROSA_TTS_API_KEY")
        if api_key is None:
            raise ValueError("Prosa TTS API key is required")

        self._opts = _TTSOptions(
            model=model,
            wait=wait,
            pitch=pitch,
            tempo=tempo,
            audio_format=audio_format,
        )

        self._client = client or Prosa.TTS(self._api_key)

    def synthesize(self, text: str) -> "ChunkedStream":
        b64audio_data = self._client.get_speech(
            text=text,
            audio_format=self._opts.audio_format,
            model=self._opts.model,
            wait=self._opts.wait,
            )

        return ChunkedStream(b64audio_data, self._opts)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        audio_data: str,
        opts: _TTSOptions,
    ) -> None:
        super().__init__()
        self._opts, self._audio_data = opts, audio_data

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        decoder = utils.codecs.Mp3StreamDecoder()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=PROSA_TTS_SAMPLE_RATE,
            num_channels=PROSA_TTS_CHANNELS,
        )

        for frame in decoder.decode_chunk(self._audio_data):
                for frame in audio_bstream.write(frame.data):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id, segment_id=segment_id, frame=frame
                        )
                    )

        for frame in audio_bstream.flush():
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id, segment_id=segment_id, frame=frame
                )
            )