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


    @staticmethod
    def create_azure_client(
        *,
        model: TTSModels = "tts-1",
        voice: TTSVoices = "alloy",
        speed: float = 1.0,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
    ) -> TTS:
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """

        azure_client = openai.AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
        )  # type: ignore

        return TTS(model=model, voice=voice, speed=speed, client=azure_client)

    def synthesize(self, text: str) -> "ChunkedStream":
        stream = self._client.audio.speech.with_streaming_response.create(
            input=text,
            model=self._opts.model,
            voice=self._opts.voice,
            response_format="mp3",
            speed=self._opts.speed,
        )

        return ChunkedStream(stream, text, self._opts)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        oai_stream: AsyncContextManager[openai.AsyncAPIResponse[bytes]],
        text: str,
        opts: _TTSOptions,
    ) -> None:
        super().__init__()
        self._opts, self._text = opts, text
        self._oai_stream = oai_stream

    @utils.log_exceptions(logger=logger)
    async def _main_task(self):
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        decoder = utils.codecs.Mp3StreamDecoder()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=PROSA_TTS_SAMPLE_RATE,
            num_channels=PROSA_TTS_CHANNELS,
        )

        async with self._oai_stream as stream:
            async for data in stream.iter_bytes():
                for frame in decoder.decode_chunk(data):
                    for frame in audio_bstream.write(frame.data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=segment_id,
                                frame=frame,
                            )
                        )

            for frame in audio_bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id, segment_id=segment_id, frame=frame
                    )
                )
