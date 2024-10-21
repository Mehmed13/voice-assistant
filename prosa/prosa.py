import base64
import requests
# from pydub import AudioSegment
# import io

class Prosa:
    _STT_URL = "https://api.prosa.ai/v2/speech/stt"
    _TTS_URL = "https://api.prosa.ai/v2/speech/tts"

    class STT:
        def __init__(self, stt_api_key):
            self.stt_api_key = stt_api_key

        def _submit_stt_request(self, filename: str, model: str, wait : bool) -> dict:
            with open(filename, "rb") as f:
                b64audio_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "config": {
                    "model": model,
                    "wait": wait  # Blocks the request until the execution is finished
                },
                "request": {
                    "data": b64audio_data
                }
            }

            response = requests.post(Prosa._STT_URL, json=payload, headers={
                "x-api-key": self.stt_api_key
            })

            return response.json()

        def create_transcription(self, filename: str,  model: str="stt-general", wait=True) -> dict:
            job = self._submit_stt_request(filename, model=model, wait=wait)
            if job["status"] == "complete":
                return job["result"]
    class TTS:
        def __init__(self, tts_api_key):
            self.tts_api_key = tts_api_key

        def _submit_tts_request(self, text: str, audio_format: str, model: str, wait: bool) -> dict:
            payload = {
                "config": {
                    "model": model,
                    "wait": wait,  # Blocks the request until the execution is finished
                    "audio_format": audio_format
                },
                "request": {
                    "text": text
                }
            }

            response = requests.post(Prosa._TTS_URL, json=payload, headers={
                "x-api-key": self.tts_api_key
            })

            return response.json()
        
        def get_speech(self, text: str, audio_format: str = "mp3", model: str="tts-dimas-formal", wait: bool=True) -> bytes:
            job = self._submit_tts_request(text, audio_format, model=model, wait=wait)
            # print(job)
            if job["status"] == "complete":
                b64audio_data = base64.b64decode(job["result"]["data"])
                # audio_segment = AudioSegment.from_file(io.BytesIO(b64audio_data), format="mp3")
                # audio_segment.export("prosa/speech.mp3", format="mp3")
                # with open("prosa/speech.mp3", "wb") as f:
                #     f.write(b64audio_data)
                # print("type dari audio data", type(b64audio_data))
                # print("audio data:", b64audio_data)
                return b64audio_data        