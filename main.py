import asyncio

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, deepgram, cartesia
from prosa.stt import STT as ProsaSTT
from prosa.tts import TTS as ProsaTTS
# from api import AssistantFnc
from function_context import AssistantFunc

load_dotenv()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "Anda adalah asisten suara yang dibuat oleh LiveKit. Antarmuka Anda dengan pengguna akan berupa suara."
            "Anda harus menggunakan respons yang singkat dan padat, dan menghindari penggunaan tanda baca yang tidak dapat diucapkan."
        ),
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFunc("data_pemesanan.csv")

    assitant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        stt=ProsaSTT(),
        llm=openai.LLM.with_groq(),
        tts=ProsaTTS(),
        # tts = cartesia.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    assitant.start(ctx.room)

    await asyncio.sleep(1)
    await assitant.say("Halo, Ada yang bisa saya bantu?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))