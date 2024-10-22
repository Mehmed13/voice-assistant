import asyncio

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, deepgram, cartesia, elevenlabs
from prosa.stt import STT as ProsaSTT
from prosa.tts import TTS as ProsaTTS
# from api import AssistantFnc
from function_context import AssistantFunc

load_dotenv()


async def entrypoint(ctx: JobContext):
    # Setup
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "Anda adalah asisten suara yang dibuat oleh LiveKit. Antarmuka Anda dengan pengguna akan berupa suara."
            "Anda harus menggunakan respons yang singkat dan padat, dan menghindari penggunaan tanda baca yang tidak dapat diucapkan."
            "Berikanlah jawaban semuanya dalam bahasa indonesia"
        ),
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = AssistantFunc("data_pemesanan.csv")

    VOICE = elevenlabs.Voice(
        id="eHmtqZsJqqzY4Q3V8ku4",
        name="Alena",
        category="cloned",
        settings=elevenlabs.VoiceSettings(
            stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
        ),
    )

    assitant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(language="id"),
        # stt=ProsaSTT(),
        llm=openai.LLM.with_groq(),
        tts=elevenlabs.TTS(voice=VOICE), 
        # tts=ProsaTTS(),
        # tts = cartesia.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    assitant.start(ctx.room)

    await asyncio.sleep(1)
    await assitant.say("Saya adalah Alena, voice assistant yang dapat membantu anda mengecek pesanan, silahkan sebutkan id atau nama barang yang ingin kamu ketahui", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))