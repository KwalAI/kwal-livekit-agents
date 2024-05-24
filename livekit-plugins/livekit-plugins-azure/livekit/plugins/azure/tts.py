from dataclasses import dataclass
from typing import AsyncIterable, Union, List
import asyncio
import contextlib
import os

from livekit import rtc
from livekit.agents import aio, tts

import azure.cognitiveservices.speech as speechsdk

from .log import logger
from .models import AudioEncoding, Gender, SpeechLanguages

LgType = Union[SpeechLanguages, str]
GenderType = Union[Gender, str]
AudioEncodingType = Union[AudioEncoding, str]


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: float | None = None  # [0.0 - 1.0]
    use_speaker_boost: bool | None = False


@dataclass
class Voice:
    id: str
    name: str
    category: str
    settings: VoiceSettings | None = None


@dataclass
class TTSOptions:
    api_key: str
    voice: Voice
    base_url: str
    sample_rate: int
    latency: int


class TTS(tts.TTS):
    def __init__(
        self,
        config = None,
        *,
        language: LgType = "en",
        gender: GenderType = "neutral",
        voice_name: str = "en-US-AriaNeural",  # Not required
        audio_encoding: AudioEncodingType = "wav",
        sample_rate: int = 16000,
        speaking_rate: float = 1.0,
    ) -> None:
        super().__init__(
            streaming_supported=True, sample_rate=sample_rate, num_channels=1
        )

        self._opts = config
        self._config = config

        self.azure_speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
        self.azure_speech_config.speech_synthesis_voice_name = voice_name
        self.azure_speech_config.prosody_rate = speaking_rate
        self.pull_stream = speechsdk.audio.PullAudioOutputStream()
        self.azure_audio_config = speechsdk.audio.AudioOutputConfig(stream=self.pull_stream)
        self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.azure_speech_config, audio_config=self.azure_audio_config)

    async def synthesize(
        self,
        *,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        try:
            # Perform the text-to-speech request on the text input with the selected
            result = self.synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_buffer = bytes(32000)  # Buffer size of 32 KB
                while True:
                    filled_size = self.pull_stream.read(audio_buffer)
                    if filled_size == 0:
                        break
                    data = audio_buffer[:filled_size]
                    yield tts.SynthesizedAudio(
                        text=text,
                        data=rtc.AudioFrame(
                            data=data,
                            sample_rate=self._config.audio_config.sample_rate_hertz,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,  # 16-bit
                        )
                    )

        except Exception as e:
            logger.error(f"failed to synthesize: {e}")

    def stream(
        self,
    ) -> "SynthesizeStream":
        return SynthesizeStream(self._config, self.synthesizer)


class SynthesizeStream(tts.SynthesizeStream):
    _STREAM_EOS = ""

    def __init__(
        self,
        opts: TTSOptions,
        synthetizer: speechsdk.SpeechSynthesizer,
        max_retry: int = 32,
    ):
        self._opts = opts

        self._queue = asyncio.Queue[str | None]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent | None]()
        self._closed = False
        self._text = ""

        self.synthesizer = synthetizer

        self._main_task = asyncio.create_task(self._run(max_retry))

    def push_text(self, token: str | None) -> None:
        """
        Push some text to internal queue to be consumed by _run method.
        """
        if self._closed:
            raise ValueError("cannot push to a closed stream")

        if token is None:
            self._flush_if_needed()
            return

        if len(token) == 0:
            # 11labs marks the EOS with an empty string, avoid users from pushing empty strings
            return

        # TODO: Naive word boundary detection may not be good enough for all languages
        # fmt: off
        splitters = (".", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}")
        # fmt: on

        self._text += token

        while True:
            last_split = -1
            for i, c in enumerate(self._text):
                if c in splitters:
                    last_split = i
                    break

            if last_split == -1:
                break

            seg = self._text[: last_split + 1]
            seg = seg.strip() + " "
            self._queue.put_nowait(seg)
            self._text = self._text[last_split + 1 :]

    async def aclose(self, *, wait: bool = True) -> None:
        self._flush_if_needed()
        self._queue.put_nowait(None)
        self._closed = True

        if not wait:
            self._main_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    def _flush_if_needed(self) -> None:
        seg = self._text.strip()
        if len(seg) > 0:
            self._queue.put_nowait(seg + " ")

        self._text = ""
        self._queue.put_nowait(SynthesizeStream._STREAM_EOS)

    async def _run(self, max_retry: int) -> None:
        retry_count = 0
        tts_task: asyncio.Task | None = None
        data_tx: aio.ChanSender[str] | None = None
        segment = ""

        try:
            while True:
                try:
                    data = await self._queue.get()

                    if data is None:
                        break

                    if data == SynthesizeStream._STREAM_EOS:
                        data_tx, data_rx = aio.channel()
                        tts_task = asyncio.create_task(self._run_tts(data_rx))

                        assert data_tx is not None
                        assert tts_task is not None

                        data_tx.send_nowait(segment) # Correctly sending this data
                    else:
                        segment += data

                except Exception:
                    if retry_count >= max_retry:
                        logger.exception(
                            f"failed to connect to GTTS after {max_retry} retries"
                        )
                        break

                    retry_delay = min(retry_count * 5, 5)  # max 5s
                    retry_count += 1

                    logger.warning(
                        f"failed to connect to Azure TTS, retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)

        except Exception:
            logger.exception("Azure TTS task failed")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                if tts_task is not None:
                    tts_task.cancel()
                    await tts_task

            self._event_queue.put_nowait(None)

    async def _run_tts(
        self, data_rx: aio.ChanReceiver[str] = None
    ) -> None:

        self._event_queue.put_nowait(
            tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
        )

        async def send_task():
            """Backupp task in case coming back to word by word streaming is needed"""
            while True:
                text_data = await data_rx.recv()

                result = self.synthesizer.speak_text_async(text_data).get()

                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        self._event_queue.put_nowait(
                            tts.SynthesisEvent(
                                type=tts.SynthesisEventType.AUDIO,
                                audio=tts.SynthesizedAudio(
                                    text="",
                                    data=rtc.AudioFrame(
                                        data=result.audio_data,
                                        sample_rate=16000,
                                        num_channels=1,
                                        samples_per_channel=len(result.audio_data) // 2,  # 16-bit
                                    )
                                ),
                            )
                        )
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = result.cancellation_details
                    print(f"Synthesis canceled: {cancellation_details.reason}")
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        print(f"Error details: {cancellation_details.error_details}")
        try:
            await asyncio.gather(send_task())
        except Exception:
            logger.exception("Azure TTS api connection failed")
        finally:
            self._event_queue.put_nowait(
                tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
            )

    async def __anext__(self) -> tts.SynthesisEvent:
        evt = await self._event_queue.get()
        if evt is None:
            raise StopAsyncIteration

        return evt


def dict_to_voices_list(data: dict) -> List[Voice]:
    voices = []
    for voice in data["voices"]:
        voices.append(
            Voice(
                id=voice["voice_id"],
                name=voice["name"],
                category=voice["category"],
                settings=None,
            )
        )
    return voices
