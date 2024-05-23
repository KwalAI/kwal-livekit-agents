from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import AsyncIterable, Optional, Union, List
import asyncio
import contextlib

from livekit import rtc
from livekit.agents import codecs, aio, tts

from google.cloud import texttospeech
from google.cloud.texttospeech_v1.types import (
    SsmlVoiceGender,
    SynthesizeSpeechResponse,
)

from gtts import gTTS

from .log import logger
from .models import AudioEncoding, Gender, SpeechLanguages

LgType = Union[SpeechLanguages, str]
GenderType = Union[Gender, str]
AudioEncodingType = Union[AudioEncoding, str]


@dataclass
class _TTSOptions:
    voice: texttospeech.VoiceSelectionParams
    audio_config: texttospeech.AudioConfig


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


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        language: LgType = "en",
        gender: GenderType = "neutral",
        voice_name: str = "",  # Not required
        encoding: AudioEncodingType = "linear16",
        sample_rate: int = 24000,
        speaking_rate: float = 1.0,
        credentials_info: dict | None = None,
        credentials_file: str | None = None,
    ) -> None:
        """
        if no credentials is provided, it will use the credentials on the environment
        GOOGLE_APPLICATION_CREDENTIALS (default behavior of Google TextToSpeechAsyncClient)
        """
        super().__init__(
            streaming_supported=True, sample_rate=sample_rate, num_channels=1
        )

        self._client: texttospeech.TextToSpeechAsyncClient | None = None
        self._credentials_info = credentials_info
        self._credentials_file = credentials_file

        ssml_gender = SsmlVoiceGender.NEUTRAL
        if gender == "male":
            ssml_gender = SsmlVoiceGender.MALE
        elif gender == "female":
            ssml_gender = SsmlVoiceGender.FEMALE

        voice = texttospeech.VoiceSelectionParams(
            name=voice_name,
            language_code=language,
            ssml_gender=ssml_gender,
        )

        if encoding == "linear16" or encoding == "wav":
            _audio_encoding = texttospeech.AudioEncoding.LINEAR16
        elif encoding == "mp3":
            _audio_encoding = texttospeech.AudioEncoding.MP3

        else:
            raise NotImplementedError(f"audio encoding {encoding} is not supported")

        self._opts = _TTSOptions(
            voice=voice,
            audio_config=texttospeech.AudioConfig(
                audio_encoding=_audio_encoding,
                sample_rate_hertz=sample_rate,
                speaking_rate=speaking_rate,
            ),
        )

    def _ensure_client(self) -> texttospeech.TextToSpeechAsyncClient:
        if not self._client:
            if self._credentials_info:
                self._client = (
                    texttospeech.TextToSpeechAsyncClient.from_service_account_info(
                        self._credentials_info
                    )
                )

            elif self._credentials_file:
                self._client = (
                    texttospeech.TextToSpeechAsyncClient.from_service_account_file(
                        self._credentials_file
                    )
                )
            else:
                self._client = texttospeech.TextToSpeechAsyncClient()

        assert self._client is not None
        return self._client

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_client())


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, text: str, opts: _TTSOptions, client: texttospeech.TextToSpeechAsyncClient
    ) -> None:
        self._text = text
        self._opts = opts
        self._client = client
        self._main_task: asyncio.Task | None = None
        self._queue = asyncio.Queue[Optional[tts.SynthesizedAudio]]()

    async def _run(self) -> None:
        try:
            response: SynthesizeSpeechResponse = await self._client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=self._text),
                voice=self._opts.voice,
                audio_config=self._opts.audio_config,
            )

            data = response.audio_content
            if self._opts.audio_config.audio_encoding == "mp3":
                decoder = codecs.Mp3StreamDecoder()
                frames = decoder.decode_chunk(data)
                for frame in frames:
                    self._queue.put_nowait(tts.SynthesizedAudio(text="", data=frame))
            else:
                self._queue.put_nowait(
                    tts.SynthesizedAudio(
                        text="",
                        data=rtc.AudioFrame(
                            data=data,
                            sample_rate=self._opts.audio_config.sample_rate_hertz,
                            num_channels=1,
                            samples_per_channel=len(data) // 2,  # 16-bit
                        ),
                    )
                )

        except Exception as e:
            logger.error(f"failed to synthesize: {e}")

    def stream(
        self,
    ) -> "SynthesizeStream":
        return SynthesizeStream(self._config)


class SynthesizeStream(tts.SynthesizeStream):
    _STREAM_EOS = ""

    def __init__(
        self,
        opts: TTSOptions,
        max_retry: int = 32,
    ):
        self._opts = opts

        self._queue = asyncio.Queue[str | None]()
        self._event_queue = asyncio.Queue[tts.SynthesisEvent | None]()
        self._closed = False
        self._text = ""

        self._main_task = asyncio.create_task(self._run(max_retry))

    """def _stream_url(self) -> str:
        base_url = self._opts.base_url
        voice_id = self._opts.voice.id
        model_id = self._opts.model_id
        sample_rate = self._opts.sample_rate
        latency = self._opts.latency
        return f"{base_url}/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format=pcm_{sample_rate}&optimize_streaming_latency={latency}"
    """
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
                        f"failed to connect to Google TTS, retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)

        except Exception:
            logger.exception("Google TTS task failed")
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

        async def raw_text_task():
            """ Use gtts synthetize method to create only one frame and send it to queue"""
            voice = self._opts.voice
            text_data = await data_rx.recv()
            if text_data is None:
                return
            gtts = gTTS(text_data, lang=self._opts.voice.language_code, slow=False)
            decoder = codecs.Mp3StreamDecoder()
            for idx, audio_data in enumerate(gtts.stream()):
                frames = decoder.decode_chunk(audio_data)
                for frame in frames:
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(
                            type=tts.SynthesisEventType.AUDIO,
                            audio=tts.SynthesizedAudio(text="", data=frame),
                        )
                    )

        async def send_task():
            """Backupp task in case coming back to word by word streaming is needed"""
            while True:
                text_data = await data_rx.recv()
                gtts = gTTS(text_data, lang=self._opts.voice.language_code, slow=False)
                decoder = codecs.Mp3StreamDecoder()
                for idx, audio_data in enumerate(gtts.stream()):
                    frames = decoder.decode_chunk(audio_data)
                    for frame in frames:
                        self._event_queue.put_nowait(
                            tts.SynthesisEvent(
                                type=tts.SynthesisEventType.AUDIO,
                                audio=tts.SynthesizedAudio(text="", data=frame),
                            )
                        )
        try:
            await asyncio.gather(raw_text_task())
        except Exception:
            logger.exception("Google TTS api connection failed")
        finally:
            self._event_queue.put_nowait(
                tts.SynthesisEvent(type=tts.SynthesisEventType.FINISHED)
            )

    async def __anext__(self) -> tts.SynthesizedAudio:
        if not self._main_task:
            self._main_task = asyncio.create_task(self._run())

        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration

        return frame


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
