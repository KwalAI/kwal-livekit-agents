# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
class TTSOptions:
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


DEFAULT_VOICE = Voice(
    id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    category="premade",
    settings=VoiceSettings(
        stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
    ),
)


class TTS(tts.TTS):
    def __init__(
        self,
        config: Optional[TTSOptions] = None,
        *,
        language: LgType = "en-US",
        gender: GenderType = "neutral",
        voice_name: str = "",  # Not required
        voice: Voice = DEFAULT_VOICE,
        audio_encoding: AudioEncodingType = "wav",
        sample_rate: int = 24000,
        speaking_rate: float = 1.0,
        credentials_info: Optional[dict] = None,
        credentials_file: Optional[str] = None,
    ) -> None:
        super().__init__(
            streaming_supported=True, sample_rate=sample_rate, num_channels=1
        )

        self._opts = config

        if credentials_info:
            self._client = (
                texttospeech.TextToSpeechAsyncClient.from_service_account_info(
                    credentials_info
                )
            )
        elif credentials_file:
            self._client = (
                texttospeech.TextToSpeechAsyncClient.from_service_account_file(
                    credentials_file
                )
            )
        else:
            self._client = texttospeech.TextToSpeechAsyncClient()

        if not config:
            _gender = SsmlVoiceGender.NEUTRAL
            if gender == "male":
                _gender = SsmlVoiceGender.MALE
            elif gender == "female":
                _gender = SsmlVoiceGender.FEMALE
            voice = texttospeech.VoiceSelectionParams(
                name=voice_name,
                language_code=language,
                ssml_gender=_gender,
            )
            # Support wav and mp3 only
            if audio_encoding == "wav":
                _audio_encoding = texttospeech.AudioEncoding.LINEAR16
            elif audio_encoding == "mp3":
                _audio_encoding = texttospeech.AudioEncoding.MP3
            # elif audio_encoding == "opus":
            #     _audio_encoding = texttospeech.AudioEncoding.OGG_OPUS
            # elif audio_encoding == "mulaw":
            #     _audio_encoding = texttospeech.AudioEncoding.MULAW
            # elif audio_encoding == "alaw":
            #     _audio_encoding = texttospeech.AudioEncoding.ALAW
            else:
                raise NotImplementedError(
                    f"Audio encoding {audio_encoding} is not supported"
                )

            config = TTSOptions(
                voice=voice,
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=_audio_encoding,
                    sample_rate_hertz=sample_rate,
                    speaking_rate=speaking_rate,
                ),
            )
        self._config = config

    async def synthesize(
        self,
        *,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        try:
            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            response: SynthesizeSpeechResponse = await self._client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=self._config.voice,
                audio_config=self._config.audio_config,
            )

            data = response.audio_content
            if self._config.audio_config.audio_encoding == "mp3":
                decoder = codecs.Mp3StreamDecoder()
                frames = decoder.decode_chunk(data)
                for frame in frames:
                    yield tts.SynthesizedAudio(text=text, data=frame)
            else:
                yield tts.SynthesizedAudio(
                    text=text,
                    data=rtc.AudioFrame(
                        data=data,
                        sample_rate=self._config.audio_config.sample_rate_hertz,
                        num_channels=1,
                        samples_per_channel=len(data) // 2,  # 16-bit
                    ),
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
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
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
            seg = seg.strip() + " "  # 11labs expects a space at the end
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

        try:
            while True:
                try:
                    data = await self._queue.get()

                    if data is None:
                        break

                    if data == SynthesizeStream._STREAM_EOS:
                        continue

                    data_tx, data_rx = aio.channel()
                    #for idx, decoded in enumerate(gtts.stream()):
                        #fp.write(decoded)
                        #log.debug("part-%i written to %s", idx, fp)
                    tts_task = asyncio.create_task(self._run_tts(data_rx))

                    assert data_tx is not None
                    assert tts_task is not None

                    data_tx.send_nowait(data) # Correctly sending this data

                except Exception:
                    if retry_count >= max_retry:
                        logger.exception(
                            f"failed to connect to 11labs after {max_retry} retries"
                        )
                        break

                    retry_delay = min(retry_count * 5, 5)  # max 5s
                    retry_count += 1

                    logger.warning(
                        f"failed to connect to 11labs, retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)

        except Exception:
            logger.exception("11labs task failed")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                if tts_task is not None:
                    tts_task.cancel()
                    await tts_task

            self._event_queue.put_nowait(None)

    async def _run_tts(
        self, data_rx: aio.ChanReceiver[str]
    ) -> None:
        closing_ws = False

        self._event_queue.put_nowait(
            tts.SynthesisEvent(type=tts.SynthesisEventType.STARTED)
        )

        async def send_task():
            voice = self._opts.voice
            #voice_settings = (
            #    asdict(voice.settings) if voice.settings else None
            #)
            while True:
                data = await data_rx.recv()
                gtts = gTTS(data, lang=self._opts.voice.language_code)
                for idx, decoded in enumerate(gtts.stream()):
                    frame = rtc.AudioFrame(
                        data=decoded,
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=len(decoded) // 2,
                    )
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(
                            type=tts.SynthesisEventType.AUDIO,
                            audio=tts.SynthesizedAudio(text="", data=frame),
                        )
                    )

        """async def recv_task():
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected
                        return

                    raise Exception("11labs connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected 11labs message type %s", msg.type)
                    continue

                data: dict = json.loads(msg.data)
                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    frame = rtc.AudioFrame(
                        data=b64data,
                        sample_rate=self._opts.sample_rate,
                        num_channels=1,
                        samples_per_channel=len(b64data) // 2,
                    )
                    self._event_queue.put_nowait(
                        tts.SynthesisEvent(
                            type=tts.SynthesisEventType.AUDIO,
                            audio=tts.SynthesizedAudio(text="", data=frame),
                        )
                    )
                elif data.get("isFinal"):
                    return"""

        try:
            await asyncio.gather(send_task())
        except Exception:
            logger.exception("11labs connection failed")
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
