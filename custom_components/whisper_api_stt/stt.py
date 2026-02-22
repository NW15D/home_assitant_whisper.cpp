r"""
Support for Whisper API STT.
"""
from __future__ import annotations

import logging
import os
import tempfile
import wave
from typing import AsyncIterable

import aiohttp
import voluptuous as vol

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
from homeassistant.const import CONF_LANGUAGE, CONF_NAME, CONF_TEMPERATURE, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = "api_key"
DEFAULT_LANG = "en-US"
CONF_MODEL = "model"
CONF_URL = "server_url"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"

OPENAI_STT_URL = "http://192.168.0.55:5005/v1/audio/transcriptions"

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_API_KEY, default=""): cv.string,
    vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANG): cv.string,
    vol.Optional(CONF_MODEL, default="whisper-1"): cv.string,
    vol.Optional(CONF_URL, default=OPENAI_STT_URL): cv.string,
    vol.Optional(CONF_PROMPT): cv.string,
    vol.Optional(CONF_TEMPERATURE, default=0.0): vol.Coerce(float),
})


async def async_get_engine(hass: HomeAssistant, config: dict, discovery_info=None):
    """Set up Whisper API STT speech component."""
    api_key = config.get(CONF_API_KEY)
    language = config.get(CONF_LANGUAGE, DEFAULT_LANG)
    model = config.get(CONF_MODEL)
    url = config.get(CONF_URL)
    prompt = config.get(CONF_PROMPT)
    temperature = config.get(CONF_TEMPERATURE)
    return OpenAISTTProvider(hass, api_key, language, model, url, prompt, temperature)


class OpenAISTTProvider(Provider):
    """The Whisper API STT provider."""

    def __init__(self, hass, api_key, lang, model, url, prompt, temperature):
        """Initialize Whisper API STT provider."""
        self.hass = hass
        self._api_key = api_key
        self._language = lang
        self._model = model
        self._url = url
        self._prompt = prompt
        self._temperature = temperature

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._language.split(',')[0]

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        return [self._language]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to text."""
        data = b""
        async for chunk in stream:
            data += chunk

        if not data:
            _LOGGER.error("Received empty audio stream")
            return SpeechResult("", SpeechResultState.ERROR)

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file_path = temp_file.name
                with wave.open(temp_file_path, "wb") as wav_file:
                    wav_file.setnchannels(metadata.channel)
                    wav_file.setsampwidth(2)  # 16-bit PCM
                    wav_file.setframerate(metadata.sample_rate)
                    wav_file.writeframes(data)

            # OpenAI expects ISO-639-1 (e.g. 'en')
            lang_iso = self._language.split("-")[0].split("_")[0]

            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            form = aiohttp.FormData()
            form.add_field("model", self._model)
            form.add_field("language", lang_iso)
            if self._prompt:
                form.add_field("prompt", self._prompt)
            form.add_field("temperature", str(self._temperature))
            
            # Open file for reading to send
            with open(temp_file_path, "rb") as audio_file:
                form.add_field(
                    "file", audio_file, filename="audio.wav", content_type="audio/wav"
                )

                session = async_get_clientsession(self.hass)
                async with session.post(
                    self._url, data=form, headers=headers, timeout=60
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(
                            "Error from Whisper API (status %s): %s",
                            response.status,
                            error_text,
                        )
                        return SpeechResult("", SpeechResultState.ERROR)

                    json_response = await response.json()
                    text = json_response.get("text", "")
                    return SpeechResult(text, SpeechResultState.SUCCESS)

        except Exception as err:
            _LOGGER.exception("Error processing audio stream: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as err:
                    _LOGGER.warning("Could not remove temporary file %s: %s", temp_file_path, err)

