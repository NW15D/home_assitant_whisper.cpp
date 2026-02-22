"""Custom integration for OpenAI Whisper API STT."""
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

DOMAIN = "whisper_api_stt"

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Whisper API STT integration."""
    return True