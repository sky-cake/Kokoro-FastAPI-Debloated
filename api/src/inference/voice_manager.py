"""Voice management with controlled resource handling."""

from typing import Dict, List, Optional

import torch

from ..core import paths
from ..core.config import settings


class VoiceManager:
    _instance = None

    def __init__(self):
        self._device = settings.get_device()
        self._voices: Dict[str, torch.Tensor] = {}

    async def get_voice_path(self, voice_name: str) -> str:
        return await paths.get_voice_path(voice_name)

    async def load_voice(self, voice_name: str, device: Optional[str] = None) -> torch.Tensor:
        try:
            voice_path = await self.get_voice_path(voice_name)
            target_device = device or self._device
            voice = await paths.load_voice_tensor(voice_path, target_device)
            self._voices[voice_name] = voice
            return voice
        except Exception as e:
            raise RuntimeError(f"Failed to load voice {voice_name}: {e}")

    async def combine_voices(self, voices: List[str], device: Optional[str] = None) -> torch.Tensor:
        if len(voices) < 2:
            raise ValueError("Need at least 2 voices to combine")

        target_device = device or self._device
        voice_tensors = [await self.load_voice(name, target_device) for name in voices]
        combined = torch.mean(torch.stack(voice_tensors), dim=0)
        return combined

    async def list_voices(self) -> List[str]:
        return await paths.list_voices()

    def cache_info(self) -> Dict[str, int]:
        return {"loaded_voices": len(self._voices), "device": self._device}


async def get_manager() -> VoiceManager:
    if VoiceManager._instance is None:
        VoiceManager._instance = VoiceManager()
    return VoiceManager._instance
