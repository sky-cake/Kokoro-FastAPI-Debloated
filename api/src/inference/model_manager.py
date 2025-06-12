from typing import Optional

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend
from .kokoro_v1 import KokoroV1


class ModelManager:
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        self._config = config or model_config
        self._backend: Optional[KokoroV1] = None
        self._device: Optional[str] = None

    def _determine_device(self) -> str:
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        self._device = self._determine_device()
        self._backend = KokoroV1()

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        import time

        start = time.perf_counter()

        await self.initialize()
        model_path = self._config.pytorch_kokoro_v1_file
        await self.load_model(model_path)

        voices = await paths.list_voices()
        voice_path = await paths.get_voice_path(settings.default_voice)

        warmup_text = "Warmup text for initialization."
        voice_name = settings.default_voice
        async for _ in self.generate(warmup_text, (voice_name, voice_path)):
            pass

        ms = int((time.perf_counter() - start) * 1000)

        return self._device, "kokoro_v1", len(voices)

    def get_backend(self) -> BaseModelBackend:
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        await self._backend.load_model(path)

    async def generate(self, *args, **kwargs):
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        async for chunk in self._backend.generate(*args, **kwargs):
            yield chunk

    def unload_all(self) -> None:
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        return "kokoro_v1"


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance
