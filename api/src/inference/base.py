from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Tuple, Union

import numpy as np
import torch


class AudioChunk:
    """Represents audio chunks returned by model backends."""

    def __init__(
        self,
        audio: np.ndarray,
        word_timestamps: Optional[List] = None,
        output: Optional[Union[bytes, np.ndarray]] = b"",
    ):
        self.audio = audio
        self.word_timestamps = word_timestamps or []
        self.output = output

    @staticmethod
    def combine(audio_chunks: List["AudioChunk"]) -> "AudioChunk":
        combined_audio = audio_chunks[0].audio
        combined_timestamps = audio_chunks[0].word_timestamps.copy() if audio_chunks[0].word_timestamps else []

        for chunk in audio_chunks[1:]:
            combined_audio = np.concatenate((combined_audio, chunk.audio), dtype=np.int16)
            if chunk.word_timestamps:
                combined_timestamps += chunk.word_timestamps

        return AudioChunk(combined_audio, combined_timestamps)


class ModelBackend(ABC):
    """Abstract base class for model inference backend."""

    @abstractmethod
    async def load_model(self, path: str) -> None:
        """Load model from file path.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        pass

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio from text.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass


class BaseModelBackend(ModelBackend):

    def __init__(self):
        self._model: Optional[torch.nn.Module] = None
        self._device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
