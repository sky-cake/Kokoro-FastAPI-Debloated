import io
import json
import os
from typing import Callable, List, Optional, Set

import aiofiles
import aiofiles.os
import torch

from .config import settings


async def _find_file(
    filename: str,
    search_paths: List[str],
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    if os.path.isabs(filename) and await aiofiles.os.path.exists(filename):
        return filename

    for path in search_paths:
        full_path = os.path.join(path, filename)
        if await aiofiles.os.path.exists(full_path):
            if filter_fn is None or filter_fn(full_path):
                return full_path


async def _scan_directories(search_paths: List[str], filter_fn: Optional[Callable[[str], bool]] = None) -> Set[str]:
    results = set()

    for path in search_paths:
        if not await aiofiles.os.path.exists(path):
            continue

        entries = await aiofiles.os.scandir(path)
        for entry in entries:
            if filter_fn is None or filter_fn(entry.name):
                results.add(entry.name)

    return results


async def get_model_path(model_name: str) -> str:
    api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_dir = os.path.join(api_dir, settings.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    search_paths = [model_dir]
    return await _find_file(model_name, search_paths)


async def get_voice_path(voice_name: str) -> str:
    api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    voice_dir = os.path.join(api_dir, settings.voices_dir)
    os.makedirs(voice_dir, exist_ok=True)
    voice_file = f"{voice_name}.pt"
    search_paths = [voice_dir]
    return await _find_file(voice_file, search_paths)


async def list_voices() -> List[str]:
    api_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    voice_dir = os.path.join(api_dir, settings.voices_dir)
    os.makedirs(voice_dir, exist_ok=True)
    search_paths = [voice_dir]

    def filter_voice_files(name: str) -> bool:
        return name.endswith(".pt")

    voices = await _scan_directories(search_paths, filter_voice_files)
    return sorted([name[:-3] for name in voices])


async def load_voice_tensor(voice_path: str, device: str = "cpu", weights_only=False) -> torch.Tensor:
    async with aiofiles.open(voice_path, "rb") as f:
        data = await f.read()
        return torch.load(io.BytesIO(data), map_location=device, weights_only=weights_only)


async def save_voice_tensor(tensor: torch.Tensor, voice_path: str) -> None:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    async with aiofiles.open(voice_path, "wb") as f:
        await f.write(buffer.getvalue())


async def load_json(path: str) -> dict:
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return json.loads(await f.read())


async def load_model_weights(path: str, device: str = "cpu") -> dict:
    async with aiofiles.open(path, "rb") as f:
        return torch.load(io.BytesIO(await f.read()), map_location=device, weights_only=True)


async def read_file(path: str) -> str:
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def read_bytes(path: str) -> bytes:
    async with aiofiles.open(path, "rb") as f:
        return await f.read()


async def get_web_file_path(filename: str) -> str:
    web_dir = os.path.join("/app", settings.web_player_path)
    return await _find_file(filename, [web_dir])


async def get_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".html": "text/html",
        ".js": "application/javascript",
        ".css": "text/css",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }.get(ext, "application/octet-stream")


async def verify_model_path(model_path: str) -> bool:
    return await aiofiles.os.path.exists(model_path)


async def cleanup_temp_files() -> None:
    if not await aiofiles.os.path.exists(settings.temp_file_dir):
        await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
        return
    entries = await aiofiles.os.scandir(settings.temp_file_dir)
    for entry in entries:
        if entry.is_file():
            stat = await aiofiles.os.stat(entry.path)
            max_age = stat.st_mtime + settings.max_temp_dir_age_hours * 3600
            if max_age < stat.st_mtime:
                await aiofiles.os.remove(entry.path)


async def get_temp_file_path(filename: str) -> str:
    temp_path = os.path.join(settings.temp_file_dir, filename)
    if not await aiofiles.os.path.exists(settings.temp_file_dir):
        await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
    return temp_path


async def list_temp_files() -> List[str]:
    if not await aiofiles.os.path.exists(settings.temp_file_dir):
        return []
    entries = await aiofiles.os.scandir(settings.temp_file_dir)
    return [entry.name for entry in entries if entry.is_file()]


async def get_temp_dir_size() -> int:
    if not await aiofiles.os.path.exists(settings.temp_file_dir):
        return 0
    total = 0
    entries = await aiofiles.os.scandir(settings.temp_file_dir)
    for entry in entries:
        if entry.is_file():
            stat = await aiofiles.os.stat(entry.path)
            total += stat.st_size
    return total
