import os
import tempfile

import aiofiles
from loguru import logger

from ..core.config import settings


async def cleanup_temp_files() -> None:
    try:
        if not await aiofiles.os.path.exists(settings.temp_file_dir):
            await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
            return

        files = []
        total_size = 0

        for entry in os.scandir(settings.temp_file_dir):
            if entry.is_file():
                stat = await aiofiles.os.stat(entry.path)
                files.append((entry.path, stat.st_mtime, stat.st_size))
                total_size += stat.st_size

        files.sort(key=lambda x: x[1])
        current_time = (await aiofiles.os.stat(settings.temp_file_dir)).st_mtime
        max_age = settings.max_temp_dir_age_hours * 3600

        for path, mtime, size in files:
            should_delete = False
            if current_time - mtime > max_age:
                should_delete = True
                logger.info(f"Deleting old temp file: {path}")
            elif len(files) > settings.max_temp_dir_count:
                should_delete = True
                logger.info(f"Deleting excess temp file: {path}")
            elif total_size > settings.max_temp_dir_size_mb * 1024 * 1024:
                should_delete = True
                logger.info(f"Deleting to reduce directory size: {path}")

            if should_delete:
                try:
                    await aiofiles.os.remove(path)
                    total_size -= size
                    logger.info(f"Deleted temp file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")
    except Exception as e:
        logger.warning(f"Error during temp file cleanup: {e}")


class TempFileWriter:
    def __init__(self, format: str):
        self.format = format
        self.temp_file = None
        self._finalized = False
        self._write_error = False

    async def __aenter__(self):
        try:
            await cleanup_temp_files()
            await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
            temp = tempfile.NamedTemporaryFile(
                dir=settings.temp_file_dir,
                delete=False,
                suffix=f".{self.format}",
                mode="wb",
            )
            self.temp_file = await aiofiles.open(temp.name, mode="wb")
            self.temp_path = temp.name
            temp.close()
            self.download_path = f"/download/{os.path.basename(self.temp_path)}"
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            self._write_error = True
            self.temp_path = f"unavailable_{self.format}"
            self.download_path = f"/download/{self.temp_path}"
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.temp_file and not self._finalized:
                await self.temp_file.close()
                self._finalized = True
        except Exception as e:
            logger.error(f"Error closing temp file: {e}")
            self._write_error = True

    async def write(self, chunk: bytes) -> None:
        if self._finalized:
            raise RuntimeError("Cannot write to finalized temp file")
        if self._write_error or not self.temp_file:
            return
        try:
            await self.temp_file.write(chunk)
            await self.temp_file.flush()
        except Exception as e:
            logger.error(f"Failed to write to temp file: {e}")
            self._write_error = True

    async def finalize(self) -> str:
        if self._finalized:
            raise RuntimeError("Temp file already finalized")
        if self._write_error or not self.temp_file:
            self._finalized = True
            return self.download_path
        try:
            await self.temp_file.close()
            self._finalized = True
        except Exception as e:
            logger.error(f"Failed to finalize temp file: {e}")
            self._write_error = True
            self._finalized = True
        return self.download_path
