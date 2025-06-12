from fastapi import APIRouter
from fastapi.responses import Response

from ..core.config import settings
from ..core.paths import get_content_type, get_web_file_path, read_bytes

router = APIRouter(
    tags=["Web Player"],
    responses={404: {"description": "Not found"}},
)

if settings.enable_web_player:
    @router.get("/{filename:path}")
    async def serve_web_file(filename: str):
        if filename == "" or filename == "/":
            filename = "index.html"

        file_path = await get_web_file_path(filename)
        if file_path:
            content = await read_bytes(file_path)
            content_type = await get_content_type(file_path)

            return Response(
                content=content,
                media_type=content_type,
                headers={
                    "Cache-Control": "no-cache",
                },
            )
