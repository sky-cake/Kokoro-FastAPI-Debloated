from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .core.config import settings
from .routers.api import router as api_router
from .routers.web_player import router as web_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .inference.model_manager import get_manager
    from .inference.voice_manager import get_manager as get_voice_manager
    from .services.temp_manager import cleanup_temp_files

    await cleanup_temp_files()

    model_manager = await get_manager()
    voice_manager = await get_voice_manager()

    device, model, voicepack_count = await model_manager.initialize_with_warmup(voice_manager)
    yield


app = FastAPI(
    lifespan=lifespan,
    openapi_url="/openapi.json",
)

app.include_router(api_router, prefix="/v1")
if settings.enable_web_player:
    app.include_router(web_router, prefix="/web")


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
