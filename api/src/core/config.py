import torch
from pydantic_settings import BaseSettings

import os


root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
api_path = os.path.join(root_path, 'api')
web_path = os.path.join(root_path, 'web')

os.environ["PYTHONPATH"] = f"{root_path}:{api_path}"
os.environ["WEB_PLAYER_PATH"] = web_path


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8880

    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0
    default_voice: str = "af_heart"
    default_voice_code: str | None = None
    use_gpu: bool = True
    use_ONNX: bool = False
    device_type: str | None = None
    allow_local_voice_saving: bool = False

    model_dir: str = "src/models"
    voices_dir: str = "src/voices/v1_0"

    sample_rate: int = 24000

    target_min_tokens: int = 175
    target_max_tokens: int = 250
    absolute_max_tokens: int = 450
    advanced_text_normalization: bool = True
    voice_weight_normalization: bool = True

    gap_trim_ms: int = 1
    dynamic_gap_trim_padding_ms: int = 410
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    enable_web_player: bool = True
    web_player_path: str = "web"

    temp_file_dir: str = "api/temp_files"
    max_temp_dir_size_mb: int = 2048
    max_temp_dir_age_hours: int = 1
    max_temp_dir_count: int = 3

    def get_device(self) -> str:
        assert torch.cuda.is_available()
        return "cuda"

settings = Settings()
