import json
from pathlib import Path
from urllib.request import urlretrieve


def verify_files(model_path: Path, config_path: Path) -> bool:
    if not model_path.is_file() or not config_path.is_file():
        return False
    if model_path.stat().st_size == 0:
        return False
    with config_path.open() as f:
        json.load(f)
    return True


def download_model(output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = "kokoro-v1_0.pth"
    config_file = "config.json"
    model_path = output_path / model_file
    config_path = output_path / config_file

    if verify_files(model_path, config_path):
        return

    base_url = "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4"
    urlretrieve(f"{base_url}/{model_file}", model_path)
    urlretrieve(f"{base_url}/{config_file}", config_path)

    if not verify_files(model_path, config_path):
        raise RuntimeError("Downloaded files failed verification.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Kokoro v1.0 model")
    parser.add_argument("--output", required=True, help="Output directory for model files")
    args = parser.parse_args()

    download_model(args.output)


if __name__ == "__main__":
    main()
