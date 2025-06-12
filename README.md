## About

A debloated version of https://github.com/remsky/Kokoro-FastAPI.

Only contains the necessary Python packages and endpoints for doing TTS.

## Running

This script assumes you have python3.12 installed.

`./start-gpu`

## API

```python
import requests

response = requests.get("http://localhost:8880/v1/audio/voices")
voices = response.json()["voices"]

response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "model": "kokoro",  
        "input": "Hello world!",
        "voice": "af_bella",
        "response_format": "mp3",  # mp3, wav, opus, flac
        "speed": 1.0
    }
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```
