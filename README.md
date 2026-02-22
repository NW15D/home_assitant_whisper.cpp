# Home Assistant: Whisper API Integration for Speech-to-Text

Integration works for Assist pipelines. 

### Requirements:
- Installed Whisper.cpp server in network

### Server setup (whisper.cpp):
1. Git pull [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
2. Build the server (example for CUDA):
```bash
cmake -B build -DGGML_CUDA=1
cmake --build build --config Release
```

3. Run the server:
```bash
./ai/whisper.cpp/build/bin/whisper-server -m /ai/models/whisper/ggml-large-v3-turbo-q8_0.bin --host 192.168.0.55 --port 5045 -l ru  -sow -sns --vad --vad-model /ai/models/whisper/ggml-silero-v6.2.0.bin --inference-path /v1/audio/transcriptions

```

### Configuration:

Add to your `configuration.yaml`:

```yaml
stt:
  - platform: whisper_api_stt
    server_url: "http://192.168.0.55:5045/v1/audio/transcriptions"
    model: "whisper-1"
    language: "ru-RU"
    temperature: 0.0
```

#### Parameters:
- `server_url` (Optional): URL of your whisper.cpp or OpenAI-compatible server. Defaults to OpenAI API.
- `api_key` (Optional): API key if required by your server.
- `model` (Optional): Model name. Defaults to `whisper-1`.
- `language` (Optional): Language code (e.g., `en-US`, `ru-RU`).
- `temperature` (Optional): Sampling temperature between 0 and 1. Defaults to `0.0`.
- `prompt` (Optional): Optional text to guide the model's style.

### Notes:
- The integration converts the language code to ISO-639-1 (e.g., `ru-RU` -> `ru`) for API compatibility.
- Ensure your `server_url` includes the full path to the endpoint, e.g., `/v1/audio/transcriptions`.

### Used sources + thanks to:
- sfortis/openai_tts: https://github.com/sfortis/openai_tts
