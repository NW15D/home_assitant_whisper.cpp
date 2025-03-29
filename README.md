# Home Assistant: Whisper API Integration von Speech-to-Text

Integration works for Assist pipelines. 

### Requirements:
- Installed Whisper.cpp server in network

1.Git pull whisper.cpp

my config for Tesla P40 or use just cmake -B build  -DGGML_CUDA=1 -DGGML_CCACHE=OFF

2.cmake -B build  -DGGML_CUDA_FORCE_MMQ=1 -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA=1 -DGGML_F16C=OFF -DGGML_AVX512=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_CCACHE=OFF

3.cmake --build build --config Release

run 
./build/bin/server -m /ai/models/whisper/ggml-large-v3-q5_0.bin --host 192.168.0.55 --port 5005 -l en -fa

### Configuration:

Remarks:
- Add your own server_url
- language MUST be set AND has to be ISO-639-1 format
- There will be an error in the home assistant logs, that configuring stt is not allowed in configuration.yaml - you can ignore this

configuration.yaml:


```
stt:
  - platform: whisper_api_stt
    api_key: "a132b20c-96be-467f-a15a-ed08aed67345"
    server_url: "http://192.168.0.176:9000/v1/audio/transcriptions"
    model: "whisper-1"
    language: "ru-RU"
    temperature: "0.5"

```

### Used sources + thanks to:
- sfortis/openai_tts: https://github.com/sfortis/openai_tts


up!
