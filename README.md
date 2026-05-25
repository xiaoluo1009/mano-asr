# mano-asr

`mano-asr` is a local ASR service built around MLX Fun-ASR-Nano. It provides a small FastAPI server for single-audio transcription, optional FSMN VAD segmentation, hotword extraction from personal context, request/session logging, and evaluation/benchmark scripts for Fun-ASR-Nano experiments.

The current runtime path is intentionally narrow:

- load one local/uploaded audio file;
- optionally run FSMN VAD and transcribe each speech segment;
- return plain transcript text;
- expose an HTTP API compatible with a simple voice-transcription client flow.

## Features

- MLX Fun-ASR-Nano inference through `core.models.funasr`.
- Optional MLX FSMN VAD through `core.models.fsmn`.
- Audio upload API with validation, size/duration limits, and CORS enabled.
- Supported upload extensions: `.wav`, `.mp3`, `.ogg`, `.webm`, `.m4a`, `.flac`.
- Non-WAV uploads are decoded to 16 kHz mono WAV with `ffmpeg` before ASR.
- Context-aware hotword prompt extraction from `personal_context`.
- Session records saved under `sessions/YYYY-MM-DD/`, including request metadata, response body, and retained audio artifacts.
- Local client script, inference scripts, CER evaluation scripts, and end-to-end benchmark scripts.

## Project Layout

```text
.
├── server.py                         # FastAPI ASR service
├── client.py                         # Test client for /v1/voice/transcribe
├── core/
│   ├── auto_model.py                 # Single-audio ASR wrapper with optional VAD
│   └── models/
│       ├── funasr/                   # MLX Fun-ASR-Nano implementation
│       ├── fsmn/                     # MLX FSMN VAD implementation
│       └── qwen3_asr/                # Qwen3 ASR / forced-aligner experiments
├── utils/
│   ├── load_utils.py                 # Audio-only numpy/ffmpeg loader
│   └── hotwords_extractor.py         # Hotword extraction from context text
├── exp/
│   ├── infer/                        # One-off inference examples
│   ├── eval/                         # CER evaluation scripts
│   └── benchmarks/                   # Speed/accuracy benchmark scripts
├── scripts/                           # Convenience launch/eval commands
├── docs/voice-transcribe-api.md      # Historical/upstream API contract notes
├── assets/                           # Sample audio files
├── models/                           # Local model directories
└── sessions/                         # Runtime session logs and copied audio
```

## Requirements

This repository does not currently include a pinned `requirements.txt` or `pyproject.toml`. The dependency list below is inferred from the source.

Runtime:

- macOS on Apple Silicon is recommended for MLX.
- Python 3.10+.
- `ffmpeg` and `ffprobe` on `PATH`.
- Python packages:
  - `mlx`
  - `mlx-audio`
  - `numpy`
  - `fastapi`
  - `uvicorn`
  - `python-multipart`
  - `requests`
  - `soundfile`
  - `scipy`
  - `safetensors`
  - `transformers`
  - `tqdm`
  - `huggingface_hub` if you use the `hf download` commands below

Evaluation/benchmark extras:

- `jiwer`
- `cider` if running the Cider W8A8 experiment scripts

Install example:

```bash
brew install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install \
  mlx mlx-audio numpy fastapi "uvicorn[standard]" python-multipart \
  requests soundfile scipy safetensors transformers tqdm jiwer huggingface_hub
```

If you use Cider scripts, install `cider` in the same environment according to that project's installation instructions.

## Model Files

The server requires an ASR model path. VAD is optional.

Recommended local layout:

```text
models/
├── mlx-community/
│   └── Fun-ASR-Nano-2512-8bit/
└── fsmn-vad-mlx/
    ├── am.mvn
    ├── config.json
    └── model.safetensors
```

Example model download command:

```bash
hf download mlx-community/Fun-ASR-Nano-2512-8bit \
  --local-dir models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --max-workers 8
```

For users behind a domestic mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com hf download mlx-community/Fun-ASR-Nano-2512-8bit \
  --local-dir models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --max-workers 8
```

The checked-in `models/fsmn-vad-mlx/` directory is the expected MLX VAD format. If you use a different VAD directory, it must contain the same required files: `config.json`, `model.safetensors`, and optionally `am.mvn`.

## Run the API Server

Start without VAD:

```bash
python3 server.py \
  --model-path models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --host 0.0.0.0 \
  --port 8787 \
  --load-on-startup
```

Start with VAD:

```bash
python3 server.py \
  --model-path models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --vad-model-path models/fsmn-vad-mlx \
  --host 0.0.0.0 \
  --port 8787 \
  --load-on-startup
```

The `scripts/start.sh` script is a convenience wrapper, but check the model paths before using it because local model directories may differ by machine.

Health check:

```bash
curl http://127.0.0.1:8787/
```

Expected response:

```json
{"status":200,"service":"mano-asr","message":"ok"}
```

## API

### `POST /v1/voice/transcribe`

Transcribe one uploaded audio file.

Request type: `multipart/form-data`

Fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `audio` | file | yes | Audio file. Supported extensions: `.wav`, `.mp3`, `.ogg`, `.webm`, `.m4a`, `.flac`. |
| `context_text` | string | no | Existing text. Used by append/edit modes. Kept to the last 5000 chars. |
| `chat_context` | string | no | Chat context. Kept to the last 20000 chars. |
| `personal_context` | string | no | Personal correction/hotword context. Kept to the last 10000 chars. |
| `member_context` | string | no | Member context. Kept to the last 5000 chars. |
| `mode` | string | no | `smart`, `append_only`, or `edit_only`. Default: `smart`. |

Limits:

- default max file size: `30 MiB`;
- default max duration: `660` seconds;
- `edit_only` requires `context_text`.

Modes:

- `smart`: currently appends the transcript to `context_text`.
- `append_only`: appends the transcript to `context_text`.
- `edit_only`: returns the transcript only after requiring `context_text`.

Example:

```bash
curl -X POST http://127.0.0.1:8787/v1/voice/transcribe \
  -F "audio=@assets/BAC009S0764W0129.wav" \
  -F "mode=smart"
```

Success response:

```json
{
  "status": 200,
  "text": "转写文本",
  "m": "fun-asr-nano",
  "engine": "mlx"
}
```

### `GET /v1/voice/config`

Return current voice service limits and engine metadata.

```bash
curl http://127.0.0.1:8787/v1/voice/config
```

Example response:

```json
{
  "enabled": true,
  "max_duration": 660,
  "max_file_size": 31457280,
  "engine": "mlx",
  "edit_mode": "append"
}
```

### Authentication

Authentication is disabled by default. If the server is started with `--auth-token`, requests to `/v1/voice/transcribe` must include:

```text
Authorization: Bearer <token>
```

Example:

```bash
python3 server.py \
  --model-path models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --auth-token "$MANO_ASR_TOKEN"
```

```bash
curl -X POST http://127.0.0.1:8787/v1/voice/transcribe \
  -H "Authorization: Bearer $MANO_ASR_TOKEN" \
  -F "audio=@assets/BAC009S0764W0129.wav"
```

## Test Client

Fetch service config:

```bash
python3 client.py --config-only
```

Transcribe the bundled sample audio:

```bash
python3 client.py --audio assets/BAC009S0764W0129.wav
```

Use context and auth:

```bash
python3 client.py \
  --url http://127.0.0.1:8787 \
  --audio assets/BAC009S0764W0129.wav \
  --mode append_only \
  --context-text "已有内容：" \
  --personal-context "## 术语\n- FunASR（语音识别模型）" \
  --auth-token "$MANO_ASR_TOKEN"
```

## Python Usage

Direct ASR without VAD:

```python
from core.auto_model import AutoModel

model = AutoModel(
    model="models/mlx-community/Fun-ASR-Nano-2512-8bit",
    vad_model=None,
)

text = model.generate(
    "assets/BAC009S0764W0129.wav",
    task="translate",
    target_language="zh",
)
print(text)
```

ASR with VAD:

```python
from core.auto_model import AutoModel

model = AutoModel(
    model="models/mlx-community/Fun-ASR-Nano-2512-8bit",
    vad_model="models/fsmn-vad-mlx",
)

text = model.generate(
    "assets/BAC009S0764W0129.wav",
    task="translate",
    target_language="zh",
    merge_vad=True,
)
print(text)
```

`AutoModel.generate()` returns a plain `str`, not an `STTOutput` object.

## Evaluation and Benchmarks

CER evaluation scripts expect a dataset directory containing a JSON dataset or a `dataset_info.json` mapping.

Baseline MLX evaluation:

```bash
python3 exp/eval/fun_asr_nano_mlx.py \
  --model models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --dataset_dir /path/to/dataset_dir \
  --dataset dataset_name \
  --output_dir outputs/dataset_name_mlx
```

VAD + ASR evaluation:

```bash
python3 exp/eval/fsmn_vad_fun_asr_nano_mlx.py \
  --model models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --vad_model models/fsmn-vad-mlx \
  --dataset_dir /path/to/dataset_dir \
  --dataset dataset_name \
  --output_dir outputs/dataset_name_vad_mlx
```

End-to-end benchmark:

```bash
python3 exp/benchmarks/bench_fun_asr_nano.py \
  --model models/mlx-community/Fun-ASR-Nano-2512-8bit \
  --dataset_dir /path/to/benchmark \
  --manifest SeniorTalk_test/manifest_merged.jsonl \
  --n_samples 10
```

Several scripts under `exp/infer/`, `exp/eval/`, and `exp/benchmarks/` contain machine-specific absolute paths from development runs. Treat them as examples and replace model/dataset paths before running.

## Runtime Notes

- The server uses a single-worker `ThreadPoolExecutor` and an async lock around generation, so requests are processed serially through the model path.
- `ffmpeg` is required for decoding non-WAV uploads and by the local audio loader.
- `ffprobe` is used for duration detection when `soundfile` cannot read the upload directly.
- Session logs are written by default and may include copied audio files. Do not expose `sessions/` publicly.
- `docs/voice-transcribe-api.md` contains broader API notes from a larger voice subsystem. The current `server.py` implementation only exposes `/`, `/v1/voice/transcribe`, and `/v1/voice/config`.

## Troubleshooting

### `RuntimeError: ffmpeg is required to decode this audio format`

Install ffmpeg and make sure both `ffmpeg` and `ffprobe` are available on `PATH`.

```bash
brew install ffmpeg
ffmpeg -version
ffprobe -version
```

### `model path is required`

Start `server.py` with `--model-path`.

### `Audio file not found`

`AutoModel.generate()` accepts a local path. Expand or resolve the path before passing it if the caller runs from a different working directory.

### VAD returns no text

Run without VAD first to verify ASR works:

```bash
python3 server.py --model-path models/mlx-community/Fun-ASR-Nano-2512-8bit
```

Then verify the VAD model directory contains `config.json`, `model.safetensors`, and `am.mvn`.

## Development Checks

Basic syntax check:

```bash
python3 -m py_compile \
  server.py client.py core/auto_model.py utils/load_utils.py utils/hotwords_extractor.py
```

Search for remaining heavyweight audio dependencies in the narrow loader path:

```bash
rg -n "\btorch\b|torchaudio" utils/load_utils.py core/auto_model.py
```
