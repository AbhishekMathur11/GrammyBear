# TeddyTalk: AI Language Learning Buddy

TeddyTalk is a low-latency language tutor for kids. One teddy, two voice games, **one WebSocket**, **one UI** (served by FastAPI from `static/`). There is no separate Next.js app to run.

The browser streams 16 kHz PCM from the Web Audio API every 250ms. FastAPI runs STT, asks vLLM (Qwen) to **judge** the answer and **invent the next prompt**, then speaks with Kokoro — all in memory.

## Architecture

```
Browser  (FastAPI serves static/)
  getUserMedia → AudioContext PCM 16 kHz / 250ms
  AudioContext.decodeAudioData ← WAV
        │  wss://<cloudflare-tunnel>/ws
        ▼
FastAPI  port 8003
        ├─ Faster-Whisper tiny.en on CPU  (leaves the 5070 for vLLM)
        ├─ vLLM on :8000                  (Qwen2.5-7B-Instruct-AWQ)
        └─ Kokoro ONNX                    (TTS in RAM)
```

Cloudflare Tunnel publishes FastAPI only. Do not tunnel vLLM.

### Why vLLM died at `--gpu-memory-utilization 0.50`

That flag was unrelated to Cloudflare. The 7B AWQ weights already used **5.29 GiB**, CUDA graphs took more, and KV cache went **negative** (`Available KV cache memory: -2.69 GiB`). Use `scripts/1_vllm.sh`: utilization **0.82**, `max-model-len 1024`, `--enforce-eager` (skips the graph capture that ate VRAM and ~40s). Whisper stays on CPU so it does not fight vLLM.

### Learning modes

1. **Sentence finish** — Teddy speaks a stem. You complete it. Qwen judges freely (not a fixed script) and invents a new stem.
2. **Find the mistake** — Teddy speaks a grammar or pronunciation error. You say the fix. Qwen judges and invents the next broken sentence.

If vLLM is down, a shuffled backup bank is used so the game still runs.

## Stack

| Layer | Choice |
| --- | --- |
| GPU | RTX 5070 12GB, almost all for vLLM |
| Env | Conda `sentence_coach` |
| UI | One production page: `static/` via FastAPI |
| STT | Faster-Whisper `tiny.en` CPU |
| LLM | vLLM `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| TTS | Kokoro ONNX under `audio_utils/tts/kokoro-tts/models` (read-only) |

`audio_test/`, `audio_utils/`, and `gemma_test/` are left unchanged.

## How to run (3 terminals)

From the repo, after `conda activate sentence_coach` once in your life and with **ffmpeg** + **cloudflared** installed:

**Terminal 1 — LLM** (wait until it is serving on 8000; first start can take several minutes):

```bash
chmod +x scripts/*.sh
./scripts/1_vllm.sh
```

**Terminal 2 — app** (only after Terminal 1 is healthy):

```bash
./scripts/2_app.sh
```

Local UI: [http://localhost:8003](http://localhost:8003)

**Terminal 3 — public HTTPS** (needed for the microphone off localhost):

```bash
./scripts/3_tunnel.sh
```

Open the `https://….trycloudflare.com` URL it prints. Allow the mic. Wait for the blue **Listening** badge before you talk.

Do not run `npm` / Next.js. That path is gone on purpose.

## Protocol

* JSON: `start` (mode, name, voice), `ready`, `idle` (no speech for ~8s), `state`, `ui`, `transcript`
* Binary up: int16 PCM @ 16 kHz
* Binary down: WAV after `{"type":"audio"}`

Type the child’s name and pick a Kokoro voice (Bella, Emma, Sky, or Michael) before starting a game. Teddy introduces itself **once**, then **explains the chosen game every time you switch modes**. Qwen judges whether an answer is a reasonable fit (not one canned word). New sentences are generated each turn.

The socket ignores your mic until `ready`, so Teddy’s own voice is not scored as your answer.

## Tests

```bash
conda activate sentence_coach
python -m unittest tests.test_agent -v
```

## Troubleshooting

* **`syntax error near unexpected token '('`** — old script quoting. Use the updated `./scripts/1_vllm.sh` (run it with bash, not `sh`).
* **`Could not find nvcc` / FlashInfer JIT** — you do not need a full CUDA toolkit. The script now sets `VLLM_USE_FLASHINFER_SAMPLER=0` so warmup uses PyTorch sampling. KV cache at 0.82 is fine on the 5070 (~3.8 GiB).
* **vLLM KV cache / no memory for cache blocks** — you are still on 0.50 utilization or CUDA graphs. Use `./scripts/1_vllm.sh`. Close other GPU apps (`nvidia-smi`).
* **Long silence after you speak** — old WebM concat never decoded. This build sends PCM and shows **Thinking** as soon as an utterance is detected.
* **Whisper + vLLM OOM** — keep STT on CPU in `config.json` (`"device": "cpu"`).
* **No teddy voice** — leave Kokoro files where they are.

Built for the Nerdy AI Hackathon Challenge 2026.
