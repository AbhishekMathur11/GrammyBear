# TeddyTalk: AI Language Learning Buddy

TeddyTalk is a low-latency language tutor for kids. One teddy, two voice games, **one WebSocket**, **one UI** (the Figma React screen, built into `ui-figma/dist` and served by FastAPI). Catch-the-Mistake stays in the tutor code but is not on this home screen.

The browser streams 16 kHz PCM from the Web Audio API every 250ms. FastAPI runs STT, asks vLLM (Qwen) to **judge** the answer and **invent the next prompt**, then speaks with Kokoro — all in memory.

## Architecture

```
Browser  (FastAPI serves ui-figma/dist)
  getUserMedia → AudioContext PCM 16 kHz / 250ms
  AudioContext.decodeAudioData ← WAV
        │  wss://<cloudflare-tunnel>/ws
        ▼
FastAPI  port 8003
        ├─ Faster-Whisper tiny.en on CPU  (leaves the 5070 for vLLM)
        ├─ vLLM on :8000                  (Qwen/Qwen2.5-14B-Instruct-AWQ)
        └─ Kokoro ONNX                    (TTS in RAM)
```

Cloudflare Tunnel publishes FastAPI only. Do not tunnel vLLM.

### Why vLLM died at `--gpu-memory-utilization 0.50`

That flag was unrelated to Cloudflare. The 14B AWQ weights need almost all of the 12GB card. Use `scripts/1_vllm.sh`: `Qwen/Qwen2.5-14B-Instruct-AWQ`, utilization **0.88**, `max-model-len 512`, `--enforce-eager`, `max-num-seqs 1`. Marlin unpack is disabled (`VLLM_BATCH_INVARIANT=1`) so vLLM does not OOM while converting AWQ weights. Whisper stays on CPU so it does not fight vLLM.

### Learning modes

1. **Sentence finish** — Teddy speaks a stem. You complete it. Qwen judges freely (not a fixed script) and invents a new stem.
2. **Story Adventure** — Teddy starts a fill-in-the-blank line. Any kid-safe ending is fine. The next line continues the tale. After five good beats, Teddy narrates the whole chapter (star-party popup). Catch-the-Mistake remains in the backend unused by this UI.

If vLLM is down, a shuffled backup bank is used so the game still runs.

## Stack

| Layer | Choice |
| --- | --- |
| GPU | RTX 5070 12GB, almost all for vLLM |
| Env | Conda `sentence_coach` |
| UI | Figma React (`ui-figma/`), built by `scripts/2_app.sh`, served from FastAPI |
| STT | Faster-Whisper `tiny.en` CPU |
| LLM | vLLM `Qwen/Qwen2.5-14B-Instruct-AWQ` |
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

This PC only: [http://localhost:8003](http://localhost:8003)

**Terminal 3 — public HTTPS** (required for the mic on any other phone/laptop):

```bash
./scripts/3_tunnel.sh
```

Leave all three scripts running on **this** GPU PC. vLLM stays on `http://127.0.0.1:8000` and is not tunneled.

### Another phone or laptop

Do **not** open `http://localhost` on that device. Localhost there is that device, not this PC.

Other devices: open **[https://teddytalk.loca.lt](https://teddytalk.loca.lt)** (named by `scripts/3_tunnel.sh`). If loca.lt shows a click-through page, tap Continue once, then allow the mic.

To use your own domain instead, set `CLOUDFLARED_TUNNEL_TOKEN` and put that hostname in `config.json` → `tunnel.public_url`.

`http://<this-PC-LAN-IP>:8003` can load the page on the same Wi‑Fi, but browsers usually **block the mic** on plain HTTP. Use the HTTPS URL for voice.

`scripts/2_app.sh` runs `npm run build` in `ui-figma` then starts FastAPI. Do not run a second frontend server.

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
