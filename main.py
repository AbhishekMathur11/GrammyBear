import asyncio
import base64
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent import VOICE_CHOICES, LanguageTutor, TutorEvent, int16_bytes_to_pcm, wav_bytes_to_pcm

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
FIGMA_DIST = ROOT / "ui-figma" / "dist"


def load_config(filepath: str = "config.json") -> dict:
    with open(filepath, "r", encoding="utf-8") as handle:
        return json.load(handle)


config = load_config()
engines: Optional[LanguageTutor] = None

app = FastAPI(title="TeddyTalk", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
if (FIGMA_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FIGMA_DIST / "assets")), name="figma-assets")


def shared_engines() -> LanguageTutor:
    global engines
    if engines is None:
        print("Loading tutor engines (first request)…", flush=True)
        engines = LanguageTutor.from_config(config)
        print("Tutor engines ready.", flush=True)
    return engines


def new_session(mode: str = "complete", name: str = "", voice: str = "") -> LanguageTutor:
    base = shared_engines()
    session = LanguageTutor(
        llm=base.llm,
        stt=base.stt,
        tts=base.tts,
        sample_rate=base.sample_rate,
        assembler=base.assembler.clone(),
    )
    session.set_mode(mode)
    session.configure(name=name, voice=voice)
    return session


async def emit(ws: WebSocket, events: list[TutorEvent]) -> None:
    for event in events:
        if event.type == "audio" and event.audio:
            b64 = base64.b64encode(event.audio).decode("ascii")
            await ws.send_text(json.dumps({"type": "audio", "mime": "audio/wav", "b64": b64, **event.payload}))
            continue
        await ws.send_text(json.dumps({"type": event.type, **event.payload}))


async def emit_then_voice(ws: WebSocket, session: LanguageTutor, events: list[TutorEvent]) -> None:
    await emit(ws, events)
    await ws.send_text(
        json.dumps({"type": "state", "state": "speaking", "feedback": "Teddy is warming up her voice…"})
    )
    wav = await asyncio.to_thread(session.pending_speech_audio)
    if wav:
        print(f"Sending {len(wav)} bytes of TTS", flush=True)
        b64 = base64.b64encode(wav).decode("ascii")
        await ws.send_text(json.dumps({"type": "audio", "mime": "audio/wav", "b64": b64}))
        return
    print("TTS returned no audio; opening mic anyway", flush=True)
    session.mark_ready()
    await ws.send_text(
        json.dumps({"type": "state", "state": "listening", "feedback": "Listening… speak now!"})
    )


@app.get("/")
async def read_root():
    for index in (FIGMA_DIST / "index.html", STATIC / "index.html"):
        if index.exists():
            return FileResponse(index)
    return JSONResponse({"service": "TeddyTalk", "ws": "/ws"})


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "TeddyTalk",
        "engines_loaded": engines is not None,
        "vllm": config.get("llm", {}).get("base_url"),
    }


@app.websocket("/ws")
async def tutor_socket(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected", flush=True)
    session: Optional[LanguageTutor] = None
    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "hello",
                    "codec": "pcm16",
                    "sample_rate": config.get("audio", {}).get("sample_rate", 16000),
                    "chunk_ms": config.get("audio", {}).get("chunk_ms", 250),
                    "modes": ["complete", "story", "mistake"],
                    "voices": list(VOICE_CHOICES),
                    "default_voice": "af_bella",
                }
            )
        )
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                print("WebSocket disconnected", flush=True)
                return
            if message.get("text"):
                data = json.loads(message["text"])
                kind = data.get("type")
                if kind != "level":
                    print(f"WS text: {kind}", flush=True)
                if kind == "start":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "state",
                                "state": "thinking",
                                "feedback": "Starting the game…",
                            }
                        )
                    )
                    mode = data.get("mode", "complete")
                    name = data.get("name", "")
                    voice = data.get("voice", "")
                    if session is None:
                        session = await asyncio.to_thread(new_session, mode, name, voice)
                    else:
                        session.configure(name, voice)
                        session.set_mode(mode)
                    await emit_then_voice(websocket, session, await asyncio.to_thread(session.start_turn))
                elif kind == "set_mode" and session is not None:
                    session.set_mode(data.get("mode", "complete"))
                    await emit_then_voice(websocket, session, await asyncio.to_thread(session.start_turn))
                elif kind == "ready" and session is not None:
                    session.mark_ready()
                    await websocket.send_text(
                        json.dumps({"type": "state", "state": "listening", "feedback": "I’m listening. Your turn!"})
                    )
                elif kind == "idle" and session is not None:
                    events = await asyncio.to_thread(session.handle_idle)
                    if events:
                        await emit_then_voice(websocket, session, events)
                elif kind == "repeat" and session is not None:
                    await emit_then_voice(websocket, session, session.replay_prompt())
                elif kind == "stop" and session is not None:
                    session.stop_session()
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "state",
                                "state": "idle",
                                "feedback": "Paused. Pick a game whenever you’re ready.",
                                "prompt": "",
                            }
                        )
                    )
            elif message.get("bytes") is not None and session is not None:
                payload = message["bytes"]
                if payload[:4] == b"RIFF":
                    pcm, _sr = wav_bytes_to_pcm(payload)
                else:
                    pcm = int16_bytes_to_pcm(payload)
                utterance = await asyncio.to_thread(session.take_utterance, pcm, False)
                if utterance is not None:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "state",
                                "state": "thinking",
                                "feedback": "Teddy heard you. Thinking…",
                            }
                        )
                    )
                    events = await asyncio.to_thread(session.finish_utterance, utterance)
                    await emit_then_voice(websocket, session, events)
    except WebSocketDisconnect:
        print("WebSocket disconnected", flush=True)
        return
    except Exception as exc:
        print(f"WebSocket error: {exc}", flush=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        except Exception:
            return


if __name__ == "__main__":
    import uvicorn

    server = config.get("server", {})
    uvicorn.run(app, host=server.get("host", "0.0.0.0"), port=int(server.get("port", 8003)))
