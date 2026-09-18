import { useCallback, useEffect, useRef, useState } from 'react'

/* ─────────────────────────────────────────────────────────────
   Types — mirror the WebSocket protocol implemented in main.py
   and agent.py (see README.md "Protocol" section).
───────────────────────────────────────────────────────────── */
export type Phase = 'idle' | 'listening' | 'thinking' | 'speaking'
export type GameMode = 'complete' | 'story'

export interface VoiceOption {
  id: string
  label: string
  blurb: string
}

export interface UiSnapshot {
  mode?: GameMode
  phase?: string
  streak?: number
  turns?: number
  best?: number
  child_name?: string
  voice?: string
  item_id?: string
  stem?: string
  prompt?: string
  situation?: string
  question?: string
  skill?: string
  difficulty?: string
  feedback?: string
  correct?: boolean
  heard?: string
  word_score?: number
  phoneme_score?: number
  issue?: string
  idle?: boolean
  unclear?: boolean
  give_up?: boolean
  safety_label?: string
}

export interface TeddySessionState {
  connected: boolean
  voices: VoiceOption[]
  phase: Phase
  ui: UiSnapshot
  error: string | null
  micLevel: number
}

export interface TeddySessionActions {
  start: (mode: GameMode, name: string, voiceId: string) => void
  repeat: () => void
  stop: () => void
}

/* ─────────────────────────────────────────────────────────────
   Audio helpers — ported from static/app.js
───────────────────────────────────────────────────────────── */
const TARGET_RATE = 16000
const CHUNK_SAMPLES = Math.floor(TARGET_RATE * 0.25)
const IDLE_MS = 8000

function downsample(input: Float32Array, inRate: number): Float32Array {
  if (inRate === TARGET_RATE) return input
  const ratio = inRate / TARGET_RATE
  const outLen = Math.floor(input.length / ratio)
  const out = new Float32Array(outLen)
  let pos = 0
  for (let i = 0; i < outLen; i += 1) {
    const next = Math.min(input.length - 1, (i + 1) * ratio)
    let acc = 0
    let n = 0
    while (pos < next) {
      acc += input[pos]
      pos += 1
      n += 1
    }
    out[i] = n ? acc / n : input[Math.min(input.length - 1, Math.floor(i * ratio))]
  }
  return out
}

function floatTo16(f32: Float32Array): Int16Array {
  const out = new Int16Array(f32.length)
  for (let i = 0; i < f32.length; i += 1) {
    const s = Math.max(-1, Math.min(1, f32[i]))
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return out
}

/* ─────────────────────────────────────────────────────────────
   useTeddySession — owns the socket, mic capture, and TTS
   playback; exposes reactive state + a small action API.
───────────────────────────────────────────────────────────── */
export function useTeddySession(): [TeddySessionState, TeddySessionActions] {
  const [connected, setConnected] = useState(false)
  const [voices, setVoices] = useState<VoiceOption[]>([])
  const [phase, setPhase] = useState<Phase>('idle')
  const [ui, setUi] = useState<UiSnapshot>({})
  const [error, setError] = useState<string | null>(null)
  const [micLevel, setMicLevel] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sendingRef = useRef(false)
  const pendingRef = useRef<Float32Array[]>([])
  const pendingCountRef = useRef(0)
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null)

  const send = useCallback((payload: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload))
    }
  }, [])

  const disarmIdle = useCallback(() => {
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current)
      idleTimerRef.current = null
    }
  }, [])

  const armIdle = useCallback(() => {
    disarmIdle()
    idleTimerRef.current = setTimeout(() => {
      if (!sendingRef.current) return
      send({ type: 'idle' })
    }, IDLE_MS)
  }, [disarmIdle, send])

  const flush = useCallback(() => {
    if (!pendingCountRef.current) return
    const merged = new Float32Array(pendingCountRef.current)
    let off = 0
    pendingRef.current.forEach((chunk) => {
      merged.set(chunk, off)
      off += chunk.length
    })
    pendingRef.current = []
    pendingCountRef.current = 0
    let sum = 0
    for (let i = 0; i < merged.length; i += 1) sum += merged[i] * merged[i]
    const rms = Math.sqrt(sum / merged.length)
    setMicLevel(rms)
    if (rms > 0.02) armIdle()
    if (!sendingRef.current || wsRef.current?.readyState !== WebSocket.OPEN) return
    wsRef.current.send(floatTo16(merged).buffer)
  }, [armIdle])

  const ensureMic = useCallback(async () => {
    if (micStreamRef.current) {
      if (audioCtxRef.current?.state === 'suspended') await audioCtxRef.current.resume()
      return
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      video: false,
    })
    micStreamRef.current = stream
    const ctx = audioCtxRef.current ?? new AudioContext()
    audioCtxRef.current = ctx
    if (ctx.state === 'suspended') await ctx.resume()
    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    processor.onaudioprocess = (ev) => {
      const input = ev.inputBuffer.getChannelData(0)
      const resampled = downsample(input, ctx.sampleRate)
      pendingRef.current.push(resampled)
      pendingCountRef.current += resampled.length
      if (pendingCountRef.current >= CHUNK_SAMPLES) flush()
    }
    const mute = ctx.createGain()
    mute.gain.value = 0
    source.connect(processor)
    processor.connect(mute)
    mute.connect(ctx.destination)
    processorRef.current = processor
  }, [flush])

  const playWav = useCallback(async (bytes: ArrayBuffer) => {
    const ctx = audioCtxRef.current ?? new AudioContext()
    audioCtxRef.current = ctx
    if (ctx.state === 'suspended') await ctx.resume()
    const buffer = await ctx.decodeAudioData(bytes.slice(0))
    await new Promise<void>((resolve) => {
      const src = ctx.createBufferSource()
      currentSourceRef.current = src
      src.buffer = buffer
      src.connect(ctx.destination)
      src.onended = () => resolve()
      src.start()
    })
    currentSourceRef.current = null
  }, [])

  useEffect(() => {
    let cancelled = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`
      const ws = new WebSocket(wsUrl)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onerror = () => setError('WebSocket error. Refresh the page.')
      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) reconnectTimer = setTimeout(connect, 1200)
      }
      ws.onmessage = async (ev) => {
        const payload = ev.data
        if (payload instanceof ArrayBuffer) {
          sendingRef.current = false
          disarmIdle()
          await playWav(payload)
          send({ type: 'ready' })
          sendingRef.current = true
          setPhase('listening')
          return
        }
        let msg: Record<string, unknown>
        try {
          msg = JSON.parse(String(payload))
        } catch {
          return
        }
        if (msg.type === 'hello') {
          setVoices((msg.voices as VoiceOption[]) ?? [])
          return
        }
        if (msg.type === 'audio') return // binary WAV follows next
        if (msg.type === 'state' || msg.type === 'ui') {
          setUi((prev) => {
            const next: UiSnapshot = { ...prev, ...(msg as UiSnapshot) }
            if (msg.item_id && msg.item_id !== prev.item_id) delete next.correct
            return next
          })
          if (msg.type === 'state') {
            const state = msg.state as Phase
            setPhase(state)
            if (state !== 'listening') {
              sendingRef.current = false
              disarmIdle()
            }
          }
          return
        }
        if (msg.type === 'transcript') {
          setUi((prev) => ({ ...prev, heard: msg.text as string }))
          return
        }
        if (msg.type === 'error') {
          setError((msg.detail as string) ?? 'Server error')
        }
      }
    }
    connect()
    return () => {
      cancelled = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [playWav, send, disarmIdle])

  const start = useCallback(
    (mode: GameMode, name: string, voiceId: string) => {
      setError(null)
      ensureMic().catch((err) => setError(`Microphone: ${err instanceof Error ? err.message : String(err)}`))
      send({ type: 'start', mode, name, voice: voiceId })
    },
    [ensureMic, send],
  )

  const repeat = useCallback(() => send({ type: 'repeat' }), [send])
  const stop = useCallback(() => {
    sendingRef.current = false
    disarmIdle()
    send({ type: 'stop' })
  }, [send, disarmIdle])

  return [
    { connected, voices, phase, ui, error, micLevel },
    { start, repeat, stop },
  ]
}
