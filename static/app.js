(() => {
  const $ = (id) => document.getElementById(id);
  const connEl = $("conn");
  const statusEl = $("status");
  const promptEl = $("prompt");
  const heardEl = $("heard");
  const feedbackEl = $("feedback");
  const streakEl = $("streak");
  const turnsEl = $("turns");
  const phoneEl = $("phone");
  const bearEl = $("bear");
  const stageEl = $("stage");
  const bannerEl = $("banner");
  const meter = $("meter");
  const meterCtx = meter.getContext("2d");
  const btnComplete = $("btn-complete");
  const btnMistake = $("btn-mistake");

  const TARGET_RATE = 16000;
  const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  let ws;
  let stream;
  let audioCtx;
  let captureTimer;
  let speaking = false;
  let playQueue = Promise.resolve();

  const LABELS = {
    idle: "Idle",
    listening: "Listening",
    thinking: "Thinking",
    speaking: "Teddy talking",
  };

  function showError(text) {
    bannerEl.textContent = text;
    bannerEl.classList.remove("hidden");
    feedbackEl.textContent = text;
  }

  function setStage(state) {
    const key = LABELS[state] ? state : "idle";
    bearEl.className = `bear ${key}`;
    stageEl.className = `stage ${key} mx-auto`;
    stageEl.textContent = LABELS[key];
  }

  function applyUi(msg) {
    if (msg.prompt) promptEl.textContent = msg.prompt;
    if (msg.stem) promptEl.textContent = `${msg.stem} …`;
    if (msg.broken) promptEl.textContent = msg.broken;
    if (msg.feedback) feedbackEl.textContent = msg.feedback;
    if (msg.heard) heardEl.textContent = msg.heard;
    if (typeof msg.streak === "number") streakEl.textContent = String(msg.streak);
    if (typeof msg.turns === "number") turnsEl.textContent = String(msg.turns);
    if (typeof msg.phoneme_score === "number") phoneEl.textContent = msg.phoneme_score.toFixed(2);
    if (msg.text && msg.type === "speak_text") statusEl.textContent = "Teddy is talking";
  }

  function drawMeter(rms) {
    const w = meter.width;
    const h = meter.height;
    meterCtx.clearRect(0, 0, w, h);
    meterCtx.fillStyle = "#e2e8f0";
    meterCtx.fillRect(0, 0, w, h);
    const bars = 16;
    const level = Math.min(1, rms * 18);
    for (let i = 0; i < bars; i += 1) {
      const on = i / bars < level;
      meterCtx.fillStyle = on ? "#0ea5e9" : "#cbd5e1";
      meterCtx.fillRect(6 + i * (w / bars), h - 8 - (on ? 18 : 6), w / bars - 6, on ? 22 : 8);
    }
  }

  async function playWav(bytes) {
    speaking = true;
    setStage("speaking");
    statusEl.textContent = "Teddy is talking";
    audioCtx = audioCtx || new AudioContext();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    const buffer = await audioCtx.decodeAudioData(bytes.slice(0));
    window.__teddyExpectAudio = false;
    await new Promise((resolve) => {
      const src = audioCtx.createBufferSource();
      src.buffer = buffer;
      src.connect(audioCtx.destination);
      src.onended = resolve;
      src.start();
    });
    speaking = false;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "ready" }));
    setStage("listening");
    statusEl.textContent = "Listening — your turn";
    feedbackEl.textContent = "Listening… speak now!";
  }

  function enqueueAudio(bytes) {
    playQueue = playQueue.then(() => playWav(bytes)).catch((err) => {
      speaking = false;
      showError(`Could not play Teddy: ${err}`);
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "ready" }));
    });
  }

  function connect() {
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      connEl.textContent = "Live over WebSocket";
      bannerEl.classList.add("hidden");
    };
    ws.onerror = () => showError("WebSocket error. Refresh the page.");
    ws.onclose = () => {
      connEl.textContent = "Reconnecting…";
      setTimeout(connect, 1200);
    };
    ws.onmessage = (ev) => {
      try {
        const payload = ev.data;
        if (payload instanceof ArrayBuffer) {
          enqueueAudio(payload);
          return;
        }
        if (payload instanceof Blob) {
          payload.arrayBuffer().then(enqueueAudio);
          return;
        }
        const msg = JSON.parse(String(payload));
        if (msg.type === "hello") {
          connEl.textContent = "Live over WebSocket";
          return;
        }
        if (msg.type === "audio") {
          window.__teddyExpectAudio = true;
          return;
        }
        if (msg.type === "state") {
          statusEl.textContent = LABELS[msg.state] || msg.state;
          setStage(msg.state);
          applyUi(msg);
        }
        if (msg.type === "ui" || msg.type === "speak_text") applyUi(msg);
        if (msg.type === "transcript") {
          heardEl.textContent = msg.text || "…";
          applyUi(msg);
        }
        if (msg.type === "error") {
          showError(msg.detail || "Server error");
          setStage("idle");
        }
      } catch (err) {
        showError(`Bad message from server: ${err}`);
      }
    };
  }

  function downsample(input, inRate) {
    if (inRate === TARGET_RATE) return input;
    const ratio = inRate / TARGET_RATE;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i += 1) out[i] = input[Math.floor(i * ratio)];
    return out;
  }

  function floatTo16(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i += 1) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  async function ensureMic() {
    if (stream) {
      audioCtx = audioCtx || new AudioContext();
      if (audioCtx.state === "suspended") await audioCtx.resume();
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser cannot use the microphone.");
    }
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      video: false,
    });
    audioCtx = audioCtx || new AudioContext();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    source.connect(analyser);
    analyser.connect(mute);
    mute.connect(audioCtx.destination);
    const scratch = new Float32Array(analyser.fftSize);
    captureTimer = window.setInterval(() => {
      if (!ws || ws.readyState !== 1 || speaking) return;
      analyser.getFloatTimeDomainData(scratch);
      const resampled = downsample(scratch, audioCtx.sampleRate);
      let sum = 0;
      for (let i = 0; i < resampled.length; i += 1) sum += resampled[i] * resampled[i];
      drawMeter(Math.sqrt(sum / resampled.length));
      ws.send(floatTo16(resampled).buffer);
    }, 250);
  }

  async function begin(mode) {
    setStage("thinking");
    statusEl.textContent = "Starting…";
    feedbackEl.textContent = "Sending start to Teddy…";
    bannerEl.classList.add("hidden");
    if (!ws || ws.readyState !== 1) {
      showError("Not connected yet. Wait until the badge says Live over WebSocket, then click again.");
      return;
    }
    ws.send(JSON.stringify({ type: "start", mode }));
    try {
      await ensureMic();
    } catch (err) {
      showError(`Microphone: ${err.message || err}. Allow mic access and click again.`);
    }
  }

  btnComplete.onclick = () => begin("complete");
  btnMistake.onclick = () => begin("mistake");
  connect();
})();
