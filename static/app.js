(() => {
  const $ = (id) => document.getElementById(id);
  const connEl = $("conn");
  const statusEl = $("status");
  const promptEl = $("prompt");
  const heardEl = $("heard");
  const feedbackEl = $("feedback");
  const streakEl = $("streak");
  const turnsEl = $("turns");
  const bestEl = $("best");
  const bearEl = $("bear");
  const stageEl = $("stage");
  const bannerEl = $("banner");
  const chatEl = $("chat");
  const meter = $("meter");
  const meterCtx = meter.getContext("2d");
  const nameEl = $("kid-name");
  const voicesEl = $("voices");

  const TARGET_RATE = 16000;
  const CHUNK_SAMPLES = Math.floor(TARGET_RATE * 0.25);
  const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  let ws;
  let stream;
  let audioCtx;
  let processor;
  let speaking = false;
  let sending = false;
  let muted = false;
  let currentSource = null;
  let playQueue = Promise.resolve();
  const pending = [];
  let pendingCount = 0;
  let selectedVoice = localStorage.getItem("teddyVoice") || "af_bella";
  if (selectedVoice === "af_nicole") selectedVoice = "bf_emma";
  let idleTimer = null;
  const IDLE_MS = 8000;

  const FALLBACK_VOICES = [
    { id: "af_bella", label: "Bella", blurb: "Warm and friendly" },
    { id: "bf_emma", label: "Emma", blurb: "Soft British lady" },
    { id: "af_sky", label: "Sky", blurb: "Bright and bouncy" },
    { id: "am_michael", label: "Michael", blurb: "Kind buddy" },
  ];

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

  function addBubble(who, text) {
    if (!text) return;
    const last = chatEl.lastElementChild;
    if (last && last.dataset.who === who && last.textContent === text) return;
    const div = document.createElement("div");
    div.className = `bubble ${who}`;
    div.dataset.who = who;
    div.textContent = text;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function setStage(state) {
    const key = LABELS[state] ? state : "idle";
    bearEl.className = `bear ${key}`;
    stageEl.className = `stage ${key}`;
    stageEl.textContent = LABELS[key];
    sending = key === "listening";
    if (sending) armIdle();
    else disarmIdle();
  }

  function disarmIdle() {
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  }

  function armIdle() {
    disarmIdle();
    idleTimer = setTimeout(() => {
      if (!sending || speaking) return;
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "idle" }));
    }, IDLE_MS);
  }

  function applyUi(msg) {
    if (msg.prompt) promptEl.textContent = msg.prompt;
    if (msg.stem) promptEl.textContent = `${msg.stem} …`;
    if (msg.broken) promptEl.textContent = msg.broken;
    if (msg.feedback) feedbackEl.textContent = msg.feedback;
    if (msg.heard) {
      heardEl.textContent = msg.heard;
      addBubble("kid", msg.heard);
    }
    if (typeof msg.streak === "number") streakEl.textContent = String(msg.streak);
    if (typeof msg.turns === "number") turnsEl.textContent = String(msg.turns);
    if (typeof msg.best === "number") bestEl.textContent = String(msg.best);
    if (msg.text) addBubble("teddy", msg.text);
  }

  function renderVoices(list) {
    const voices = list && list.length ? list : FALLBACK_VOICES;
    if (![...voices.map((v) => v.id)].includes(selectedVoice)) selectedVoice = voices[0].id;
    voicesEl.innerHTML = "";
    voices.forEach((voice) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `voice-btn${voice.id === selectedVoice ? " selected" : ""}`;
      btn.dataset.voice = voice.id;
      btn.innerHTML = `<strong>${voice.label}</strong><span>${voice.blurb || ""}</span>`;
      btn.onclick = () => {
        selectedVoice = voice.id;
        localStorage.setItem("teddyVoice", selectedVoice);
        voicesEl.querySelectorAll(".voice-btn").forEach((el) => el.classList.toggle("selected", el.dataset.voice === selectedVoice));
      };
      voicesEl.appendChild(btn);
    });
  }

  function drawMeter(rms) {
    const w = meter.width;
    const h = meter.height;
    meterCtx.clearRect(0, 0, w, h);
    meterCtx.fillStyle = "#e2e8f0";
    meterCtx.fillRect(0, 0, w, h);
    const bars = 18;
    const level = Math.min(1, rms * 16);
    for (let i = 0; i < bars; i += 1) {
      const on = i / bars < level;
      meterCtx.fillStyle = on ? "#0ea5e9" : "#cbd5e1";
      meterCtx.fillRect(8 + i * (w / bars), h - 10 - (on ? 22 : 8), w / bars - 6, on ? 26 : 10);
    }
  }

  async function playWav(bytes) {
    speaking = true;
    sending = false;
    setStage("speaking");
    statusEl.textContent = "Teddy is talking";
    if (muted) {
      speaking = false;
      if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "ready" }));
      setStage("listening");
      statusEl.textContent = "Your turn — speak";
      return;
    }
    audioCtx = audioCtx || new AudioContext();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    const buffer = await audioCtx.decodeAudioData(bytes.slice(0));
    window.__teddyExpectAudio = false;
    await new Promise((resolve) => {
      const src = audioCtx.createBufferSource();
      currentSource = src;
      src.buffer = buffer;
      src.connect(audioCtx.destination);
      src.onended = resolve;
      src.start();
    });
    currentSource = null;
    speaking = false;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "ready" }));
    setStage("listening");
    statusEl.textContent = "Your turn — speak";
    feedbackEl.textContent = "I’m listening.";
  }

  function haltVoice() {
    try {
      currentSource && currentSource.stop();
    } catch (err) {
      /* already stopped */
    }
    currentSource = null;
    speaking = false;
    playQueue = Promise.resolve();
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
      connEl.textContent = "Live";
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
          connEl.textContent = "Live";
          renderVoices(msg.voices);
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
          addBubble("kid", msg.text);
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
    let pos = 0;
    for (let i = 0; i < outLen; i += 1) {
      const next = Math.min(input.length - 1, (i + 1) * ratio);
      let acc = 0;
      let n = 0;
      while (pos < next) {
        acc += input[pos];
        pos += 1;
        n += 1;
      }
      out[i] = n ? acc / n : input[Math.min(input.length - 1, Math.floor(i * ratio))];
    }
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

  function flush() {
    if (!pendingCount) return;
    const merged = new Float32Array(pendingCount);
    let off = 0;
    pending.forEach((chunk) => {
      merged.set(chunk, off);
      off += chunk.length;
    });
    pending.length = 0;
    pendingCount = 0;
    let sum = 0;
    for (let i = 0; i < merged.length; i += 1) sum += merged[i] * merged[i];
    drawMeter(Math.sqrt(sum / merged.length));
    if (Math.sqrt(sum / merged.length) > 0.02) armIdle();
    if (!ws || ws.readyState !== 1 || speaking || !sending) return;
    ws.send(floatTo16(merged).buffer);
  }

  async function ensureMic() {
    if (stream) {
      audioCtx = audioCtx || new AudioContext();
      if (audioCtx.state === "suspended") await audioCtx.resume();
      return;
    }
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      video: false,
    });
    audioCtx = new AudioContext();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    const source = audioCtx.createMediaStreamSource(stream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (ev) => {
      const input = ev.inputBuffer.getChannelData(0);
      const resampled = downsample(input, audioCtx.sampleRate);
      pending.push(resampled);
      pendingCount += resampled.length;
      if (pendingCount >= CHUNK_SAMPLES) flush();
    };
    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    source.connect(processor);
    processor.connect(mute);
    mute.connect(audioCtx.destination);
    window.__teddyProcessor = processor;
  }

  async function begin(mode) {
    haltVoice();
    setStage("thinking");
    statusEl.textContent = "Starting…";
    feedbackEl.textContent = "Getting the first line ready…";
    bannerEl.classList.add("hidden");
    if (!ws || ws.readyState !== 1) {
      showError("Not connected yet. Wait until the badge says Live, then click again.");
      return;
    }
    ws.send(JSON.stringify({ type: "start", mode, name: (nameEl.value || "").trim(), voice: selectedVoice }));
    try {
      await ensureMic();
    } catch (err) {
      showError(`Microphone: ${err.message || err}. Allow mic access and click again.`);
    }
  }

  $("btn-complete").onclick = () => begin("complete");
  $("btn-mistake").onclick = () => begin("mistake");
  $("btn-repeat").onclick = () => {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "repeat" }));
  };
  $("btn-mute").onclick = () => {
    muted = !muted;
    $("btn-mute").classList.toggle("active", muted);
    $("btn-mute").textContent = muted ? "Unmute Teddy" : "Mute Teddy";
    if (muted) haltVoice();
  };
  $("btn-stop").onclick = () => {
    haltVoice();
    sending = false;
    setStage("idle");
    disarmIdle();
    statusEl.textContent = "Paused";
    promptEl.textContent = "";
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "stop" }));
  };
  nameEl.value = localStorage.getItem("teddyName") || "";
  nameEl.addEventListener("change", () => localStorage.setItem("teddyName", nameEl.value.trim()));
  nameEl.addEventListener("blur", () => localStorage.setItem("teddyName", nameEl.value.trim()));
  renderVoices(FALLBACK_VOICES);
  connect();
})();
