/* ============================================================
   Application Logic
   ============================================================ */

// --- State ---
let mediaRecorder;
let audioChunks = [];
let ws;
let isRecording = false;
let stream;
let lastSentPayload = null;
let wsReconnectAttempts = 0;
const WS_MAX_RECONNECT = 5;
const WS_BASE_DELAY = 1000;

// --- DOM References ---
const $ = (id) => document.getElementById(id);

const ringArea = document.querySelector('.ring-area');
const ring = $('assistantRing');
const statusEl = $('status');
const startBtn = $('startBtn');
const stopBtn = $('stopBtn');
const transcriptEl = $('transcript');
const responseEl = $('response');
const errorBox = $('errorBox');
const transcriptBox = $('transcriptBox');
const responseBox = $('responseBox');
const stateInfo = $('stateInfo');
const turnCount = $('turnCount');
const modeBadge = $('modeBadge');
const thinkingIndicator = $('thinkingIndicator');
const thinkingText = $('thinkingText');
const emotionHistory = [];

// ============================================================
// SESSION PERSISTENCE (localStorage)
// ============================================================

function generateSessionId() {
  return 'session_' + Math.random().toString(36).substring(2, 15);
}

function getOrCreateSessionId() {
  let id = localStorage.getItem('currentSessionId');
  if (!id) {
    id = generateSessionId();
    localStorage.setItem('currentSessionId', id);
  }
  return id;
}

let sessionId = getOrCreateSessionId();

function getLocalSessions() {
  try { return JSON.parse(localStorage.getItem('chatSessions') || '[]'); }
  catch { return []; }
}

function saveLocalSessions(sessions) {
  localStorage.setItem('chatSessions', JSON.stringify(sessions));
}

function upsertLocalSession(id, title, turns) {
  const sessions = getLocalSessions();
  const existing = sessions.find((s) => s.session_id === id);
  if (existing) {
    if (title) existing.title = title;
    if (turns !== undefined) existing.turn_count = turns;
    existing.last_updated = new Date().toISOString();
  } else {
    sessions.unshift({
      session_id: id, title: title || 'New Conversation',
      turn_count: turns || 0,
      created_at: new Date().toISOString(), last_updated: new Date().toISOString(),
    });
  }
  sessions.sort((a, b) => (b.last_updated || '').localeCompare(a.last_updated || ''));
  saveLocalSessions(sessions.slice(0, 50));
}

function removeLocalSession(id) {
  saveLocalSessions(getLocalSessions().filter((s) => s.session_id !== id));
}

// ============================================================
// SIDEBAR
// ============================================================

function renderSidebar() {
  let sessions = getLocalSessions();
  const sessionList = $('sessionList');
  const searchInput = $('sessionSearchInput');

  if (searchInput && searchInput.value.trim()) {
    const q = searchInput.value.toLowerCase().trim();
    sessions = sessions.filter((s) => (s.title || '').toLowerCase().includes(q));
  }

  if (sessions.length === 0) {
    const isSearching = searchInput && searchInput.value.trim();
    sessionList.innerHTML = `<div class="session-empty">${isSearching ? 'No matches' : 'No conversations yet'}</div>`;
    return;
  }

  sessionList.innerHTML = sessions.map((s) => {
    const isActive = s.session_id === sessionId;
    const date = new Date(s.last_updated);
    const timeStr = date.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    const title = s.title || 'New Conversation';
    const turns = s.turn_count || 0;
    return `
      <div class="session-item ${isActive ? 'active' : ''}" onclick="switchToSession('${s.session_id}')">
        <div class="session-info">
          <div class="session-title">${escapeHtml(title)}</div>
          <div class="session-meta">${timeStr} · ${turns} turn${turns !== 1 ? 's' : ''}</div>
        </div>
        <button class="delete-btn" onclick="deleteSession(event, '${s.session_id}')" title="Delete">✕</button>
      </div>`;
  }).join('');
}

async function loadSessions() {
  renderSidebar();
  try {
    const resp = await fetch('/api/sessions');
    if (resp.ok) {
      const data = await resp.json();
      if (data.sessions && data.sessions.length > 0) {
        for (const s of data.sessions) upsertLocalSession(s.session_id, s.title, s.turn_count);
        renderSidebar();
      }
    }
  } catch { /* localStorage sidebar works offline */ }
}

// ============================================================
// SESSION MANAGEMENT
// ============================================================

async function createNewSession() {
  try {
    const resp = await fetch('/api/sessions/new', { method: 'POST' });
    const data = await resp.json();
    sessionId = data.session_id;
  } catch { sessionId = 'local_' + Date.now(); }
  localStorage.setItem('currentSessionId', sessionId);
  upsertLocalSession(sessionId, 'New Conversation', 0);
  clearConversationUI();
  renderSidebar();
  showStatus('New session created', 'success');
}

function switchToSession(newId) {
  if (newId === sessionId) return;
  sessionId = newId;
  localStorage.setItem('currentSessionId', sessionId);
  clearConversationUI();
  renderSidebar();
}

async function deleteSession(event, id) {
  event.stopPropagation();
  if (!confirm('Delete this conversation?')) return;
  removeLocalSession(id);
  try { await fetch(`/api/sessions/${id}`, { method: 'DELETE' }); } catch {}
  if (id === sessionId) await createNewSession();
  else renderSidebar();
  showStatus('Session deleted', 'success');
}

function clearConversationUI() {
  transcriptEl.textContent = '';
  responseEl.textContent = '';
  transcriptBox.classList.add('hidden');
  responseBox.classList.add('hidden');
  stateInfo.classList.add('hidden');
  hideThinking();
}

// ============================================================
// SEARCH / EXPORT / DEBUG
// ============================================================

function filterSessions() { renderSidebar(); }

function exportConversation() {
  const sessions = getLocalSessions();
  const active = sessions.find((s) => s.session_id === sessionId);
  const exportData = {
    exported_at: new Date().toISOString(),
    session: active || { session_id: sessionId, title: 'Unknown' },
    all_sessions: sessions,
  };
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `voice_assistant_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showStatus('Exported!', 'success');
}

function toggleDebugPanels() {
  const section = $('debugSection');
  section.style.display = section.style.display === 'none' ? '' : 'none';
}

// ============================================================
// STATUS / THINKING
// ============================================================

function showError(message) {
  const text = message.startsWith('❌') ? message : '❌ ' + message;
  errorBox.textContent = text;
  errorBox.className = 'toast error';
  errorBox.classList.remove('hidden');
  setTimeout(() => errorBox.classList.add('hidden'), 5000);
}

function showStatus(message, type = 'info') {
  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  errorBox.textContent = (icons[type] || '') + ' ' + message;
  errorBox.className = `toast ${type}`;
  errorBox.classList.remove('hidden');
  setTimeout(() => errorBox.classList.add('hidden'), 3000);
}

function showThinking(text) {
  thinkingText.textContent = text || 'Thinking...';
  thinkingIndicator.classList.remove('hidden');
}

function hideThinking() {
  thinkingIndicator.classList.add('hidden');
}

// ============================================================
// MARKDOWN STRIPPING
// ============================================================

function stripMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\*{3}(.+?)\*{3}/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/_{3}(.+?)_{3}/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/(?<!\w)_(.+?)_(?!\w)/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .trim();
}

// ============================================================
// PROGRESSIVE TEXT REVEAL
// ============================================================

async function streamText(element, text) {
  const clean = stripMarkdown(text);
  element.textContent = '';
  const words = clean.split(/(\s+)/);
  for (let i = 0; i < words.length; i++) {
    element.textContent += words[i];
    const delay = words[i].trim() ? Math.random() * 18 + 8 : 4;
    await new Promise((r) => setTimeout(r, delay));
  }
}

// ============================================================
// BROWSER TTS FALLBACK
// ============================================================

function browserSpeak(text, onEnd) {
  if (!('speechSynthesis' in window) || !text) { if (onEnd) onEnd(); return; }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(stripMarkdown(text));
  utterance.rate = 1.0;
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find((v) => v.lang.startsWith('en')) || voices[0];
  if (preferred) utterance.voice = preferred;
  setRingState('speaking');
  statusEl.textContent = 'Speaking...';
  utterance.onend = () => { if (onEnd) onEnd(); };
  utterance.onerror = () => { if (onEnd) onEnd(); };
  window.speechSynthesis.speak(utterance);
}
if ('speechSynthesis' in window) {
  window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}

// ============================================================
// RING STATE
// ============================================================

function setRingState(state) {
  ringArea.className = 'ring-area';
  if (state) ringArea.classList.add(state);
}

// ============================================================
// EMOTION SPARKLINE
// ============================================================

function drawSparkline() {
  const canvas = $('emotionSparkline');
  if (!canvas || emotionHistory.length < 2) return;
  const container = $('sparklineContainer');
  container.style.display = 'block';
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, pad = 4;
  ctx.clearRect(0, 0, w, h);

  const vals = emotionHistory.map((d) => (d.valence + 1) / 2);
  const strs = emotionHistory.map((d) => d.stress);

  function drawLine(data, color) {
    const step = (w - pad * 2) / Math.max(data.length - 1, 1);
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.lineJoin = 'round';
    for (let i = 0; i < data.length; i++) {
      const x = pad + i * step;
      const y = h - pad - data[i] * (h - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  drawLine(vals, '#34d399');
  drawLine(strs, '#f87171');
}

// ============================================================
// CIRCULAR EMOTION RING UPDATE
// ============================================================

function updateEmotionRing(ringId, value, maxVal) {
  const el = $(ringId);
  if (!el) return;
  const pct = Math.min(Math.abs(value) / maxVal, 1);
  const offset = 100 - (pct * 100);
  el.style.strokeDashoffset = offset;
  // Color based on value
  if (value > 0.3) el.style.stroke = '#34d399';
  else if (value < -0.3 || value > 0.6) el.style.stroke = '#f87171';
  else el.style.stroke = '#8b5cf6';
}

// ============================================================
// UI UPDATE HELPERS
// ============================================================

function updateState(data) {
  stateInfo.classList.remove('hidden');
  const s = data.adaptive_state;

  const val = s.valence;
  $('valence').textContent = val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
  updateEmotionRing('valenceRing', (val + 1) / 2, 1);
  $('valenceEmoji').textContent = val > 0.3 ? '😊' : val < -0.3 ? '😔' : '😐';

  const ar = s.arousal;
  $('arousal').textContent = `${(ar * 100).toFixed(0)}%`;
  updateEmotionRing('arousalRing', ar, 1);
  $('arousalEmoji').textContent = ar > 0.6 ? '🔥' : ar > 0.3 ? '⚡' : '😴';

  const cl = s.clarity;
  $('clarity').textContent = `${(cl * 100).toFixed(0)}%`;
  updateEmotionRing('clarityRing', cl, 1);
  $('clarityEmoji').textContent = cl > 0.7 ? '💡' : cl > 0.4 ? '🤔' : '❓';

  const st = s.stress;
  $('stress').textContent = `${(st * 100).toFixed(0)}%`;
  updateEmotionRing('stressRing', st, 1);
  $('stressEmoji').textContent = st > 0.6 ? '😰' : st > 0.3 ? '😐' : '😌';

  turnCount.textContent = data.turn_count;

  if (data.emotional_mode === 'trend') {
    modeBadge.innerHTML = '<span class="mode-dot"></span>Trend';
    modeBadge.className = 'mode-pill trend';
  } else {
    modeBadge.innerHTML = '<span class="mode-dot"></span>Instant';
    modeBadge.className = 'mode-pill';
  }
}

function updateTrendIndicator(elementId, trend) {
  const el = $(elementId);
  if (!el) return;
  if (trend > 0.05) { el.textContent = '↗'; el.className = 'trend trend-up'; }
  else if (trend < -0.05) { el.textContent = '↘'; el.className = 'trend trend-down'; }
  else { el.textContent = '→'; el.className = 'trend trend-stable'; }
}

function updateMemoryView(data) {
  if (!data.memory_view) return;
  const mem = data.memory_view;

  $('memSessionId').textContent = mem.session_id.substring(0, 12) + '…';
  $('memTurnCount').textContent = mem.dialogue_state.turn_count;
  $('memTopic').textContent = mem.topic_info.current_topic;
  $('memTopicConf').textContent = (mem.topic_info.confidence * 100).toFixed(0) + '%';
  $('memMode').textContent = mem.emotional_trends.mode;
  $('memModeConf').textContent = (mem.emotional_trends.confidence * 100).toFixed(0) + '%';

  const valence = mem.emotional_trends.valence;
  $('memValence').textContent = valence.current.toFixed(2);
  updateTrendIndicator('memValenceTrend', valence.trend);
  const arousal = mem.emotional_trends.arousal;
  $('memArousal').textContent = arousal.current.toFixed(2);
  updateTrendIndicator('memArousalTrend', arousal.trend);
  const stress = mem.emotional_trends.stress;
  $('memStress').textContent = stress.current.toFixed(2);
  updateTrendIndicator('memStressTrend', stress.trend);
  const clarity = mem.emotional_trends.clarity;
  $('memClarity').textContent = clarity.current.toFixed(2);
  updateTrendIndicator('memClarityTrend', clarity.trend);

  emotionHistory.push({ valence: valence.current, stress: stress.current, turn: data.turn_count });
  if (emotionHistory.length > 20) emotionHistory.shift();
  drawSparkline();

  const turnsContainer = $('memRecentTurns');
  if (mem.recent_turns.length > 0) {
    turnsContainer.innerHTML = mem.recent_turns.map((turn, idx) => {
      const turnNum = data.turn_count - mem.recent_turns.length + idx + 1;
      return `<div class="debug-turn">
        <div class="debug-turn-meta">Turn ${turnNum} · ${turn.topic}</div>
        <div class="debug-turn-user">👤 ${escapeHtml(turn.user)}</div>
        <div class="debug-turn-ai">🤖 ${escapeHtml(turn.ai)}</div>
      </div>`;
    }).join('');
  }

  if (mem.dialogue_state.recent_intents.length > 0) {
    $('memIntents').textContent = mem.dialogue_state.recent_intents.slice(-3).join(', ');
  }
  $('memCoherence').textContent = mem.dialogue_state.coherence_score.toFixed(2);

  if (mem.long_term_memory) {
    const stats = mem.long_term_memory.stats;
    $('memEpisodic').textContent = stats.episodic_count || 0;
    $('memSemantic').textContent = stats.semantic_count || 0;
    $('memLastRetrieval').textContent = mem.long_term_memory.last_retrieval || 'N/A';
  }
}

function updateModelOutputs(data) {
  if (!data.model_outputs) return;
  const m = data.model_outputs;

  if (m.asr) {
    $('modelASR').innerHTML = `
      <div class="debug-row"><span class="dl">Text</span><span class="dv">${escapeHtml(m.asr.transcript.substring(0, 40))}…</span></div>
      <div class="debug-row"><span class="dl">Latency</span><span class="dv">${m.asr.latency_ms.toFixed(0)}ms</span></div>`;
  }

  if (m.text_emotion && m.text_emotion.top_emotions) {
    let html = '';
    for (const [emotion, score] of Object.entries(m.text_emotion.top_emotions)) {
      html += `<div class="debug-row"><span class="dl">${emotion}</span><span class="dv">${(score * 100).toFixed(1)}%</span></div>
        <div class="ebar"><div class="ebar-fill" style="width:${score * 100}%"></div></div>`;
    }
    $('modelTextEmotion').innerHTML = html;
  }

  if (m.audio_analysis && m.audio_analysis.ser) {
    const ser = m.audio_analysis.ser;
    $('modelSER').innerHTML = `
      <div class="debug-row"><span class="dl">Angry</span><span class="dv">${(ser.angry * 100).toFixed(1)}%</span></div>
      <div class="debug-row"><span class="dl">Happy</span><span class="dv">${(ser.happy * 100).toFixed(1)}%</span></div>
      <div class="debug-row"><span class="dl">Neutral</span><span class="dv">${(ser.neutral * 100).toFixed(1)}%</span></div>
      <div class="debug-row"><span class="dl">Sad</span><span class="dv">${(ser.sad * 100).toFixed(1)}%</span></div>`;
  }

  if (m.audio_analysis && m.audio_analysis.ast) {
    const ast = m.audio_analysis.ast;
    if (Object.keys(ast).length > 0) {
      let html = '';
      for (const [event, score] of Object.entries(ast)) {
        html += `<div class="debug-row"><span class="dl">${event}</span><span class="dv">${(score * 100).toFixed(1)}%</span></div>`;
      }
      $('modelAST').innerHTML = html;
    } else {
      $('modelAST').innerHTML = '<div class="debug-row"><span>No events</span></div>';
    }
  }

  if (m.acoustic_features) {
    const a = m.acoustic_features;
    $('modelAcoustic').innerHTML = `
      <div class="debug-row"><span class="dl">Rate</span><span class="dv">${a.speech_rate.toFixed(1)} WPS</span></div>
      <div class="debug-row"><span class="dl">Pauses</span><span class="dv">${a.pause_duration.toFixed(2)}s</span></div>
      <div class="debug-row"><span class="dl">Jitter</span><span class="dv">${a.jitter.toFixed(3)}</span></div>
      <div class="debug-row"><span class="dl">Words</span><span class="dv">${a.word_count}</span></div>
      <div class="debug-row"><span class="dl">Duration</span><span class="dv">${a.audio_duration.toFixed(1)}s</span></div>`;
  }

  if (m.fusion && m.fusion.instant_state) {
    const s = m.fusion.instant_state;
    $('modelFusion').innerHTML = `
      <div class="debug-row"><span class="dl">Valence</span><span class="dv">${s.valence.toFixed(3)}</span></div>
      <div class="debug-row"><span class="dl">Arousal</span><span class="dv">${s.arousal.toFixed(3)}</span></div>
      <div class="debug-row"><span class="dl">Stress</span><span class="dv">${s.stress.toFixed(3)}</span></div>
      <div class="debug-row"><span class="dl">Clarity</span><span class="dv">${s.clarity.toFixed(3)}</span></div>`;
  }
}

// ============================================================
// WEBSOCKET
// ============================================================

function connectWebSocket() {
  return new Promise((resolve, reject) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/conversation`);

    ws.onopen = () => { wsReconnectAttempts = 0; resolve(ws); };

    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.error) {
          hideThinking(); showError(data.error); resetUI(); lastSentPayload = null;
          return;
        }

        // --- thinking ---
        if (data.status === 'thinking') {
          showThinking('Analyzing your voice...');
          setRingState('processing');
          statusEl.textContent = 'Listening...';
          return;
        }

        // --- transcript ready ---
        if (data.status === 'transcript') {
          hideThinking();
          showThinking('Generating response...');
          lastSentPayload = null;
          transcriptBox.classList.remove('hidden');
          transcriptEl.textContent = data.transcript;
          responseBox.classList.remove('hidden');
          responseEl.textContent = '';
          if (data.adaptive_state) updateState(data);
          if (data.model_outputs) updateModelOutputs(data);
          statusEl.textContent = 'Generating...';
          return;
        }

        // --- response text ready (no audio yet) ---
        if (data.status === 'response_text') {
          hideThinking();
          transcriptBox.classList.remove('hidden');
          transcriptEl.textContent = data.transcript;
          responseBox.classList.remove('hidden');
          streamText(responseEl, data.llm_reply);
          updateState(data);
          updateMemoryView(data);
          updateModelOutputs(data);
          upsertLocalSession(sessionId, data.transcript, data.turn_count);
          renderSidebar();
          statusEl.textContent = 'Generating audio...';
          setRingState('processing');
          return;
        }

        // --- audio ready ---
        if (data.status === 'audio_ready') {
          hideThinking(); lastSentPayload = null;
          if (data.tts_audio) {
            playAzureAudio(data.tts_audio, responseEl.textContent || responseEl.innerText);
          } else {
            browserSpeak(responseEl.textContent || responseEl.innerText, () => {
              resetUI(); ws.close();
            });
          }
          return;
        }

        // --- fallback: legacy full response ---
        hideThinking(); lastSentPayload = null;
        transcriptBox.classList.remove('hidden');
        transcriptEl.textContent = data.transcript;
        responseBox.classList.remove('hidden');
        streamText(responseEl, data.llm_reply);
        updateState(data);
        updateMemoryView(data);
        updateModelOutputs(data);
        upsertLocalSession(sessionId, data.transcript, data.turn_count);
        renderSidebar();
        if (data.tts_audio) {
          playAzureAudio(data.tts_audio, data.llm_reply);
        } else {
          browserSpeak(data.llm_reply, () => { resetUI(); ws.close(); });
        }
      } catch (e) {
        console.error('WS error:', e);
        hideThinking(); showError('Error processing response'); resetUI();
      }
    };

    ws.onerror = () => reject(new Error('WebSocket failed'));
    ws.onclose = (event) => {
      if (lastSentPayload && wsReconnectAttempts < WS_MAX_RECONNECT) {
        const delay = WS_BASE_DELAY * Math.pow(2, wsReconnectAttempts);
        wsReconnectAttempts++;
        statusEl.textContent = `Reconnecting (${wsReconnectAttempts}/${WS_MAX_RECONNECT})...`;
        setRingState('processing');
        setTimeout(async () => {
          try {
            await connectWebSocket();
            ws.send(lastSentPayload);
          } catch {
            showError('Connection lost');
            lastSentPayload = null; resetUI();
          }
        }, delay);
      } else if (!lastSentPayload) { resetUI(); }
    };
  });
}

function playAzureAudio(base64Audio, fallbackText) {
  const audioData = atob(base64Audio);
  const audioArray = new Uint8Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) audioArray[i] = audioData.charCodeAt(i);
  const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
  const audioUrl = URL.createObjectURL(audioBlob);
  const audio = new Audio(audioUrl);
  setRingState('speaking');
  statusEl.textContent = 'Speaking...';
  audio.play().catch(() => {
    browserSpeak(fallbackText, () => { resetUI(); ws.close(); });
  });
  audio.onended = () => { resetUI(); ws.close(); };
}

function resetUI() {
  setRingState('');
  statusEl.textContent = 'Ready to listen';
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// ============================================================
// RECORDING
// ============================================================

async function startRecording() {
  if (isRecording) return;
  try {
    transcriptBox.classList.add('hidden');
    responseBox.classList.add('hidden');
    hideThinking();

    if (!ws || ws.readyState !== WebSocket.OPEN) await connectWebSocket();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const arrayBuffer = await audioBlob.arrayBuffer();
      setRingState('processing');
      statusEl.textContent = 'Processing...';
      const base64Audio = btoa(new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), ''));
      const selectedVoice = $('voiceSelect').value;
      const payload = JSON.stringify({ audio: base64Audio, session_id: sessionId, voice_name: selectedVoice });
      lastSentPayload = payload;
      ws.send(payload);
      stream.getTracks().forEach((track) => track.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    setRingState('listening');
    ringArea.classList.add('active');
    statusEl.textContent = 'Listening... (press Stop when done)';
    startBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (error) {
    console.error('Recording error:', error);
    showError('Microphone access denied or connection failed');
    resetUI();
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  mediaRecorder.stop();
  isRecording = false;
  ringArea.classList.remove('active');
  stopBtn.disabled = true;
}

// ============================================================
// UTILITIES
// ============================================================

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space' && !e.repeat) {
    e.preventDefault();
    if (isRecording) stopRecording();
    else if (!startBtn.disabled) startRecording();
  }
  if (e.code === 'Escape' && isRecording) {
    stopRecording();
    showError('Recording cancelled');
  }
});

// ============================================================
// INIT
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  startBtn.addEventListener('click', startRecording);
  stopBtn.addEventListener('click', stopRecording);
  renderSidebar();
  loadSessions();
});
