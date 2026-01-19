<template>
  <div class="container">
    <div class="header" style="justify-content: flex-end;">
      <button class="settings-btn" @click="showSettings = !showSettings">⚙️</button>
    </div>
    
    <!-- Settings Area -->
    <div v-if="showSettings" class="settings-panel">
      <div class="form-group">
        <label>Backend URL:</label>
        <input v-model="backendUrl" @change="saveSettings" placeholder="http://localhost:8000" />
      </div>
      <div class="form-group">
        <label>JWT Token:</label>
        <input v-model="jwtToken" @change="saveSettings" placeholder="Bearer token..." type="password" />
      </div>
      <div class="form-group">
        <label>Voice:</label>
        <select v-model="selectedVoice" @change="saveSettings">
          <option v-for="voice in voiceList" :key="voice.id" :value="voice.id">
            {{ voice.name }}
          </option>
        </select>
      </div>
      <div class="form-group">
        <label>Speed:</label>
        <select v-model="selectedRate" @change="saveSettings">
          <option value="-50%">-50% (Slow)</option>
          <option value="-25%">-25%</option>
          <option value="+0%">Normal</option>
          <option value="+25%">+25%</option>
          <option value="+50%">+50% (Fast)</option>
        </select>
      </div>
      <div class="form-group">
        <label>Preload Buffer (1-10):</label>
        <input type="number" v-model.number="preloadCount" min="1" max="10" @change="saveSettings" />
      </div>
      <button @click="fetchVoices" :disabled="isLoadingVoices">
        {{ isLoadingVoices ? 'Refreshing Voices...' : 'Refresh Voice List' }}
      </button>
    </div>

    <div class="controls">
      <button @click="readPage" :disabled="isPlaying || isLoading">
        {{ isLoading ? 'Loading...' : 'Read Page' }}
      </button>
      <button @click="togglePause" :disabled="!isPlaying && !isPaused">
        {{ isPaused ? 'Resume' : 'Pause' }}
      </button>
      <button @click="stopPlayback" :disabled="!isPlaying && !isPaused">Stop</button>
    </div>

    <div class="status" v-if="status">
      {{ status }}
    </div>

    <div class="error" v-if="error">
      {{ error }}
    </div>

    <div class="chunks" v-if="chunks.length">
      <div 
        v-for="(chunk, index) in chunks" 
        :key="index"
        :ref="(el) => setChunkRef(el, index)"
        :class="{ active: currentChunkIndex === index }"
        class="chunk-item"
        @click="playChunk(index)"
      >
        <span class="chunk-status">
            <span v-if="chunk.status === 'playing'">🔊</span>
            <span v-else-if="chunk.status === 'loading'">⏳</span>
            <span v-else-if="chunk.status === 'ready'">✅</span>
            <span v-else-if="chunk.status === 'done'">✔️</span>
            <span v-else-if="chunk.status === 'failed'" 
                  @click.stop="retryChunk(index)" 
                  class="retry-btn" 
                  title="Retry download">🔄</span>
            <span v-else>⚪</span>
        </span>
        <span class="chunk-text">{{ chunk.text }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue';

interface Chunk {
  id: number;
  text: string;
  status: 'pending' | 'loading' | 'ready' | 'playing' | 'done' | 'failed';
}

interface Voice {
  id: string;
  name: string;
  engine: string;
}

// State
const chunks = ref<Chunk[]>([]);
const currentChunkIndex = ref(-1);
const isPlaying = ref(false);
const isPaused = ref(false);
const isLoading = ref(false);
const status = ref('');
const error = ref('');

// Scroll Refs
const chunkRefs = ref<Record<number, HTMLElement>>({});
const setChunkRef = (el: any, index: number) => {
    if (el) chunkRefs.value[index] = el as HTMLElement;
};

// Settings State
const showSettings = ref(false);
const backendUrl = ref('http://localhost:8000');
const jwtToken = ref('');
const selectedVoice = ref('zh-CN-XiaoxiaoNeural');
const selectedRate = ref('+0%');
const preloadCount = ref(3);
const voiceList = ref<Voice[]>([]);
const isLoadingVoices = ref(false);

const audioCache = new Map<number, Promise<string>>();

onMounted(async () => {
  await loadSettings();
  
  if (voiceList.value.length === 0) {
    fetchVoices();
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'AUDIO_ENDED') {
      playNextChunk();
    } else if (msg.type === 'AUDIO_ERROR') {
      error.value = 'Audio playback error';
      isPlaying.value = false;
      if (currentChunkIndex.value !== -1 && chunks.value[currentChunkIndex.value]) {
          chunks.value[currentChunkIndex.value].status = 'failed';
          audioCache.delete(currentChunkIndex.value);
      }
    }
  });
});

async function loadSettings() {
  const result = await chrome.storage.local.get(['backendUrl', 'jwtToken', 'voice', 'rate', 'voiceList', 'preloadCount']);
  if (result.backendUrl) backendUrl.value = result.backendUrl;
  if (result.jwtToken) jwtToken.value = result.jwtToken;
  if (result.voice) selectedVoice.value = result.voice;
  if (result.rate) selectedRate.value = result.rate;
  if (result.preloadCount) preloadCount.value = Math.max(1, Math.min(10, result.preloadCount));
  if (result.voiceList) voiceList.value = result.voiceList;
}

async function saveSettings() {
  preloadCount.value = Math.max(1, Math.min(10, preloadCount.value));
  await chrome.storage.local.set({
    backendUrl: backendUrl.value,
    jwtToken: jwtToken.value,
    voice: selectedVoice.value,
    rate: selectedRate.value,
    preloadCount: preloadCount.value,
    voiceList: voiceList.value
  });
}

function getApiUrl(path: string) {
    const base = backendUrl.value.replace(/\/$/, '');
    return `${base}/api/v1${path}`;
}

function getHeaders() {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json'
    };
    if (jwtToken.value) {
        headers['Authorization'] = `Bearer ${jwtToken.value}`;
    }
    return headers;
}

async function fetchVoices() {
    isLoadingVoices.value = true;
    try {
        const res = await fetch(getApiUrl('/tts/voices?engine=edge'), {
            headers: getHeaders()
        });
        
        if (res.status === 401) {
            alert("认证失败，请检查 JWT Token 配置。\nAuthentication failed. Please check your JWT Token.");
            showSettings.value = true;
            return;
        }
        
        if (!res.ok) throw new Error("Failed to fetch voices");
        const data = await res.json();
        voiceList.value = data;
        saveSettings(); 
    } catch (e: any) {
        error.value = "Voice fetch failed: " + e.message;
        alert("无法连接到后端，请配置后端地址。\nCannot connect to backend. Please configure backend URL.");
        showSettings.value = true;
    } finally {
        isLoadingVoices.value = false;
    }
}

function clearCache() {
    audioCache.forEach(async (promise) => {
        try {
            const url = await promise;
            URL.revokeObjectURL(url); 
        } catch(e) {}
    });
    audioCache.clear();
}

async function readPage() {
  error.value = '';
  status.value = 'Extracting text...';
  isLoading.value = true;
  clearCache(); 
  chunkRefs.value = {};

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error("No active tab found.");

    let text = "";
    try {
      const response = await chrome.tabs.sendMessage(tab.id, { type: 'GET_PAGE_TEXT' });
      text = response?.text;
    } catch (e) {
      throw new Error("Cannot connect to page. Please REFRESH the page.");
    }

    if (!text) {
        throw new Error("No text found. Select text or refresh page.");
    }

    status.value = 'Chunking text...';
    const chunkRes = await fetch(getApiUrl('/text/chunk'), {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify({ text })
    });
    
    if (chunkRes.status === 401) {
      alert("认证失败，请检查 JWT Token 配置。\nAuthentication failed. Please check your JWT Token.");
      showSettings.value = true;
      throw new Error("Unauthorized");
    }

    if (!chunkRes.ok) throw new Error(`Backend error: ${chunkRes.statusText}`);

    const data = await chunkRes.json();
    chunks.value = data.chunks.map((c: any) => ({ ...c, status: 'pending' }));
    
    if (chunks.value.length > 0) {
      playChunk(0);
    } else {
        status.value = "No chunks to play.";
    }

  } catch (e: any) {
    console.error(e);
    error.value = e.message || "Unknown error";
    status.value = "Failed";
  } finally {
    isLoading.value = false;
  }
}

async function fetchAudioBlob(text: string): Promise<string> {
    const response = await fetch(getApiUrl('/tts/stream'), {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({
            text: text,
            engine: 'edge',
            voice: selectedVoice.value,
            rate: selectedRate.value
        })
    });

    if (response.status === 401) {
        alert("认证失败，请检查 JWT Token 配置。\nAuthentication failed. Please check your JWT Token.");
        showSettings.value = true;
        throw new Error("Unauthorized");
    }

    if (!response.ok) throw new Error("TTS API Error");
    const blob = await response.blob();
    return URL.createObjectURL(blob);
}

function preloadChunk(index: number) {
    if (index >= chunks.value.length) return;
    if (audioCache.has(index)) return; 

    if (chunks.value[index].status === 'pending' || chunks.value[index].status === 'failed') {
        chunks.value[index].status = 'loading';
    }

    const promise = fetchAudioBlob(chunks.value[index].text)
        .then(url => {
            if (chunks.value[index].status === 'loading') {
                chunks.value[index].status = 'ready';
            }
            return url;
        })
        .catch(err => {
            console.error(`Failed to preload chunk ${index}`, err);
            chunks.value[index].status = 'failed';
            audioCache.delete(index); // Remove from cache to allow retry
            throw err;
        });
    audioCache.set(index, promise);
}

function retryChunk(index: number) {
    console.log(`Retrying chunk ${index}...`);
    // Remove from cache if exists (though preloadChunk should have cleared it on error)
    if (audioCache.has(index)) {
        audioCache.delete(index);
    }
    chunks.value[index].status = 'pending';
    preloadChunk(index);
}

async function playChunk(index: number) {
  if (index >= chunks.value.length) {
    status.value = 'Finished';
    isPlaying.value = false;
    isPaused.value = false;
    return;
  }

  if (currentChunkIndex.value !== -1 && currentChunkIndex.value < chunks.value.length) {
      chunks.value[currentChunkIndex.value].status = 'done';
  }

  currentChunkIndex.value = index;
  isPlaying.value = true;
  isPaused.value = false;
  
  chunks.value[index].status = 'playing';
  status.value = `Playing ${index + 1}/${chunks.value.length}`;

  nextTick(() => {
      const el = chunkRefs.value[index];
      if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
  });

  for (let i = 1; i <= preloadCount.value; i++) {
      preloadChunk(index + i);
  }

  try {
    let blobUrl: string;

    if (audioCache.has(index)) {
        try {
            blobUrl = await audioCache.get(index)!;
        } catch(e) {
            audioCache.delete(index);
            throw e;
        }
    } else {
        chunks.value[index].status = 'loading';
        const promise = fetchAudioBlob(chunks.value[index].text);
        audioCache.set(index, promise);
        try {
            blobUrl = await promise;
        } catch(e) {
            audioCache.delete(index);
            throw e;
        }
    }
    
    chunks.value[index].status = 'playing';
    await chrome.runtime.sendMessage({ 
        type: 'PLAY_AUDIO_REQUEST', 
        data: { blobUrl } 
    });
    
  } catch (e: any) {
    console.error(e);
    error.value = 'Error: ' + e.message;
    isPlaying.value = false;
    chunks.value[index].status = 'failed';
    audioCache.delete(index);
  }
}

function playNextChunk() {
  playNextChunkSafe();
}

function playNextChunkSafe() {
    if (currentChunkIndex.value + 1 < chunks.value.length) {
        playChunk(currentChunkIndex.value + 1);
    } else {
        status.value = 'Finished';
        isPlaying.value = false;
    }
}

function togglePause() {
    if (!isPlaying.value && !isPaused.value) return;

    if (isPaused.value) {
        chrome.runtime.sendMessage({ type: 'RESUME_AUDIO' });
        isPaused.value = false;
        status.value = `Playing ${currentChunkIndex.value + 1}/${chunks.value.length}`;
        if (currentChunkIndex.value !== -1) chunks.value[currentChunkIndex.value].status = 'playing';
    } else {
        chrome.runtime.sendMessage({ type: 'PAUSE_AUDIO' });
        isPaused.value = true;
        status.value = 'Paused';
    }
}

function stopPlayback() {
  isPlaying.value = false;
  isPaused.value = false;
  status.value = 'Stopped';
  if (currentChunkIndex.value !== -1 && chunks.value[currentChunkIndex.value]) {
      chunks.value[currentChunkIndex.value].status = 'ready'; 
  }
  chrome.runtime.sendMessage({ type: 'STOP_AUDIO' }).catch(() => {});
}
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 16px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  min-width: 320px;
  box-sizing: border-box;
  overflow: hidden;
}
.header {
  flex-shrink: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.header h2 { margin: 0; font-size: 1.2rem; }
.settings-btn {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
}
.settings-panel {
  flex-shrink: 0;
  background: #f5f5f5;
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 16px;
  border: 1px solid #ddd;
}
.form-group {
  margin-bottom: 8px;
}
.form-group label {
  display: block;
  font-size: 0.85rem;
  margin-bottom: 4px;
  color: #666;
}
.form-group input, .form-group select {
  width: 100%;
  padding: 6px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.controls {
  flex-shrink: 0;
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
.controls button {
  flex: 1;
  padding: 8px;
  cursor: pointer;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
}
.controls button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}
.status {
  flex-shrink: 0;
  font-size: 0.9rem;
  color: #007bff;
  margin-bottom: 8px;
}
.error {
  flex-shrink: 0;
  font-size: 0.9rem;
  color: #dc3545;
  margin-bottom: 8px;
}
.chunks {
  flex-grow: 1;
  overflow-y: auto;
  border-top: 1px solid #eee;
  padding-right: 4px;
}
.chunk-item {
  padding: 8px;
  border-bottom: 1px solid #eee;
  cursor: pointer;
  font-size: 0.9rem;
  line-height: 1.4;
  display: flex;
  gap: 8px;
}
.chunk-status {
    min-width: 24px;
    display: inline-block;
    text-align: center;
}
.chunk-text {
    flex: 1;
}
.chunk-item:hover {
  background-color: #f8f9fa;
}
.chunk-item.active {
  background-color: #e6f7ff;
  border-left: 4px solid #007bff;
  font-weight: 500;
}
.retry-btn {
    cursor: pointer;
    color: #dc3545;
    font-weight: bold;
}
.retry-btn:hover {
    transform: scale(1.1);
}
</style>