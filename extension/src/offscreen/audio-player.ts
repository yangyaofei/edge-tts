// Listen for messages from SidePanel/Background
chrome.runtime.onMessage.addListener(async (msg, sender, sendResponse) => {
    console.log("Offscreen: Received message:", msg.type);
    
    if (msg.type === 'PLAY_AUDIO') {
        const { data } = msg;
        // Check if we received a blobUrl (new preloading logic) or raw text (fallback)
        if (data.blobUrl) {
            console.log("Offscreen: Playing preloaded blob URL");
            await playBlob(data.blobUrl);
        } else {
            console.log("Offscreen: Fetching raw text (legacy path)");
            await fetchAndPlay(data);
        }
        sendResponse({ success: true });
    } else if (msg.type === 'STOP_AUDIO') {
        console.log("Offscreen: Stopping audio");
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0; // Reset
            // We might want to keep currentAudio instance if we just want to stop but not destroy
        }
    } else if (msg.type === 'PAUSE_AUDIO') {
        console.log("Offscreen: Pausing audio");
        if (currentAudio) {
            currentAudio.pause();
        }
    } else if (msg.type === 'RESUME_AUDIO') {
        console.log("Offscreen: Resuming audio");
        if (currentAudio) {
            currentAudio.play().catch(e => console.error("Resume failed", e));
        }
    }
});

let currentAudio: HTMLAudioElement | null = null;

async function playBlob(url: string) {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.src = ""; // Detach previous source
        currentAudio = null;
    }

    try {
        const audio = new Audio(url);
        currentAudio = audio;

        // Wait for metadata to be loaded to check duration
        await new Promise((resolve, reject) => {
            audio.onloadedmetadata = resolve;
            audio.onerror = reject;
            // Timeout fallback
            setTimeout(() => reject(new Error("Timeout loading metadata")), 5000);
        });

        console.log(`Offscreen: Audio loaded. Duration: ${audio.duration}s`);

        if (audio.duration === Infinity || isNaN(audio.duration)) {
             // For streaming responses sometimes duration is Infinity, which is fine, 
             // but for Blob URLs it should be finite.
             console.warn("Offscreen: Duration is Infinity or NaN");
        }

        audio.onplay = () => console.log("Offscreen: Audio started playing");
        
        // Use a Promise to wrap the playback lifecycle
        await new Promise<void>((resolve, reject) => {
            audio.onended = () => {
                console.log("Offscreen: Audio ended");
                resolve();
            };
            audio.onerror = (e) => {
                console.error("Offscreen: Audio element error during playback", e);
                reject(e);
            };

            // Start playback
            audio.play().catch(reject);
        });

        // Notify SidePanel only after the promise resolves (real end)
        chrome.runtime.sendMessage({ type: 'AUDIO_ENDED' });

    } catch (e) {
        console.error("Offscreen: Error in playBlob", e);
        // Only verify if it's a real error or just an interruption
        chrome.runtime.sendMessage({ type: 'AUDIO_ERROR' });
    }
}

async function fetchAndPlay(data: any) {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }

    try {
        console.log("Offscreen: Fetching audio from backend...");
        // Use the unified endpoint
        const response = await fetch('http://localhost:8000/api/v1/tts/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...data,
                engine: 'edge' // Explicitly select the engine
            })
        });

        if (!response.ok) throw new Error(`Backend returned ${response.status}`);

        const blob = await response.blob();
        console.log("Offscreen: Received blob, size:", blob.size);
        const url = URL.createObjectURL(blob);
        
        const audio = new Audio(url);
        currentAudio = audio;

        audio.onplay = () => console.log("Offscreen: Audio started playing");
        
        audio.onended = () => {
            console.log("Offscreen: Audio ended");
            URL.revokeObjectURL(url);
            chrome.runtime.sendMessage({ type: 'AUDIO_ENDED' });
        };

        audio.onerror = (e) => {
            console.error("Offscreen: Audio element error", e);
            chrome.runtime.sendMessage({ type: 'AUDIO_ERROR' });
        };

        await audio.play();
    } catch (e) {
        console.error("Offscreen: Error in fetchAndPlay", e);
        chrome.runtime.sendMessage({ type: 'AUDIO_ERROR' });
    }
}