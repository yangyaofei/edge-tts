// Open side panel on icon click
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((error) => console.error("Background: Error setting side panel behavior", error));

// Ensure offscreen document exists
async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) {
    console.log("Background: Offscreen document already exists");
    return;
  }
  console.log("Background: Creating offscreen document...");
  await chrome.offscreen.createDocument({
    url: 'src/offscreen/index.html',
    reasons: [chrome.offscreen.Reason.AUDIO_PLAYBACK],
    justification: 'Play TTS audio in background',
  });
  console.log("Background: Offscreen document created");
}

// Handle messages
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  console.log("Background: Received message:", msg.type, "from:", sender.id ? "extension" : "content script");
  
  if (msg.type === 'PLAY_AUDIO_REQUEST') {
    (async () => {
      try {
        await ensureOffscreen();
        console.log("Background: Forwarding PLAY_AUDIO to offscreen...");
        // Use a slight delay to ensure offscreen is ready to listen
        setTimeout(() => {
            chrome.runtime.sendMessage({
                type: 'PLAY_AUDIO',
                data: msg.data
            });
            console.log("Background: PLAY_AUDIO message broadcasted");
        }, 200);
      } catch (e) {
        console.error("Background: Failed to handle play request", e);
      }
    })();
    return true;
  }
});
