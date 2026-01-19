// Listen for requests to get text
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'GET_PAGE_TEXT') {
        const text = getPageContent();
        sendResponse({ text });
    }
});

function getPageContent() {
    // 1. Try to get user selection
    const selection = window.getSelection()?.toString().trim();
    if (selection && selection.length > 0) {
        console.log("Using user selection");
        return selection;
    }

    // 2. Fallback to full body text (Simple extraction)
    // Future improvement: Use readability.js
    console.log("Using full page body");
    return document.body.innerText;
}
