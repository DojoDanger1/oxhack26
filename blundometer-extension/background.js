chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "PREDICT") {
        fetch("https://meat-jewellery-emotional-care.trycloudflare.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(message.data)
        })
        .then(async response => {
            const body = await response.text();

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${body}`);
            }

            return JSON.parse(body);
        })
        .then(result => sendResponse({ success: true, result }))
        .catch(error => sendResponse({
            success: false,
            error: error.message
        }));

        return true;
    }
});