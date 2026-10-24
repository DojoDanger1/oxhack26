async function streamLichessGame(gameId, onUpdate) {
    const url = `https://lichess.org/api/stream/game/${encodeURIComponent(gameId)}`;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        if (!response.body) {
            throw new Error("Response has no readable stream");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true }); // Decode the incoming bytes and append them to our buffer.
            const lines = buffer.split("\n"); // Each JSON object occupies a separate line.
            buffer = lines.pop(); // Keep the final, potentially incomplete line for next time.

            for (const line of lines) {
                if (!line.trim()) {
                    continue;
                }

                const update = JSON.parse(line);
                onUpdate(update); // Pass the parsed update to the caller.
            }
        }

        buffer += decoder.decode(); // Process any remaining data after the stream closes.

        if (buffer.trim()) {
            onUpdate(JSON.parse(buffer));
        }

    } catch (error) {
        console.error("Lichess game stream failed:", error);
    }
}

const url = new URL(window.location.href).pathname.split("/");
const gameId = url[2];
var pov = 'black'
if (url.length == 4) {
    pov = url[3]
} else if (url.length == 3) {
    pov = 'white'
}

console.log(gameId);
console.log(url)
console.log(pov)

var mostRecentPosition = '';

streamLichessGame(gameId, update => {
    mostRecentPosition = update;
    console.log(update);
});

const button = document.createElement("button");

button.textContent = "My Custom Button";
button.id = "my-extension-button";

button.addEventListener("click", () => {
    var fen = mostRecentPosition.fen
    var times = [...document.querySelectorAll('.time')].map(el => {
        const [minutes,seconds] = el.textContent.split(':').map(Number);
        return minutes * 60 + seconds
    })
    if (pov == 'white') {
        times.reverse()
    }
    var ratings = [...document.querySelectorAll('.rating')].map(el => parseInt(el.textContent.slice(2,-1)))
    var control = [...document.querySelectorAll('.setup')].map(el => el.textContent)[0].split(' ')[0].split('+').map(Number)
    control[0] = control[0]*60

    // these values will be continuously fed into server
    console.log(fen)
    console.log(times)
    console.log(ratings)
    console.log(control)
    
    chrome.runtime.sendMessage(
        {
            type: "PREDICT",
            data: {"fen": fen, "time": times, "rating": ratings, "control": control}
        },
        response => {
            if (chrome.runtime.lastError) {
                console.error(chrome.runtime.lastError.message);
                return;
            }

            if (!response.success) {
                console.error("API error:", response.error);
                return;
            }

            console.log("Prediction:", response.result);
        }
    );
});

document.body.appendChild(button);