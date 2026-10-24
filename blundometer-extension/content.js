// =====================================================================
// Blundometer — live blunder thermometers + win/draw/loss bar for Lichess
// =====================================================================

const POLL_MS = 1000;          // how often to ask the model while the game runs
const SCALE_MAX = 0.4;         // a thermometer is full at 40% blunder chance
const DANGER_AT = 0.2;         // above this the thermometer starts to pulse

// ---------------------------------------------------------------------
// Lichess game stream (unchanged)
// ---------------------------------------------------------------------
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

var mostRecentPosition = '';

// ---------------------------------------------------------------------
// Read the model inputs from the page (same logic as before, as a function)
// ---------------------------------------------------------------------
function collectInputs() {
    var fen = mostRecentPosition.fen
    var times = [...document.querySelectorAll('.time')].map(el => {
        const [minutes, seconds] = el.textContent.split(':').map(Number);
        return minutes * 60 + seconds
    })
    if (pov == 'white') {
        times.reverse()
    }
    var ratings = [...document.querySelectorAll('.rating')].map(el => parseInt(el.textContent.slice(2, -1)))
    var control = [...document.querySelectorAll('.setup')].map(el => el.textContent)[0].split(' ')[0].split('+').map(Number)
    control[0] = control[0] * 60

    return { "fen": fen, "time": times, "rating": ratings, "control": control }
}

// ---------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------
const BULB_URL = chrome.runtime.getURL("imgs/blunder.png");

function el(tag, className, parent) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (parent) parent.appendChild(node);
    return node;
}

function makeThermometer(direction) {
    // direction: "ltr" (bulb on the left, fills rightwards) or "rtl"
    const root = el("div", `bm-thermo bm-${direction}`);
    const bulb = el("div", "bm-bulb", root);
    el("div", "bm-bulb-glow", bulb);
    const img = el("img", "bm-bulb-img", bulb);
    img.src = BULB_URL;
    img.alt = "??";

    const tube = el("div", "bm-tube", root);
    const fill = el("div", "bm-fill", tube);
    el("div", "bm-fill-stripes", fill);
    el("div", "bm-fill-shine", fill);
    el("div", "bm-fill-tip", fill);
    const ticks = el("div", "bm-ticks", tube);
    for (let i = 1; i < 4; i++) {
        const t = el("div", "bm-tick", ticks);
        t.style.setProperty("--p", i * 25);
    }
    const text = el("div", "bm-text", tube);
    const name = el("span", "bm-name", text);
    const value = el("span", "bm-value", text);
    value.textContent = "–";

    return {
        root,
        setName(n) { name.textContent = n; },
        set(p) {
            const frac = Math.max(0, Math.min(1, p / SCALE_MAX));
            value.textContent = `${(p * 100).toFixed(1)}%`;
            root.style.setProperty("--heat", frac.toFixed(3));
            root.classList.toggle("bm-danger", p >= DANGER_AT);
        },
    };
}

function makeWdlBar() {
    const root = el("div", "bm-wdl");
    const segs = {};
    for (const key of ["black", "draw", "white"]) {
        const seg = el("div", `bm-seg bm-seg-${key}`, root);
        const lab = el("span", "bm-seg-label", seg);
        segs[key] = { seg, lab };
    }
    return {
        root,
        set(white, draw, black, whiteAtBottom) {
            root.classList.toggle("bm-flipped", !whiteAtBottom);
            for (const [key, p] of [["white", white], ["draw", draw], ["black", black]]) {
                segs[key].seg.style.flexGrow = Math.max(p, 0.0001);
                segs[key].lab.textContent = p >= 0.07 ? `${Math.round(p * 100)}` : "";
            }
        },
    };
}

const ui = (() => {
    const overlay = el("div", "bm-overlay");
    const top = el("div", "bm-slot bm-slot-top", overlay);
    const bottom = el("div", "bm-slot bm-slot-bottom", overlay);
    const left = el("div", "bm-slot bm-slot-left", overlay);

    const thermoTop = makeThermometer("rtl");
    const thermoBottom = makeThermometer("ltr");
    const wdl = makeWdlBar();
    top.appendChild(thermoTop.root);
    bottom.appendChild(thermoBottom.root);
    left.appendChild(wdl.root);

    const status = el("div", "bm-status", overlay);
    status.dataset.state = "wait";
    status.innerHTML = '<span class="bm-dot"></span><span class="bm-brand">BLUNDOMETER</span><span class="bm-msg">waiting for game…</span>';
    const statusMsg = status.querySelector(".bm-msg");

    document.body.appendChild(overlay);
    return { overlay, thermoTop, thermoBottom, wdl, status, statusMsg };
})();

function boardElement() {
    return document.querySelector("cg-wrap") || document.querySelector(".round__app__board");
}

function whiteAtBottom() {
    const wrap = document.querySelector("cg-wrap");
    if (wrap && wrap.classList.contains("orientation-black")) return false;
    if (wrap && wrap.classList.contains("orientation-white")) return true;
    return pov !== "black";
}

// Keep the overlay glued to the board (it moves when the page resizes).
function placeOverlay() {
    const board = boardElement();
    if (!board) {
        ui.overlay.style.display = "none";
        return;
    }
    const r = board.getBoundingClientRect();
    ui.overlay.style.display = "";
    ui.overlay.style.left = `${r.left + window.scrollX}px`;
    ui.overlay.style.top = `${r.top + window.scrollY}px`;
    ui.overlay.style.width = `${r.width}px`;
    ui.overlay.style.height = `${r.height}px`;

    // Each player's thermometer sits on their own side of the board.
    const wb = whiteAtBottom();
    ui.thermoBottom.setName(wb ? "WHITE" : "BLACK");
    ui.thermoTop.setName(wb ? "BLACK" : "WHITE");
}

window.addEventListener("resize", placeOverlay);
new ResizeObserver(placeOverlay).observe(document.documentElement);
setInterval(placeOverlay, 500);
placeOverlay();

function render(result) {
    const wb = whiteAtBottom();
    (wb ? ui.thermoBottom : ui.thermoTop).set(result.white_blunder);
    (wb ? ui.thermoTop : ui.thermoBottom).set(result.black_blunder);
    ui.wdl.set(result.white_win, result.draw, result.black_win, wb);
}

function setStatus(text, state) {
    ui.statusMsg.textContent = text;
    ui.status.dataset.state = state;
}

// ---------------------------------------------------------------------
// Ask the model continuously (replaces the old click-to-predict button)
// ---------------------------------------------------------------------
let inFlight = false;

function requestPrediction() {
    if (inFlight || !mostRecentPosition || !mostRecentPosition.fen) return;

    let data;
    try {
        data = collectInputs();
        console.log(data)
    } catch (error) {
        setStatus("reading page…", "wait");
        return;
    }

    inFlight = true;
    chrome.runtime.sendMessage({ type: "PREDICT", data }, response => {
        inFlight = false;
        if (chrome.runtime.lastError) {
            setStatus("extension error", "error");
            console.error(chrome.runtime.lastError.message);
            return;
        }
        if (!response || !response.success) {
            setStatus("server offline", "error");
            console.error("API error:", response && response.error);
            return;
        }
        if (data.fen !== mostRecentPosition.fen) return; // a move happened meanwhile
        setStatus("live", "live");
        render(response.result);
    });
}

streamLichessGame(gameId, update => {
    if (update && update.fen) {
        mostRecentPosition = update;
        requestPrediction(); // react to new moves immediately
    }
});

setInterval(requestPrediction, POLL_MS); // and keep up with the ticking clocks
