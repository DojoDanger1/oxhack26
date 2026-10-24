# Blundometer backend

Requires Python 3.10+. Run in this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

- GET /health returns process health (not model readiness).
- POST /predict accepts {"fen":"<current position>","time":[120,95],"rating":[1500,1600],"control":[300,5]}.
- Success: {"blunder_probability":0.32,"demo":false}.
- 422: invalid request fields. FEN semantics are not yet validated.
- 503: model not connected. 500: inference failed or returned an invalid probability.

For fixed 32% demo data, set `$env:BLUNDOMETER_DEMO = "1"` before starting
(or restarting) the server. Demo responses carry `"demo":true`; label these in
the UI. Remove with `Remove-Item Env:BLUNDOMETER_DEMO` for real-model mode.

Implement predict_blunder in predictor.py. Load the model once at startup.
The provisional target is the probability of a next-move blunder by the side
to move. Agree the chess variant, blunder definition and any extra inputs
(rating, clocks) with the model owner before freezing the contract.

Set ALLOWED_ORIGINS to a comma-separated list of actual website or extension
origins. A background-worker caller also needs the API host in its extension
manifest host permissions. The existing empty extension files are unchanged.

The caller should POST JSON, handle errors, use a timeout, and ignore responses
for outdated board positions. Convert probability to percent with
Math.round(blunder_probability * 100).

This is local development only: 127.0.0.1 is reachable only on the same computer.
Public deployment requires an HTTPS host and its API URL in the extension.

Request fields: fen is a non-blank string; time, rating, and control are required
JSON arrays containing exactly two integers each. Strings, booleans, and floats
are rejected as array items. Agree units and list ordering with the model owner;
no meaning for the control pair is assumed by the API. The entire validated
request is passed to predict_blunder(request). Old timeA/timeB/eloA/eloB fields
do not replace the required arrays.

