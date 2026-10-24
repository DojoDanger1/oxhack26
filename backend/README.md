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
- Success: {"white_blunder":0.32,"black_blunder":0.24,"demo":false}.
- 422: invalid request fields. FEN semantics are not yet validated.
- 503: model not connected. 500: inference failed or returned an invalid probability.

For fixed demo data (White 32%, Black 24%), set `$env:BLUNDOMETER_DEMO = "1"` before starting
(or restarting) the server. Demo responses carry `"demo":true`; label these in
the UI. Remove with `Remove-Item Env:BLUNDOMETER_DEMO` for real-model mode.

predictor.py loads the unchanged yehor model on the first real prediction and
reuses it. Both White and Black blunder probabilities are returned. The side
to move refers to the current move; the opponent refers to its reply. These
probabilities do not need to sum to one. Engine evaluation defaults to zero.

Set ALLOWED_ORIGINS to a comma-separated list of actual website or extension
origins. A background-worker caller also needs the API host in its extension
manifest host permissions. The existing empty extension files are unchanged.

The caller should POST JSON, handle errors, use a timeout, and ignore responses
for outdated board positions. Convert probability to percent with
Math.round(white_blunder * 100) and Math.round(black_blunder * 100).

This is local development only: 127.0.0.1 is reachable only on the same computer.
Public deployment requires an HTTPS host and its API URL in the extension.

Request fields: fen is a non-blank string; time, rating, and control are required
JSON arrays containing exactly two integers each. Strings, booleans, and floats
are rejected as array items. time is [White seconds, Black seconds], rating is [White Elo, Black Elo],
and control is [base seconds, increment seconds]. The entire validated
request is passed to predict_blunder(request). Old timeA/timeB/eloA/eloB fields
do not replace the required arrays.

