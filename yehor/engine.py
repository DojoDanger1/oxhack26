"""
Stockfish evaluation of a position, via python-chess's chess.engine module.

    from engine import evaluate
    eval_cp, eval_mate = evaluate(fen)          # both from WHITE's point of view

Exactly one of the two is set:
    eval_cp   = centipawns (+35 = White a third of a pawn better), or
    eval_mate = mate in N (+3 = White mates in 3, -2 = Black mates in 2).
This is the same format as the Lichess %eval used in training, so the result
can go straight into encode_position / predict.

Stockfish itself is a separate program: install it with `brew install stockfish`
(Mac). If it lives somewhere unusual, set the STOCKFISH_PATH environment variable.

One Stockfish process is started the first time evaluate() is called and kept
running, so later calls only pay for the search itself. It is shared safely
between threads (e.g. web server requests) and shut down when Python exits.
"""

from __future__ import annotations

import os
import shutil
import threading
from typing import Optional

import chess
import chess.engine

DEFAULT_DEPTH = 12  # ~50-150 ms per position; plenty for an eval bar

_CANDIDATE_PATHS = [
    "/opt/homebrew/bin/stockfish",   # Homebrew on Apple Silicon
    "/usr/local/bin/stockfish",      # Homebrew on Intel Macs, manual installs
    "/usr/games/stockfish",          # Debian / Ubuntu
    "/usr/bin/stockfish",
]

_engine: Optional[chess.engine.SimpleEngine] = None
_lock = threading.Lock()


def find_stockfish() -> str:
    """Return the path to the Stockfish binary, or raise a helpful error."""
    candidates = [os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish"), *_CANDIDATE_PATHS]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "Stockfish not found. Install it with `brew install stockfish`, "
        "or set STOCKFISH_PATH to the binary's location."
    )


def _get_engine() -> chess.engine.SimpleEngine:
    """Start Stockfish on first use; restart it if it has died. Call with _lock held."""
    global _engine
    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    return _engine


def close() -> None:
    """Shut Stockfish down (also happens automatically when Python exits)."""
    global _engine
    with _lock:
        if _engine is not None:
            try:
                _engine.quit()
            except chess.engine.EngineError:
                pass
            _engine = None


# Use threading's exit hook rather than atexit: it runs BEFORE Python waits for
# background threads, so quit() can still talk to the engine's I/O thread
# (with plain atexit, interpreter shutdown hangs).
if hasattr(threading, "_register_atexit"):
    threading._register_atexit(close)
else:  # very old / unusual Pythons
    import atexit
    atexit.register(close)


def evaluate(fen: str, depth: int = DEFAULT_DEPTH,
             time_limit: Optional[float] = None) -> tuple[Optional[float], Optional[int]]:
    """
    Evaluate a position with Stockfish.

    Args:
        fen:        position in FEN.
        depth:      search depth (ignored if time_limit is given).
        time_limit: seconds to think instead of a fixed depth, e.g. 0.1.

    Returns:
        (eval_cp, eval_mate), White's point of view; exactly one is not None.
    """
    board = chess.Board(fen)

    # Game already over: no search needed (and Stockfish's answer is awkward here).
    if board.is_checkmate():
        # The side to move has been mated, so the other side has "mated in 0";
        # we use +-1 so the sign is unambiguous.
        return None, (-1 if board.turn == chess.WHITE else 1)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0, None

    limit = chess.engine.Limit(time=time_limit) if time_limit else chess.engine.Limit(depth=depth)
    with _lock:
        try:
            info = _get_engine().analyse(board, limit)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
            # Stockfish crashed or was killed: start a fresh one and retry once.
            global _engine
            _engine = None
            info = _get_engine().analyse(board, limit)

    score = info["score"].white()
    if score.is_mate():
        return None, score.mate()
    return float(score.score()), None


if __name__ == "__main__":
    import sys
    import time

    fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    t0 = time.time()
    cp, mate = evaluate(fen)
    ms = (time.time() - t0) * 1000
    print(f"Stockfish: {find_stockfish()}")
    print(f"FEN:       {fen}")
    print(f"eval_cp={cp}  eval_mate={mate}  ({ms:.0f} ms, includes engine start-up)")
