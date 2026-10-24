"""
Stream a Lichess PGN database and turn it into one row per position.

Only standard games that have BOTH engine evals ([%eval ...]) and clocks
([%clk ...]) are used; everything else is skipped. The file is read line by
line, so the size of the PGN doesn't matter.

Each row describes the position BEFORE a move is played:
    game_id, ply, fen, move_uci,
    white_clock, black_clock          seconds left before the move
    white_elo, black_elo,
    base_time, increment              seconds
    eval_cp, eval_mate                engine eval of the position, White's POV
    result                            1.0 White won, 0.5 draw, 0.0 Black won
    is_blunder                        1 if Lichess marked the move "??" (blunder)
    opp_blunder_next                  1 if the opponent blunders on the reply

Command line:
    python database_reader.py                     # first row of database.pgn
    python database_reader.py --rows 10           # first 10 rows
    python database_reader.py --rows 3 --encode   # also show network inputs
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

import chess

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Blunder labels come straight from Lichess's own "??" annotations, which every
# analysed game in the database has. (A fixed win%-drop threshold only roughly
# reproduces them: checked on 3,000 games, a 30-point drop caught 43% of them.)
BLUNDER_GLYPH = "??"
WINPCT_K = 0.00368208
CP_CLIP = 1000

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
TOKEN_RE = re.compile(r"\{([^}]*)\}|(\S+)")
EVAL_RE = re.compile(r"\[%eval\s+(#?)(-?\d+(?:\.\d+)?)")
CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
MOVE_NUMBER_RE = re.compile(r"^\d+\.+$")
RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
SKIP_TERMINATIONS = {"Abandoned", "Rules infraction"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def white_win_pct(eval_cp: Optional[float], eval_mate: Optional[int]) -> float:
    """White's winning chance on a 0-100 scale."""
    if eval_mate is not None:
        return 100.0 if eval_mate > 0 else 0.0
    cp = max(-CP_CLIP, min(CP_CLIP, eval_cp))
    return 100.0 / (1.0 + math.exp(-WINPCT_K * cp))


def parse_time_control(tc: str) -> Optional[tuple[float, float]]:
    """'180+2' -> (180.0, 2.0); '-' (correspondence) -> None."""
    if "+" not in tc:
        return None
    base, inc = tc.split("+", 1)
    return float(base), float(inc)


def parse_comment(text: str, ann: dict) -> None:
    """Pull %eval and %clk out of a {...} comment into ann."""
    m = EVAL_RE.search(text)
    if m:
        if m.group(1) == "#":
            ann["mate"] = int(m.group(2))
            ann["cp"] = None
        else:
            ann["cp"] = float(m.group(2)) * 100.0  # pawns -> centipawns
            ann["mate"] = None
    m = CLK_RE.search(text)
    if m:
        ann["clk"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


# ---------------------------------------------------------------------------
# PGN streaming
# ---------------------------------------------------------------------------
def iter_raw_games(path: str) -> Iterator[tuple[dict, str]]:
    """Yield (headers, movetext) for every game in the file."""
    headers: dict = {}
    movetext: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = HEADER_RE.match(line)
            if m:
                if movetext:  # a new header after movetext = new game
                    yield headers, " ".join(movetext)
                    headers, movetext = {}, []
                headers[m.group(1)] = m.group(2)
            else:
                movetext.append(line)
    if movetext:
        yield headers, " ".join(movetext)


def parse_moves(movetext: str) -> list[dict]:
    """Movetext -> [{'san', 'cp', 'mate', 'clk'}, ...]; eval/clk are AFTER the move."""
    moves: list[dict] = []
    for comment, token in TOKEN_RE.findall(movetext):
        if comment:
            if moves:
                parse_comment(comment, moves[-1])
            continue
        if MOVE_NUMBER_RE.match(token) or token in RESULTS or token == "*":
            continue
        if token.startswith("$"):  # numeric annotation glyph
            continue
        san = re.sub(r"^\d+\.+", "", token)
        glyph = re.search(r"[?!]+$", san)
        glyph = glyph.group(0) if glyph else ""
        san = san.rstrip("?!")
        if san:
            moves.append({"san": san, "glyph": glyph, "cp": None, "mate": None, "clk": None})
    return moves


def game_rows(headers: dict, movetext: str) -> Iterator[dict]:
    """Yield one row per position in a single game (or nothing if unusable)."""
    if "[%eval" not in movetext or "[%clk" not in movetext:
        return
    if headers.get("Variant", "Standard") != "Standard":
        return
    if headers.get("Termination") in SKIP_TERMINATIONS:
        return
    result = RESULTS.get(headers.get("Result", "*"))
    tc = parse_time_control(headers.get("TimeControl", "-"))
    try:
        white_elo = float(headers["WhiteElo"])
        black_elo = float(headers["BlackElo"])
    except (KeyError, ValueError):  # missing or "?"
        return
    if result is None or tc is None:
        return
    base_time, increment = tc

    moves = parse_moves(movetext)
    if len(moves) < 2:
        return

    board = chess.Board()
    clocks = {chess.WHITE: base_time, chess.BLACK: base_time}
    prev_cp: Optional[float] = None    # eval of the current position
    prev_mate: Optional[int] = None
    pending: list[dict] = []           # rows still needing opp_blunder_next

    for ply, mv in enumerate(moves):
        mover = board.turn
        fen = board.fen()
        try:
            move = board.parse_san(mv["san"])
        except ValueError:
            return  # corrupt game; drop it
        board.push(move)

        # eval after this move (a checkmate move often has no eval comment)
        if board.is_checkmate():
            after_cp, after_mate = None, (1 if mover == chess.WHITE else -1)
        else:
            after_cp, after_mate = mv["cp"], mv["mate"]
        have_after = after_cp is not None or after_mate is not None
        have_before = prev_cp is not None or prev_mate is not None

        # ply 0 has no eval before it, so it produces no row
        if have_before and have_after:
            is_blunder = int(mv["glyph"] == BLUNDER_GLYPH)

            if pending:  # this move is the opponent's reply to the last row
                pending[-1]["opp_blunder_next"] = is_blunder
                yield pending.pop()

            pending.append({
                "game_id": headers.get("Site", ""),
                "ply": ply,
                "fen": fen,
                "move_uci": move.uci(),
                "white_clock": clocks[chess.WHITE],
                "black_clock": clocks[chess.BLACK],
                "white_elo": white_elo,
                "black_elo": black_elo,
                "base_time": base_time,
                "increment": increment,
                "eval_cp": prev_cp,
                "eval_mate": prev_mate,
                "result": result,
                "is_blunder": is_blunder,
                "opp_blunder_next": None,
            })
        else:
            pending.clear()  # gap in evals: can't label the previous row

        if mv["clk"] is not None:
            clocks[mover] = mv["clk"]
        prev_cp, prev_mate = after_cp, after_mate
        if not have_after:
            break  # evals stop (e.g. analysis cut off); stop the game here

    # the last move has no reply: the opponent can't blunder anymore
    for row in pending:
        row["opp_blunder_next"] = 0
        yield row


def iter_rows(path: str) -> Iterator[dict]:
    """Yield rows for every usable position in the whole database."""
    for headers, movetext in iter_raw_games(path):
        yield from game_rows(headers, movetext)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Print rows from a Lichess PGN database.")
    parser.add_argument("path", nargs="?", default=str(HERE / "database.pgn"))
    parser.add_argument("--rows", type=int, default=1, help="number of rows to print")
    parser.add_argument("--encode", action="store_true",
                        help="also show the encoded network inputs (uses main.py)")
    args = parser.parse_args()

    if args.encode:
        from main import SCALAR_NAMES, encode_position

    for i, row in enumerate(iter_rows(args.path)):
        if i >= args.rows:
            break
        print(f"--- row {i} ---")
        for key, value in row.items():
            print(f"{key:18s} {value}")
        if args.encode:
            planes, scalars = encode_position(
                row["fen"], row["white_clock"], row["black_clock"],
                row["white_elo"], row["black_elo"],
                row["base_time"], row["increment"],
                eval_cp=row["eval_cp"], eval_mate=row["eval_mate"],
            )
            print(f"{'board planes':18s} {planes.shape} ({int(planes.sum())} pieces/marks)")
            for name, value in zip(SCALAR_NAMES, scalars):
                print(f"  {name:22s} {value: .4f}")
        print()


if __name__ == "__main__":
    main()
