"""
Input encoding for the clock-aware evaluation / blunder model.

Everything is encoded from the perspective of the SIDE TO MOVE ("me"):
if Black is to move, the board is mirrored and colours swapped, and the
clocks / ratings are swapped to match. This lets the network learn one
set of patterns instead of two.

Use `encode_position` both when building the training set and in the live
app, so the two can never drift apart.
"""

from __future__ import annotations

import math
from typing import Optional

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Board tensor layout: (BOARD_PLANES, 8, 8), dtype uint8
# row index = rank (0 = my back rank), column index = file (0 = a-file)
# ---------------------------------------------------------------------------
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

BOARD_PLANE_NAMES = [
    "my_pawns", "my_knights", "my_bishops", "my_rooks", "my_queens", "my_king",
    "their_pawns", "their_knights", "their_bishops", "their_rooks", "their_queens", "their_king",
    "en_passant_target",
]
BOARD_PLANES = len(BOARD_PLANE_NAMES)  # 13

# ---------------------------------------------------------------------------
# Scalar vector layout: (N_SCALARS,), dtype float32
# ---------------------------------------------------------------------------
SCALAR_NAMES = [
    "eval",                 # 2*winpct - 1, in [-1, 1], from my perspective
    "mate_flag",            # +1 I have a forced mate, -1 I'm getting mated, 0 otherwise
    "my_clock_log",         # log(1 + seconds)
    "their_clock_log",      # log(1 + seconds)
    "my_clock_frac",        # my_clock / estimated game time, clipped to [0, 1.5]
    "their_clock_frac",     # their_clock / estimated game time, clipped to [0, 1.5]
    "base_time_log",        # log(1 + base seconds)
    "increment_log",        # log(1 + increment seconds)
    "my_elo",               # (elo - 1500) / 400
    "their_elo",            # (elo - 1500) / 400
    "my_castle_kingside",   # 0/1
    "my_castle_queenside",  # 0/1
    "their_castle_kingside",
    "their_castle_queenside",
    "move_number",          # fullmove number / 100
]
N_SCALARS = len(SCALAR_NAMES)  # 15

# Constants for the transforms
ELO_CENTER = 1500.0
ELO_SCALE = 400.0          # 400 Elo = 10:1 expected-score odds
WINPCT_K = 0.00368208      # Lichess centipawn -> win% coefficient
CP_CLIP = 1000             # Lichess also clips evals at +-10 pawns
CLOCK_FRAC_CLIP = 1.5      # clock can exceed base thanks to increment
EST_MOVES = 40             # Lichess estimated game length: base + 40 * increment


def cp_to_eval_feature(cp: float) -> float:
    """Centipawns -> 2*winpct - 1, i.e. a value in [-1, 1]."""
    cp = max(-CP_CLIP, min(CP_CLIP, cp))
    win_pct = 1.0 / (1.0 + math.exp(-WINPCT_K * cp))  # in [0, 1]
    return 2.0 * win_pct - 1.0


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a board that is ALREADY oriented so that White = side to move.
    Returns uint8 array of shape (13, 8, 8).
    """
    planes = np.zeros((BOARD_PLANES, 8, 8), dtype=np.uint8)
    for i, piece_type in enumerate(PIECE_TYPES):
        for sq in board.pieces(piece_type, chess.WHITE):
            planes[i, chess.square_rank(sq), chess.square_file(sq)] = 1
        for sq in board.pieces(piece_type, chess.BLACK):
            planes[6 + i, chess.square_rank(sq), chess.square_file(sq)] = 1
    if board.ep_square is not None and board.has_legal_en_passant():
        sq = board.ep_square
        planes[12, chess.square_rank(sq), chess.square_file(sq)] = 1
    return planes


def encode_position(
    fen: str,
    white_clock: float,
    black_clock: float,
    white_elo: float,
    black_elo: float,
    base_time: float,
    increment: float,
    eval_cp: Optional[float] = None,
    eval_mate: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw inputs into network inputs.

    Args:
        fen:          position in FEN.
        white_clock:  White's remaining time in seconds.
        black_clock:  Black's remaining time in seconds.
        white_elo:    White's rating.
        black_elo:    Black's rating.
        base_time:    time control base in seconds (e.g. 180 for 3+2).
        increment:    time control increment in seconds (e.g. 2 for 3+2).
        eval_cp:      engine eval in centipawns from WHITE's point of view
                      (as in Lichess %eval / Stockfish "score white").
        eval_mate:    mate-in-N from WHITE's point of view (positive = White
                      mates). If given, it takes priority over eval_cp.
                      If both are None the eval is treated as 0.

    Returns:
        board:   uint8   array, shape (13, 8, 8)
        scalars: float32 array, shape (15,)
    """
    board = chess.Board(fen)
    white_to_move = board.turn == chess.WHITE

    # --- orient everything to the side to move -------------------------
    if white_to_move:
        oriented = board
        my_clock, their_clock = white_clock, black_clock
        my_elo, their_elo = white_elo, black_elo
        sign = 1.0
    else:
        oriented = board.mirror()  # flips ranks and swaps colours
        my_clock, their_clock = black_clock, white_clock
        my_elo, their_elo = black_elo, white_elo
        sign = -1.0

    board_planes = encode_board(oriented)

    # --- eval ------------------------------------------------------------
    if eval_mate is not None:
        mate_flag = sign * (1.0 if eval_mate > 0 else -1.0)
        eval_feat = mate_flag  # a forced mate is as good/bad as it gets
    else:
        mate_flag = 0.0
        eval_feat = cp_to_eval_feature(sign * (eval_cp or 0.0))

    # --- time ------------------------------------------------------------
    my_clock = max(0.0, my_clock)
    their_clock = max(0.0, their_clock)
    est_game_time = max(1.0, base_time + EST_MOVES * increment)

    def clock_frac(t: float) -> float:
        return min(CLOCK_FRAC_CLIP, t / est_game_time)

    # --- castling (oriented board: White = me) -------------------------
    castling = [
        oriented.has_kingside_castling_rights(chess.WHITE),
        oriented.has_queenside_castling_rights(chess.WHITE),
        oriented.has_kingside_castling_rights(chess.BLACK),
        oriented.has_queenside_castling_rights(chess.BLACK),
    ]

    scalars = np.array(
        [
            eval_feat,
            mate_flag,
            math.log1p(my_clock),
            math.log1p(their_clock),
            clock_frac(my_clock),
            clock_frac(their_clock),
            math.log1p(max(0.0, base_time)),
            math.log1p(max(0.0, increment)),
            (my_elo - ELO_CENTER) / ELO_SCALE,
            (their_elo - ELO_CENTER) / ELO_SCALE,
            *[float(c) for c in castling],
            board.fullmove_number / 100.0,
        ],
        dtype=np.float32,
    )
    assert scalars.shape == (N_SCALARS,)
    return board_planes, scalars


if __name__ == "__main__":
    # Example: after 1. e4 e5 2. Nf3, Black to move, 3+2 blitz
    fen = "1q3r1k/p1n1b1pp/3p1n2/4pQ2/2b5/2N1P1P1/P2PNPBP/2B2RK1 b - - 0 16"
    planes, scalars = encode_position(
        fen,
        white_clock=176.0,
        black_clock=181.0,
        white_elo=1850,
        black_elo=1720,
        base_time=180,
        increment=2,
        eval_cp=35,
    )
    print("board planes:", planes.shape, planes.dtype)
    print("my pawns (rank 8 at top):")
    print(planes[0][::-1])
    print()
    for name, value in zip(SCALAR_NAMES, scalars):
        print(f"{name:24s} {value: .4f}")
