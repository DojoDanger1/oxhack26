import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import load_model, predict

model = load_model()

print(predict(model, "r1bqkbnr/pppp1ppp/2n5/4N3/3pP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 1",
              20, 20, 1200, 1200, 600, 0))

# def predict(model: ClockNet, fen: str, white_clock: float, black_clock: float,
#             white_elo: float, black_elo: float, base_time: float, increment: float,
#             eval_cp: float | None = None, eval_mate: int | None = None,
#             engine_depth: int = 12) -> dict:
#
# Returns white_win, draw, black_win, white_blunder, black_blunder,
# plus eval_cp / eval_mate (White's point of view) that were used.
