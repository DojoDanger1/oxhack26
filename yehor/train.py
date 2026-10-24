"""
Build the dataset, train ClockNet, and save it to models/clocknet.pt.

    python train.py --positions 200000 --epochs 2      # quick smoke test
    python train.py                                     # full run (2M positions)
    python train.py --rebuild                           # force re-reading the PGN

The dataset is cached in data/ after the first run, so later runs with the
same --positions / --per-game skip straight to training.

The live app only needs:
    from train import load_model, predict
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# All paths are relative to THIS file's folder, so the scripts work no matter
# which directory you run them from (or import them from).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from database_reader import iter_raw_games, game_rows  # noqa: E402
from main import BOARD_PLANES, BOARD_PLANE_NAMES, N_SCALARS, SCALAR_NAMES, encode_position  # noqa: E402

DATA_DIR = str(HERE / "data")
MODEL_PATH = str(HERE / "models" / "clocknet.pt")
DEFAULT_PGN = str(HERE / "database.pgn")
VAL_FRACTION_MOD = 20   # 1 in 20 games (5%) goes to validation
LABEL_VERSION = 2       # bump whenever database_reader's labels change (2 = Lichess "??" glyphs)


# ===========================================================================
# Model
# ===========================================================================
class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class ClockNet(nn.Module):
    """
    Inputs (side-to-move perspective, see main.py):
        board   (N, 13, 8, 8) uint8/float
        scalars (N, 15) float
    Outputs:
        wdl_logits     (N, 3)  win / draw / loss for the side to move
        blunder_logits (N, 2)  [side to move blunders now, opponent blunders next]
    """

    def __init__(self, n_scalars: int = N_SCALARS, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.board_fc = nn.Linear(channels * 64, 256)
        self.scalar_fc = nn.Sequential(nn.Linear(n_scalars, 64), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.wdl = nn.Linear(128, 3)
        self.blunder = nn.Linear(128, 2)

    def forward(self, board, scalars):
        b = self.tower(self.stem(board.float())).flatten(1)
        h = torch.cat([F.relu(self.board_fc(b)), self.scalar_fc(scalars)], dim=1)
        h = self.trunk(h)
        return self.wdl(h), self.blunder(h)


def loss_fn(wdl_logits, bl_logits, wdl_target, bl_target):
    wdl_loss = F.cross_entropy(wdl_logits, wdl_target)
    bl_loss = F.binary_cross_entropy_with_logits(bl_logits, bl_target)
    return wdl_loss + bl_loss, wdl_loss, bl_loss


# ===========================================================================
# Dataset building
# ===========================================================================
def wdl_target(result_white: float, white_to_move: bool) -> int:
    """0 = side to move wins, 1 = draw, 2 = side to move loses."""
    if result_white == 0.5:
        return 1
    mover_won = (result_white == 1.0) == white_to_move
    return 0 if mover_won else 2


def build_dataset(pgn_path: str, n_positions: int, per_game: int, seed: int = 0) -> None:
    if not os.path.isfile(pgn_path):
        raise FileNotFoundError(f"PGN not found: {pgn_path}")
    os.makedirs(DATA_DIR, exist_ok=True)
    meta_path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)  # a half-built dataset must never look valid
    rng = random.Random(seed)
    mm = np.lib.format.open_memmap
    boards = mm(os.path.join(DATA_DIR, "boards.npy"), "w+", np.uint8, (n_positions, BOARD_PLANES, 8, 8))
    scalars = mm(os.path.join(DATA_DIR, "scalars.npy"), "w+", np.float32, (n_positions, N_SCALARS))
    wdl = mm(os.path.join(DATA_DIR, "wdl.npy"), "w+", np.int64, (n_positions,))
    blunder = mm(os.path.join(DATA_DIR, "blunder.npy"), "w+", np.float32, (n_positions, 2))
    is_val = mm(os.path.join(DATA_DIR, "is_val.npy"), "w+", np.bool_, (n_positions,))

    n = 0
    games_used = 0
    games_seen = 0
    t0 = time.time()
    for headers, movetext in iter_raw_games(pgn_path):
        games_seen += 1
        rows = list(game_rows(headers, movetext))
        if not rows:
            continue
        if len(rows) > per_game:
            rows = rng.sample(rows, per_game)
        val = zlib.crc32(headers.get("Site", "").encode()) % VAL_FRACTION_MOD == 0
        games_used += 1

        for row in rows:
            if n >= n_positions:
                break
            b, s = encode_position(
                row["fen"], row["white_clock"], row["black_clock"],
                row["white_elo"], row["black_elo"],
                row["base_time"], row["increment"],
                eval_cp=row["eval_cp"], eval_mate=row["eval_mate"],
            )
            white_to_move = row["fen"].split()[1] == "w"
            boards[n] = b
            scalars[n] = s
            wdl[n] = wdl_target(row["result"], white_to_move)
            blunder[n] = (row["is_blunder"], row["opp_blunder_next"])
            is_val[n] = val
            n += 1

        if games_used % 5000 == 0:
            rate = n / max(1e-9, time.time() - t0)
            print(f"  {n:,}/{n_positions:,} positions | {games_used:,} games used "
                  f"of {games_seen:,} read | {rate:,.0f} pos/s")
        if n >= n_positions:
            break

    for arr in (boards, scalars, wdl, blunder, is_val):
        arr.flush()
    meta = {
        "pgn": os.path.abspath(pgn_path),
        "requested_positions": n_positions,
        "positions": n,
        "per_game": per_game,
        "games_used": games_used,
        "label_version": LABEL_VERSION,
        "board_planes": BOARD_PLANE_NAMES,
        "scalar_names": SCALAR_NAMES,
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Dataset done: {n:,} positions from {games_used:,} games "
          f"in {time.time() - t0:.0f}s")


def dataset_is_current(n_positions: int, per_game: int) -> bool:
    try:
        with open(os.path.join(DATA_DIR, "meta.json")) as f:
            meta = json.load(f)
    except FileNotFoundError:
        return False
    return (meta["requested_positions"] == n_positions
            and meta["per_game"] == per_game
            and meta.get("label_version") == LABEL_VERSION
            and meta["board_planes"] == BOARD_PLANE_NAMES
            and meta["scalar_names"] == SCALAR_NAMES)


def load_dataset():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        n = json.load(f)["positions"]
    load = lambda name: np.load(os.path.join(DATA_DIR, name), mmap_mode="r")[:n]
    boards, scalars, wdl, blunder, is_val = (
        load("boards.npy"), load("scalars.npy"), load("wdl.npy"),
        load("blunder.npy"), load("is_val.npy"),
    )
    is_val = np.asarray(is_val)

    def split(mask):
        idx = np.nonzero(mask)[0]
        return (
            torch.from_numpy(np.ascontiguousarray(boards[idx])),
            torch.from_numpy(np.ascontiguousarray(scalars[idx])),
            torch.from_numpy(np.ascontiguousarray(wdl[idx])),
            torch.from_numpy(np.ascontiguousarray(blunder[idx])),
        )

    return split(~is_val), split(is_val)


# ===========================================================================
# Baselines (what the model has to beat)
# ===========================================================================
def eval_only_baseline(train, val, device) -> float:
    """Multinomial logistic regression on [eval, mate_flag] only: no clock, no Elo."""
    _, s_tr, w_tr, _ = train
    _, s_va, w_va, _ = val
    k = min(len(s_tr), 500_000)
    x_tr, y_tr = s_tr[:k, :2].to(device), w_tr[:k].to(device)
    lin = nn.Linear(2, 3).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(lin(x_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return F.cross_entropy(lin(s_va[:, :2].to(device)), w_va.to(device)).item()


def constant_blunder_baseline(train, val) -> float:
    rate = train[3].mean(0).clamp(1e-6, 1 - 1e-6)
    target = val[3]
    return F.binary_cross_entropy(rate.expand_as(target), target).item()


# ===========================================================================
# Training
# ===========================================================================
@torch.no_grad()
def evaluate(model, data, device, batch_size=8192):
    model.eval()
    boards, scalars, wdl, blunder = data
    totals = np.zeros(3)
    blunder_pred_sum = torch.zeros(2)
    n = len(wdl)
    for i in range(0, n, batch_size):
        b = boards[i:i + batch_size].to(device)
        s = scalars[i:i + batch_size].to(device)
        w = wdl[i:i + batch_size].to(device)
        bl = blunder[i:i + batch_size].to(device)
        wl, bll = model(b, s)
        _, wdl_loss, bl_loss = loss_fn(wl, bll, w, bl)
        m = len(w)
        totals += np.array([(wdl_loss + bl_loss).item(), wdl_loss.item(), bl_loss.item()]) * m
        blunder_pred_sum += torch.sigmoid(bll).sum(0).cpu()
    totals /= n
    return {
        "loss": float(totals[0]), "wdl_loss": float(totals[1]), "blunder_loss": float(totals[2]),
        "mean_pred_blunder": (blunder_pred_sum / n).tolist(),
        "true_blunder_rate": blunder.mean(0).tolist(),
    }


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(args) -> None:
    device = pick_device()
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    train_data, val_data = load_dataset()
    n_train = len(train_data[2])
    print(f"Train positions: {n_train:,} | validation positions: {len(val_data[2]):,}")
    print(f"Blunder rate (train): now {train_data[3][:, 0].mean():.4f}, "
          f"next {train_data[3][:, 1].mean():.4f}")

    base_wdl = eval_only_baseline(train_data, val_data, device)
    base_bl = constant_blunder_baseline(train_data, val_data)
    print(f"Baselines on validation: WDL loss (eval only) {base_wdl:.4f} | "
          f"blunder loss (constant rate) {base_bl:.4f}")

    config = {"n_scalars": N_SCALARS, "channels": args.channels, "blocks": args.blocks}
    model = ClockNet(**config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    steps_per_epoch = math.ceil(n_train / args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs, pct_start=0.1)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best = float("inf")
    boards, scalars, wdl, blunder = train_data

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n_train)
        running = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            b = boards[idx].to(device, non_blocking=True)
            s = scalars[idx].to(device, non_blocking=True)
            w = wdl[idx].to(device, non_blocking=True)
            bl = blunder[idx].to(device, non_blocking=True)

            wl, bll = model(b, s)
            loss, _, _ = loss_fn(wl, bll, w, bl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            running += loss.item()
            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch} step {step + 1}/{steps_per_epoch} "
                      f"train loss {running / 200:.4f}")
                running = 0.0

        metrics = evaluate(model, val_data, device)
        print(f"Epoch {epoch}: val WDL {metrics['wdl_loss']:.4f} (baseline {base_wdl:.4f}) | "
              f"val blunder {metrics['blunder_loss']:.4f} (baseline {base_bl:.4f}) | "
              f"{time.time() - t0:.0f}s")

        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": config,
                "board_planes": BOARD_PLANE_NAMES,
                "scalar_names": SCALAR_NAMES,
                "epoch": epoch,
                "metrics": metrics,
                "baselines": {"wdl_loss": base_wdl, "blunder_loss": base_bl},
            }, MODEL_PATH)
            print(f"  saved best model to {MODEL_PATH}")

    print("\nSanity checks on the best model:")
    sanity_report(load_model(MODEL_PATH, device), val_data, device)


# ===========================================================================
# Sanity checks: does the model agree with the data, feature by feature?
# ===========================================================================
@torch.no_grad()
def predict_batch(model, boards, scalars, device, batch_size=8192):
    p_wdl, p_bl = [], []
    for i in range(0, len(boards), batch_size):
        wl, bll = model(boards[i:i + batch_size].to(device), scalars[i:i + batch_size].to(device))
        p_wdl.append(torch.softmax(wl, 1).cpu())
        p_bl.append(torch.sigmoid(bll).cpu())
    return torch.cat(p_wdl).numpy(), torch.cat(p_bl).numpy()


def sanity_report(model, val_data, device) -> None:
    """
    For each group of positions, compare what the model predicts on average
    with what actually happened. If the two columns disagree a lot, or move in
    opposite directions, the model hasn't learnt that feature properly.
    """
    model.eval()
    boards, scalars, wdl, blunder = val_data
    p_wdl, p_bl = predict_batch(model, boards, scalars, device)
    s = scalars.numpy()
    true_bl = blunder.numpy()
    win = (wdl.numpy() == 0).astype(float)

    def table(title, values, edges, fmt):
        print(f"\n  {title}")
        print(f"    {'bucket':>16s} {'n':>7s} | {'blunder now':>17s} | {'opp blunder next':>17s} | {'win':>13s}")
        print(f"    {'':>16s} {'':>7s} | {'model':>8s} {'real':>8s} | {'model':>8s} {'real':>8s} | {'model':>6s} {'real':>6s}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values < hi)
            if m.sum() < 50:
                continue
            label = f"{fmt(lo)}-{fmt(hi)}"
            print(f"    {label:>16s} {m.sum():7d} | {p_bl[m, 0].mean():8.4f} {true_bl[m, 0].mean():8.4f} | "
                  f"{p_bl[m, 1].mean():8.4f} {true_bl[m, 1].mean():8.4f} | "
                  f"{p_wdl[m, 0].mean():6.3f} {win[m].mean():6.3f}")

    inf = float("inf")
    sec = lambda x: "inf" if x == inf else f"{x:.0f}s"
    elo = lambda x: "inf" if x == inf else ("-inf" if x == -inf else f"{x:.0f}")
    my_clock = np.expm1(s[:, 2])
    their_clock = np.expm1(s[:, 3])
    my_elo = s[:, 8] * 400 + 1500
    their_elo = s[:, 9] * 400 + 1500
    table("by MY clock (side to move)", my_clock, [0, 10, 30, 60, 180, inf], sec)
    table("by THEIR clock", their_clock, [0, 10, 30, 60, 180, inf], sec)
    table("by MY Elo", my_elo, [0, 1200, 1500, 1800, 2100, inf], elo)
    table("by THEIR Elo", their_elo, [0, 1200, 1500, 1800, 2100, inf], elo)
    table("by Elo difference (mine - theirs)", my_elo - their_elo,
          [-inf, -300, -100, 100, 300, inf], elo)
    table("by predicted win probability (calibration)", p_wdl[:, 0],
          [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01], lambda x: f"{min(x, 1):.1f}")

    # Pure what-if: change one input, keep everything else fixed.
    print("\n  What-if: shift one input for every validation position, average the predictions")
    sub = torch.arange(0, len(wdl), max(1, len(wdl) // 5000))
    b, base_s = boards[sub], scalars[sub]
    shifts = [
        ("my Elo +400", 8, 1.0), ("their Elo +400", 9, 1.0),
        ("my clock x3", 2, math.log(3)), ("my clock /3", 2, -math.log(3)),
    ]
    p0_wdl, p0_bl = predict_batch(model, b, base_s, device)
    print(f"    {'unchanged':>16s}: win {p0_wdl[:, 0].mean():.3f}  blunder now {p0_bl[:, 0].mean():.4f}"
          f"  opp next {p0_bl[:, 1].mean():.4f}")
    for name, col, delta in shifts:
        x = base_s.clone()
        x[:, col] += delta
        if col == 2:  # keep the clock fraction consistent with the log clock
            x[:, 4] = (x[:, 4] * math.exp(delta)).clamp(max=1.5)
        pw, pb = predict_batch(model, b, x, device)
        print(f"    {name:>16s}: win {pw[:, 0].mean():.3f}  blunder now {pb[:, 0].mean():.4f}"
              f"  opp next {pb[:, 1].mean():.4f}")


# ===========================================================================
# Loading + prediction for the live app
# ===========================================================================
def load_model(path: str = MODEL_PATH, device: str | None = None) -> ClockNet:
    device = device or pick_device()
    ckpt = torch.load(path, map_location=device)
    if ckpt["board_planes"] != BOARD_PLANE_NAMES or ckpt["scalar_names"] != SCALAR_NAMES:
        raise RuntimeError("Model was trained with a different input encoding than main.py. "
                           "Retrain with: python train.py --rebuild")
    model = ClockNet(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: ClockNet, fen: str, white_clock: float, black_clock: float,
            white_elo: float, black_elo: float, base_time: float, increment: float,
            eval_cp: float | None = None, eval_mate: int | None = None,
            engine_depth: int = 12) -> dict:
    """
    Raw inputs -> probabilities, converted back to White / Black.

    If neither eval_cp nor eval_mate is given, Stockfish evaluates the
    position (see engine.py). The eval used is returned too, for the eval bar.
    """
    if eval_cp is None and eval_mate is None:
        from engine import evaluate  # imported here so training never needs Stockfish
        eval_cp, eval_mate = evaluate(fen, depth=engine_depth)

    b, s = encode_position(fen, white_clock, black_clock, white_elo, black_elo,
                           base_time, increment, eval_cp=eval_cp, eval_mate=eval_mate)
    device = next(model.parameters()).device
    wl, bll = model(torch.from_numpy(b)[None].to(device), torch.from_numpy(s)[None].to(device))
    win, draw, loss = torch.softmax(wl, 1)[0].tolist()
    blunder_now, blunder_next = torch.sigmoid(bll)[0].tolist()

    if fen.split()[1] == "w":
        result = {"white_win": win, "draw": draw, "black_win": loss,
                  "white_blunder": blunder_now, "black_blunder": blunder_next}
    else:
        result = {"white_win": loss, "draw": draw, "black_win": win,
                  "white_blunder": blunder_next, "black_blunder": blunder_now}
    result["eval_cp"] = eval_cp      # White's point of view
    result["eval_mate"] = eval_mate
    return result


# ===========================================================================
# CLI
# ===========================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Build dataset and train ClockNet.")
    p.add_argument("--pgn", default=DEFAULT_PGN)
    p.add_argument("--positions", type=int, default=2_000_000)
    p.add_argument("--per-game", type=int, default=8)
    p.add_argument("--rebuild", action="store_true", help="re-read the PGN even if data/ exists")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true",
                   help="only run the sanity checks on the saved model, no training")
    args = p.parse_args()

    if args.check:
        device = pick_device()
        _, val_data = load_dataset()
        sanity_report(load_model(MODEL_PATH, device), val_data, device)
        return

    if args.rebuild or not dataset_is_current(args.positions, args.per_game):
        print(f"Building dataset from {args.pgn} ...")
        build_dataset(args.pgn, args.positions, args.per_game, args.seed)
    else:
        print("Using cached dataset in data/")
    train(args)


if __name__ == "__main__":
    main()
