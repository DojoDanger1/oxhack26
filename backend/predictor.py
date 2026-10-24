"""Adapt API requests to the unchanged model code in ../yehor."""

import builtins
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Protocol, TypedDict

MODEL_DIR = Path(__file__).resolve().parent.parent / "yehor"
MODEL_PATH = MODEL_DIR / "models" / "clocknet.pt"
_MODEL_LOCK = Lock()


class PredictionInput(Protocol):
    """Fields supplied by the API's validated PredictionRequest."""

    fen: str
    time: list[int]       # [White seconds, Black seconds]
    rating: list[int]     # [White Elo, Black Elo]
    control: list[int]    # [base seconds, increment seconds]


class BlunderResult(TypedDict):
    white_win: float
    draw: float
    black_win: float
    white_blunder: float
    black_blunder: float


def _load_module(
    name: str, filename: str, imports: dict[str, ModuleType] | None = None,
) -> ModuleType:
    """Load under a private name; redirect only this module's legacy imports.

    train.py imports 'main', which otherwise collides with backend/main.py.
    Never replace sys.modules['main'] or the global builtins import function.
    """
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model module: {filename}")
    module = importlib.util.module_from_spec(spec)

    if imports:
        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in imports:
                return imports[name]
            return builtins.__import__(name, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = {
            **vars(builtins), "__import__": local_import,
        }

    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@lru_cache(maxsize=1)
def _get_runtime():
    """Load weights on the first real request; reuse them per server process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    encoder = _load_module("_blundometer_encoder", "main.py")
    reader = _load_module("_blundometer_reader", "database_reader.py")
    training = _load_module(
        "_blundometer_training", "train.py",
        imports={"main": encoder, "database_reader": reader},
    )
    model = training.load_model(path=str(MODEL_PATH), device="cpu")
    return model, training.predict


def predict_blunder(request: PredictionInput) -> BlunderResult:
    """Return separate White/Black probabilities, each in [0, 1].

    The mover's probability refers to its current move; the other probability
    refers to the reply. With no engine eval supplied, yehor uses zero.
    """
    with _MODEL_LOCK:
        model, predict = _get_runtime()
        result = predict(
            model=model,
            fen=request.fen,
            white_clock=request.time[0],
            black_clock=request.time[1],
            white_elo=request.rating[0],
            black_elo=request.rating[1],
            base_time=request.control[0],
            increment=request.control[1],
        )

    return result
