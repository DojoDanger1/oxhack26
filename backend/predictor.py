"""Load the unchanged yehor model and return probabilities for both colours."""
import builtins
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock

MODEL_DIR = Path(__file__).resolve().parent.parent / "yehor"
MODEL_LOCK = Lock()


def load_module(name, path, imports=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if imports:
        original_import = builtins.__import__

        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in imports:
                return imports[name]
            return original_import(name, globals, locals, fromlist, level)

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
def get_runtime():
    encoder = load_module("_blundometer_encoder", MODEL_DIR / "main.py")
    reader = load_module("_blundometer_reader", MODEL_DIR / "database_reader.py")
    training = load_module(
        "_blundometer_training", MODEL_DIR / "train.py",
        imports={"main": encoder, "database_reader": reader},
    )
    model = training.load_model(
        path=str(MODEL_DIR / "models" / "clocknet.pt"), device="cpu",
    )
    return model, training.predict


def predict_blunder(request) -> dict[str, float]:
    """time/rating: [White, Black]; control: [base seconds, increment seconds].

    The model uses zero for the engine evaluation when it is not supplied.
    The side-to-move probability refers to the current move; the opponent's
    probability refers to its reply. These are independent probabilities.
    """
    with MODEL_LOCK:
        model, predict = get_runtime()
        result = predict(
            model=model, fen=request.fen,
            white_clock=request.time[0], black_clock=request.time[1],
            white_elo=request.rating[0], black_elo=request.rating[1],
            base_time=request.control[0], increment=request.control[1],
        )
    return {
        "white_blunder": float(result["white_blunder"]),
        "black_blunder": float(result["black_blunder"]),
    }
