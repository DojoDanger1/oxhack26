import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import load_model, predict
model = load_model()
print(predict(model, "r1bqkbnr/1ppp1ppp/p7/4n2Q/4P3/2P5/PP1P1PPP/RNB1KB1R b KQkq - 0 1", 300, 300, 1200, 2100, 600, 0, eval_cp=35))