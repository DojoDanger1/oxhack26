import math
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StrictInt
from predictor import predict_blunder

app = FastAPI(title="Blundometer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if s.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class PredictionRequest(BaseModel):
    fen: str = Field(min_length=1, max_length=512, pattern=r"\S")
    time: list[StrictInt] = Field(min_length=2, max_length=2)
    rating: list[StrictInt] = Field(min_length=2, max_length=2)
    control: list[StrictInt] = Field(min_length=2, max_length=2)

class PredictionResponse(BaseModel):
    white_win: float = Field(ge=0, le=1)
    draw: float = Field(ge=0, le=1)
    black_win: float = Field(ge=0, le=1)
    white_blunder: float = Field(ge=0, le=1)
    black_blunder: float = Field(ge=0, le=1)
    demo: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    demo = os.getenv("BLUNDOMETER_DEMO", "0") == "1"
    try:
        result = (
            {"white_win": 1, "draw": 0, "black_win": 0, "white_blunder": 0.32, "black_blunder": 0.24}
            if demo else predict_blunder(request)
        )
        for key in ("white_win", "draw", "black_win","white_blunder", "black_blunder",):
            probability = result[key]
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid model probability: {key}")
    except NotImplementedError:
        raise HTTPException(503, "Prediction model is not connected") from None
    except Exception:
        logging.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed") from None
    return PredictionResponse(**result, demo=demo)

