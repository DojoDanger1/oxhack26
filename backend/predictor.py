def predict_blunder(request) -> float:
    fen = request.fen
    time_a = request.timeA
    time_b = request.timeB
    elo_a = request.eloA
    elo_b = request.eloB
    """Return next-move blunder probability in [0, 1] for the side to move.

    Replace with neural-network inference. Load weights once at server startup.
    Agree chess variant, blunder definition and extra inputs with the model owner.
    """
    raise NotImplementedError("Connect the trained neural network here")
