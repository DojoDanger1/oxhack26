def predict_blunder(request) -> float:
    """Return next-move blunder probability in [0, 1] for the side to move.

    The complete validated request contains fen plus time, rating, and control;
    each list contains exactly two integers. Agree units and ordering with the
    model owner. Load model weights once at server startup.
    """
    fen = request.fen
    time = request.time
    rating = request.rating
    control = request.control
    # Convert all four fields into neural-network inputs here.
    raise NotImplementedError("Connect the trained neural network here")
