def low_quality(q, threshold):
    import math
    if not math.isfinite(q) or not 0 <= q <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Expected finite probability-scale quality")
    return q < threshold
