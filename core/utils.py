def classify_risk_and_color(cl_score: float):
    """Classify risk and assign a color based on the EWCL score."""
    if cl_score is None:
        return {"risk_class": "unknown", "color_hex": "#808080"}  # Grey for unknown

    # New 5-color palette: Blue -> Green -> Yellow -> Purple -> Red
    colours = ["#4285F4", "#34A853", "#FBBC05", "#A142F4", "#EA4335"]
    risk_levels = ["very_low", "low", "moderate", "high", "very_high"]
    
    if 0.0 <= cl_score < 0.2:
        index = 0
    elif 0.2 <= cl_score < 0.4:
        index = 1
    elif 0.4 <= cl_score < 0.6:
        index = 2
    elif 0.6 <= cl_score < 0.8:
        index = 3
    elif 0.8 <= cl_score <= 1.0:
        index = 4
    else:  # Fallback for scores outside [0,1]
        return {"risk_class": "out_of_range", "color_hex": "#000000"}  # Black for out of range

    return {"risk_class": risk_levels[index], "color_hex": colours[index]}
