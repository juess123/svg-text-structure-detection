import json
import numpy as np

FEATURE_ORDER = [
    "direction_change_ratio",
    "small_segment_ratio",
    "fill_ratio",
    "subpath_density",
    "point_density",
    "avg_segment_length",
    "segment_length_std",
    "direction_variance",
    "curvature_std",

    "log_aspect_ratio",
    "length_per_bbox_area",
    "compactness2",
    "sharp_turn_density",
    "subpath_count",
    "closed_subpath_ratio",
    "line_ratio",
    "curve_ratio",
    "move_ratio",
    "close_ratio",
]

def load_features(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = []
    for idx, item in enumerate(data):
        features = item["features"]

        missing = [k for k in FEATURE_ORDER if k not in features]
        if missing:
            raise KeyError(
                f"Sample idx={idx} missing feature keys: {missing}"
            )

        vec = [features[k] for k in FEATURE_ORDER]
        X.append(vec)

    return np.array(X, dtype=np.float32)