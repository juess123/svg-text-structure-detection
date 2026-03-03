import json
import numpy as np

FEATURE_ORDER = [
    "direction_change_ratio",
    "turning_density",
    "small_segment_ratio",
    "relative_bbox_area",
    "fill_ratio",
    "curve_ratio",
    "avg_cmd_per_subpath",
    "compactness",
    "normalized_length",
    "mean_curvature",
    "curvature_std",
    "command_density",
    "subpath_density",
    "point_density",
    "avg_segment_length",
    "segment_length_std",
    "direction_variance",
    "command_entropy"
]

def load_features(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = []
    for item in data:
        features = item["features"]
        vec = [features[k] for k in FEATURE_ORDER]
        X.append(vec)

    return np.array(X, dtype=np.float32)