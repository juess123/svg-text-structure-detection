import numpy as np
import torch

FEATURE_KEYS = [
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

LABELS = ["text", "art_text", "non_text"]


def classify_path(model, scaler, features_dict):
    feature_vector = np.array(
        [features_dict[k] for k in FEATURE_KEYS],
        dtype=np.float32
    )

    feature_vector = scaler.transform(feature_vector.reshape(1, -1))[0]
    x_tensor = torch.from_numpy(feature_vector).float().unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tensor)                  # [1, 3]
        probs = torch.softmax(logits, dim=1)     # [1, 3]
        conf, pred = torch.max(probs, dim=1)

    pred_idx = pred.item()
    pred_label = LABELS[pred_idx]
    confidence = conf.item()

    return {
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": probs[0].tolist(),
    }