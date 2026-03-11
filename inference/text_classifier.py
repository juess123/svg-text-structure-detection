import numpy as np
import torch


def classify_path(model, scaler, features_dict, low_th=5.3, high_th=9.3):

    feature_vector = np.array([
        features_dict["direction_change_ratio"],
        features_dict["small_segment_ratio"],
        features_dict["fill_ratio"],
        features_dict["subpath_density"],
        features_dict["point_density"],
        features_dict["avg_segment_length"],
        features_dict["segment_length_std"],
        features_dict["direction_variance"],
        features_dict["curvature_std"],
    ], dtype=np.float32)

    feature_vector = scaler.transform(feature_vector.reshape(1, -1))[0]

    X_tensor = torch.from_numpy(feature_vector).float().unsqueeze(0)

    with torch.no_grad():
        _, energy = model(X_tensor)

    e = energy.item()

    if e < low_th:
        return "text"
    elif e > high_th:
        return "non-text"
    else:
        return "uncertain"