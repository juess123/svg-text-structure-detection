from features.feature_pipeline import extract_features_from_raw_d
from inference.text_classifier import classify_path
from inference.sampling import parse_path_d_multi


def infer_text_elements(model, scaler, svg_paths):
    text_results = []
    art_text_results = []
    non_text_results = []

    CONF_TH = 0.8

    for item in svg_paths:
        elem_id = item["id"]
        d = item["d"]

        features_dict = extract_features_from_raw_d(d)
        result = classify_path(model, scaler, features_dict)

        pred_label = result["pred_label"]
        confidence = result["confidence"]
        probs = result["probs"]

        # 先做置信度过滤
        if confidence < CONF_TH:
            continue

        raw_paths = parse_path_d_multi(d)
        path_items = []

        for path in raw_paths:
            if len(path) < 2:
                continue
            path_items.append(path)

        out_item = {
            "id": elem_id,
            "d": d,
            "paths": path_items,
            "pred_label": pred_label,
            "confidence": confidence,
            "probs": probs,
            "features": features_dict,
        }

        if pred_label == "text":
            text_results.append(out_item)

        elif pred_label == "art_text":
            art_text_results.append(out_item)

        elif pred_label == "non_text":
            non_text_results.append(out_item)

    return text_results, art_text_results, non_text_results