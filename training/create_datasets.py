from utils.build_raw_dataset import build_raw_dataset
from utils.generate_features import process_json
import json


if __name__ == "__main__":

    configs = [
        {
            "svg_folder": "./data/raw_svg_text",
            "raw_path": "./data/dataset_raw_text.json",
            "feat_path": "./data/dataset_features_text.json",
            "label": "text",
        },
        {
            "svg_folder": "./data/raw_svg_art_text",
            "raw_path": "./data/dataset_raw_art_text.json",
            "feat_path": "./data/dataset_features_art_text.json",
            "label": "art_text",
        },
        {
            "svg_folder": "./data/raw_svg_nontext",
            "raw_path": "./data/dataset_raw_nontext.json",
            "feat_path": "./data/dataset_features_nontext.json",
            "label": "non_text",
        },
    ]

    all_data = []

    for cfg in configs:
        print(f"\n===== Processing: {cfg['label']} =====")
        build_raw_dataset(cfg["svg_folder"], cfg["raw_path"], label=cfg["label"])
        process_json(cfg["raw_path"], cfg["feat_path"])

        with open(cfg["feat_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)

        print(f"{cfg['label']} done: {len(data)} samples")

    with open("./data/dataset_features.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print("\nPipeline completed.")
    print(f"Total merged samples: {len(all_data)}")