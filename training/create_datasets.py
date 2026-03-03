from utils.build_raw_dataset import build_raw_dataset
from utils.generate_features import process_json
import json

if __name__ == "__main__":

    # ---------- TEXT ----------
    text_svg_folder = "./data/raw_svg_text"
    text_raw_path = "./data/dataset_raw_text.json"
    text_feat_path = "./data/dataset_features_text.json"

    build_raw_dataset(text_svg_folder, text_raw_path, label="text")
    process_json(text_raw_path, text_feat_path)

    # ---------- NON TEXT ----------
    non_svg_folder = "./data/raw_svg_nontext"
    non_raw_path = "./data/dataset_raw_nontext.json"
    non_feat_path = "./data/dataset_features_nontext.json"

    build_raw_dataset(non_svg_folder, non_raw_path, label="non_text")
    process_json(non_raw_path, non_feat_path)

    # ---------- MERGE ----------
    with open(text_feat_path) as f:
        text_data = json.load(f)

    with open(non_feat_path) as f:
        non_data = json.load(f)

    all_data = text_data + non_data

    with open("./data/dataset_features.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print("Pipeline completed.")