import torch
import numpy as np
import pickle
import xml.etree.ElementTree as ET

from models.text_energy_model import TextEnergyModel
from features.feature_pipeline import extract_features_from_raw_d


# ========= 1️⃣ 加载模型 =========
def load_model():
    model = TextEnergyModel(feature_dim=18)
    model.load_state_dict(torch.load("models/text_energy_binary.pth"))
    model.eval()

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ========= 2️⃣ 判断是否为文字 =========
def is_text_path(model, scaler, features_dict, threshold=15.0):

    # ⚠ 顺序必须和训练一致
    feature_vector = np.array([
        features_dict["direction_change_ratio"],
        features_dict["turning_density"],
        features_dict["small_segment_ratio"],
        features_dict["relative_bbox_area"],
        features_dict["fill_ratio"],
        features_dict["curve_ratio"],
        features_dict["avg_cmd_per_subpath"],
        features_dict["compactness"],
        features_dict["normalized_length"],
        features_dict["mean_curvature"],
        features_dict["curvature_std"],
        features_dict["command_density"],
        features_dict["subpath_density"],
        features_dict["point_density"],
        features_dict["avg_segment_length"],
        features_dict["segment_length_std"],
        features_dict["direction_variance"],
        features_dict["command_entropy"],
    ], dtype=np.float32)

    # 标准化
    feature_vector = (feature_vector - scaler.mean) / scaler.std

    X_tensor = torch.tensor([feature_vector])

    with torch.no_grad():
        _, energy = model(X_tensor)

    return energy.item() < threshold


# ========= 3️⃣ 处理 SVG =========
def process_svg(input_svg, output_svg):

    model, scaler = load_model()

    tree = ET.parse(input_svg)
    root = tree.getroot()

    text_count = 0

    for elem in root.iter():
        if elem.tag.endswith("path"):

            d = elem.attrib.get("d")
            if not d:
                continue

            features_dict = extract_features_from_raw_d(d)

            if is_text_path(model, scaler, features_dict):
                elem.set("class",  "#14DB1E")
                text_count += 1

    tree.write(output_svg)
    print(f"检测到文字 path 数量: {text_count}")
    print("输出文件:", output_svg)


# ========= 4️⃣ 主函数 =========
if __name__ == "__main__":
    process_svg("input.svg", "output.svg")