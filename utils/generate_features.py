import json
from features.feature_pipeline import extract_features_from_raw_d

def process_json(input_path, output_path):
    """
    读取 raw_json 文件
    提取所有 raw_d 的特征
    写入新的 features_json 文件
    """

    # 1️⃣ 读取原始数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    # 2️⃣ 遍历每一条 path
    for record in data:

        raw_d = record.get("raw_d")
        label = record.get("label")

        if not raw_d:
            continue

        features = extract_features_from_raw_d(
            raw_d,
            record.get("svg_area")
        )

        # 合并结果
        new_record = {
            "file_id": record.get("file_id"),
            "path_index": record.get("path_index"),
            "label": label,
            "features": features
        }

        results.append(new_record)

    # 3️⃣ 保存特征文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} paths.")
    return features