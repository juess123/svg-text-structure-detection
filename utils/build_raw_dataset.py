import os
import json
from utils.extract_svg_paths import extract_paths_from_svg  # 你的函数文件路径

def build_raw_dataset(svg_folder, output_path, label="text"):
    all_data = []

    for filename in os.listdir(svg_folder):
        if not filename.lower().endswith(".svg"):
            continue

        svg_path = os.path.join(svg_folder, filename)
        print(f"Processing SVG: {filename}")

        records = extract_paths_from_svg(svg_path, label=label)
        all_data.extend(records)

    print(f"Total paths collected: {len(all_data)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Saved raw dataset to: {output_path}")