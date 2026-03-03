import xml.etree.ElementTree as ET
import json
import os


def extract_paths_from_svg(svg_path, label="text"):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 处理命名空间
    namespace = ""
    if "}" in root.tag:
        namespace = root.tag.split("}")[0] + "}"

    file_id = os.path.splitext(os.path.basename(svg_path))[0]

    # ==============================
    # 1️⃣ 读取 viewBox 并计算 svg_area
    # ==============================
    viewbox = root.attrib.get("viewBox")

    svg_area = None

    if viewbox:
        parts = viewbox.strip().split()
        if len(parts) == 4:
            _, _, vb_width, vb_height = parts
            svg_area = float(vb_width) * float(vb_height)

    # ==============================

    data = []

    # 2️⃣ 找第一个 <g>
    first_g = root.find(f"{namespace}g")
    if first_g is None:
        return []

    path_index = 0

    # 3️⃣ 从这个 g 开始递归查找 path
    for elem in first_g.iter():
        if elem.tag == f"{namespace}path":
            raw_d = elem.attrib.get("d")
            if raw_d is None:
                continue

            record = {
                "file_id": file_id,
                "path_index": path_index,
                "label": label,
                "raw_d": raw_d,
                "svg_area": svg_area   # ✅ 只存面积
            }

            data.append(record)
            path_index += 1

    return data


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)