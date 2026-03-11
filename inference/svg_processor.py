import ezdxf

from inference.svg_text_extractor import extract_svg_paths
from inference.text_size import compute_path_size, normalize_size, classify_size
from inference.text_classifier import classify_path
from features.feature_pipeline import extract_features_from_raw_d
from inference.sampling import parse_path_d_multi


def process_svg(model, scaler, input_svg, output_dxf):

    doc = ezdxf.new()

    doc.layers.new(name="TEXT_SMALL", dxfattribs={"color": 3})
    doc.layers.new(name="TEXT_MEDIUM", dxfattribs={"color": 5})
    doc.layers.new(name="TEXT_LARGE", dxfattribs={"color": 1})

    msp = doc.modelspace()

    text_count = 0
    uncertain_count = 0

    text_paths = []
    sizes = []

    # ======================
    # 读取 SVG path
    # ======================

    svg_paths = extract_svg_paths(input_svg)

    # ======================
    # 第一轮：检测文字 + 收集size
    # ======================

    for d in svg_paths:

        features_dict = extract_features_from_raw_d(d)

        result = classify_path(model, scaler, features_dict)

        if result == "text":

            text_count += 1

            paths = parse_path_d_multi(d)

            for path in paths:

                if len(path) < 2:
                    continue

                size = compute_path_size(path)

                text_paths.append(path)
                sizes.append(size)

        elif result == "uncertain":

            uncertain_count += 1

    if not sizes:
        print("没有检测到文字")
        return

    # ======================
    # 计算size范围
    # ======================

    min_size = min(sizes)
    max_size = max(sizes)
    # ======================
    # 第二轮：归一化 + 分类 + 画DXF
    # ======================

    for path, size in zip(text_paths, sizes):

        norm_size = normalize_size(size, min_size, max_size)
        #print(f"text size:{norm_size}")
        size_type = classify_size(norm_size)

        layer = f"TEXT_{size_type}"

        pts = [(x, -y) for x, y in path]

        msp.add_lwpolyline(
            pts,
            dxfattribs={"layer": 0}
        )

    doc.saveas(output_dxf)

    print("text:", text_count)
    print("uncertain:", uncertain_count)
    print("DXF输出:", output_dxf)