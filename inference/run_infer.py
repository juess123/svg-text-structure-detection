from inference.model_loader import load_model
from inference.svgaddid import add_ids_to_svg
from inference.svg_text_extractor import extract_svg_paths
from inference.text_infer import infer_text_elements
from inference.dxf_exporter import export_text_to_dxf


def main():
    model, scaler = load_model()

    input_svg = "source13.svg"

    text_dxf = "text_paths.dxf"
    art_text_dxf = "art_text_paths.dxf"
    non_text_dxf = "non_text_paths.dxf"

    # 1. 预处理：加 id
    svg_with_id = add_ids_to_svg(input_svg)
    print("使用的新 SVG 文件:", svg_with_id)

    # 2. 提取 path
    svg_paths = extract_svg_paths(svg_with_id)

    # 3. 三分类推理
    text_results, art_text_results, non_text_results = infer_text_elements(
        model, scaler, svg_paths
    )

    # 4. 导出 3 个 DXF
    export_text_to_dxf(text_results, text_dxf)
    export_text_to_dxf(art_text_results, art_text_dxf)
    export_text_to_dxf(non_text_results, non_text_dxf)

    print("text:", len(text_results))
    print("art_text:", len(art_text_results))
    print("non_text:", len(non_text_results))

    print("DXF输出:")
    print(" -", text_dxf)
    print(" -", art_text_dxf)
    print(" -", non_text_dxf)

    print("\n===== TEXT SAMPLES =====")
    for item in text_results[:10]:
        print(
            f"id={item['id']}, "
            f"pred={item['pred_label']}, "
            f"conf={item['confidence']:.4f}"
        )

    print("\n===== ART_TEXT SAMPLES =====")
    for item in art_text_results[:10]:
        print(
            f"id={item['id']}, "
            f"pred={item['pred_label']}, "
            f"conf={item['confidence']:.4f}"
        )

    print("\n===== NON_TEXT SAMPLES =====")
    for item in non_text_results[:10]:
        print(
            f"id={item['id']}, "
            f"pred={item['pred_label']}, "
            f"conf={item['confidence']:.4f}"
        )


if __name__ == "__main__":
    main()