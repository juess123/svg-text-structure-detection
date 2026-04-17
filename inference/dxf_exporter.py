import ezdxf


def compute_element_size(paths):
    all_points = []

    for path in paths:
        if not path or len(path) < 2:
            continue
        all_points.extend(path)

    if not all_points:
        return 0.0

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return max(width, height)


def group_sizes_by_tolerance(items, tolerance=0.2):
    """
    items: [{"id":..., "size":..., "paths":...}, ...]
    tolerance: 相对误差阈值，例如 0.2 表示 20%

    返回:
    groups = [
        {
            "ref_size": ...,
            "members": [item1, item2, ...]
        },
        ...
    ]
    """
    if not items:
        return []

    items = sorted(items, key=lambda x: x["size"])
    groups = []

    current_group = {
        "ref_size": items[0]["size"],
        "members": [items[0]],
    }

    for item in items[1:]:
        ref_size = current_group["ref_size"]
        size = item["size"]

        rel_error = abs(size - ref_size) / ref_size if ref_size > 1e-9 else 0.0

        if rel_error <= tolerance:
            current_group["members"].append(item)

            # 更新当前组参考值：用组内均值更稳
            current_group["ref_size"] = (
                sum(m["size"] for m in current_group["members"])
                / len(current_group["members"])
            )
        else:
            groups.append(current_group)
            current_group = {
                "ref_size": size,
                "members": [item],
            }

    groups.append(current_group)
    return groups


def export_text_to_dxf(text_results, output_dxf, tolerance=0.2):
    doc = ezdxf.new()
    msp = doc.modelspace()

    # ======================
    # 先收集所有元素级 size
    # ======================
    element_items = []

    for item in text_results:
        elem_id = item.get("id", "NO_ID")
        paths = item.get("paths", [])

        elem_size = compute_element_size(paths)
        if elem_size <= 0:
            continue

        element_items.append({
            "id": elem_id,
            "paths": paths,
            "size": elem_size,
        })

    if not element_items:
        print("没有检测到文字")
        return

    # ======================
    # 按 20% 误差自动分组
    # ======================
    groups = group_sizes_by_tolerance(element_items, tolerance=tolerance)

    # ======================
    # 为每个组创建 layer
    # ======================
    # ACI 常用颜色循环
    #aci_colors = [1, 3, 5, 6, 2, 4, 30, 140, 200, 7]
    aci_colors = [7]

    for idx, group in enumerate(groups, start=1):
        layer_name = f"TEXT_G{idx:02d}"
        color = aci_colors[(idx - 1) % len(aci_colors)]

        if layer_name not in doc.layers:
            doc.layers.new(name=layer_name, dxfattribs={"color": color})

        

        for member in group["members"]:
            elem_id = member["id"]
            elem_size = member["size"]
            paths = member["paths"]

           

            for path in paths:
                if not path or len(path) < 2:
                    continue

                pts = [(x, -y) for x, y in path]

                msp.add_lwpolyline(
                    pts,
                    dxfattribs={"layer": layer_name}
                )

    doc.saveas(output_dxf)