import os
import gc
import itertools
from lxml import etree


def add_ids_to_svg(svg_path: str) -> str:
    """为SVG文件中所有控件添加统一格式的连续ID"""

    # 创建新文件名
    base, ext = os.path.splitext(svg_path)
    new_svg_path = f"{base}_addid{ext}"

    try:
        # XML解析器
        parser = etree.XMLParser(
            remove_blank_text=True,
            huge_tree=True,
            recover=True
        )

        # 解析文档
        tree = etree.parse(svg_path, parser)
        root = tree.getroot()

        # 需要处理的元素
        SVG_ELEMENTS = {
            "rect", "circle", "ellipse", "line",
            "polyline", "polygon", "path", "text", "g", "image"
        }

        counter = itertools.count(1)
        element_count = 0

        for elem in root.iter():
            tag = etree.QName(elem).localname

            if tag in SVG_ELEMENTS:
                new_id = f"addid{next(counter):04d}"
                elem.set("id", new_id)
                element_count += 1

                # 每 10000 清理一次
                if element_count % 10000 == 0:
                    gc.collect()
                    print(f"已处理 {element_count} 个元素...")

        # 保存
        tree.write(
            new_svg_path,
            pretty_print=True,
            encoding="utf-8",
            xml_declaration=True
        )

        # 清理内存
        root.clear()
        del root
        del tree
        gc.collect()

    except Exception as e:
        if os.path.exists(new_svg_path):
            os.remove(new_svg_path)
        raise e

    finally:
        # 防止未释放
        gc.collect()

    return new_svg_path