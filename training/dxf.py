import json
import os
import numpy as np
import ezdxf
import re

from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional

Point = Tuple[float, float]
SegmentType = Literal["line", "quadratic_bezier", "cubic_bezier"]


@dataclass
class Segment:
    type: SegmentType
    p0: Point
    p1: Optional[Point] = None
    p2: Optional[Point] = None
    p3: Optional[Point] = None


@dataclass
class Path:
    segments: List[Segment]
    closed: bool = False


def _is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def round_point(p, ndigits=3):
    return (round(p[0], ndigits), round(p[1], ndigits))


def make_point(x, y):
    return round_point((x, y))


def parse_path_d_multi(d):
    tokens = re.findall(
        r"""[MLCQZHVmlcqzhv]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?""",
        d,
        re.VERBOSE
    )

    paths = []
    current_segments = []

    x = y = 0.0
    start = None
    i = 0
    cmd = None

    def add_line(p0, p1):
        if p0 == p1:
            return
        current_segments.append(Segment(type="line", p0=p0, p1=p1))

    while i < len(tokens):
        t = tokens[i]

        if t in ("Z", "z"):
            if current_segments and start is not None:
                p0 = make_point(x, y)

                if p0 != start:
                    add_line(p0, start)

                x, y = start
                paths.append(Path(segments=current_segments, closed=True))
                current_segments = []
                start = None

            i += 1
            continue

        if t in "MLCQHVmlcqhv":
            cmd = t
            i += 1
            continue

        if cmd == "M":
            if current_segments:
                paths.append(Path(segments=current_segments, closed=False))
            current_segments = []

            x, y = float(tokens[i]), float(tokens[i + 1])
            x, y = make_point(x, y)
            start = (x, y)
            i += 2

            while i + 1 < len(tokens) and _is_number(tokens[i]):
                p0 = make_point(x, y)
                x, y = float(tokens[i]), float(tokens[i + 1])
                x, y = make_point(x, y)
                p1 = (x, y)
                add_line(p0, p1)
                i += 2

        elif cmd == "m":
            if current_segments:
                paths.append(Path(segments=current_segments, closed=False))
            current_segments = []

            x += float(tokens[i])
            y += float(tokens[i + 1])
            x, y = make_point(x, y)
            start = (x, y)
            i += 2

            while i + 1 < len(tokens) and _is_number(tokens[i]):
                p0 = make_point(x, y)
                x += float(tokens[i])
                y += float(tokens[i + 1])
                x, y = make_point(x, y)
                p1 = (x, y)
                add_line(p0, p1)
                i += 2

        elif cmd in ("L", "l"):
            while i + 1 < len(tokens) and _is_number(tokens[i]):
                p0 = make_point(x, y)

                if cmd == "L":
                    x, y = float(tokens[i]), float(tokens[i + 1])
                else:
                    x += float(tokens[i])
                    y += float(tokens[i + 1])

                x, y = make_point(x, y)
                p1 = (x, y)
                add_line(p0, p1)
                i += 2

        elif cmd in ("H", "h"):
            while i < len(tokens) and _is_number(tokens[i]):
                p0 = make_point(x, y)

                if cmd == "H":
                    x = float(tokens[i])
                else:
                    x += float(tokens[i])

                x, y = make_point(x, y)
                p1 = (x, y)
                add_line(p0, p1)
                i += 1

        elif cmd in ("V", "v"):
            while i < len(tokens) and _is_number(tokens[i]):
                p0 = make_point(x, y)

                if cmd == "V":
                    y = float(tokens[i])
                else:
                    y += float(tokens[i])

                x, y = make_point(x, y)
                p1 = (x, y)
                add_line(p0, p1)
                i += 1

        elif cmd in ("Q", "q"):
            while i + 3 < len(tokens) and _is_number(tokens[i]):
                p0 = make_point(x, y)

                if cmd == "Q":
                    p1 = make_point(float(tokens[i]), float(tokens[i + 1]))
                    p2 = make_point(float(tokens[i + 2]), float(tokens[i + 3]))
                else:
                    p1 = make_point(x + float(tokens[i]), y + float(tokens[i + 1]))
                    p2 = make_point(x + float(tokens[i + 2]), y + float(tokens[i + 3]))

                current_segments.append(
                    Segment(type="quadratic_bezier", p0=p0, p1=p1, p2=p2)
                )
                x, y = p2
                i += 4

        elif cmd in ("C", "c"):
            while i + 5 < len(tokens) and _is_number(tokens[i]):
                p0 = make_point(x, y)

                if cmd == "C":
                    p1 = make_point(float(tokens[i]), float(tokens[i + 1]))
                    p2 = make_point(float(tokens[i + 2]), float(tokens[i + 3]))
                    p3 = make_point(float(tokens[i + 4]), float(tokens[i + 5]))
                else:
                    p1 = make_point(x + float(tokens[i]), y + float(tokens[i + 1]))
                    p2 = make_point(x + float(tokens[i + 2]), y + float(tokens[i + 3]))
                    p3 = make_point(x + float(tokens[i + 4]), y + float(tokens[i + 5]))

                current_segments.append(
                    Segment(type="cubic_bezier", p0=p0, p1=p1, p2=p2, p3=p3)
                )
                x, y = p3
                i += 6

        else:
            i += 1

    if current_segments:
        paths.append(Path(segments=current_segments, closed=False))

    return paths


def sample_quadratic_bezier(p0, p1, p2, steps=24):
    pts = []
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    for t in np.linspace(0, 1, steps + 1):
        pt = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
        pts.append((float(pt[0]), float(pt[1])))
    return pts


def sample_cubic_bezier(p0, p1, p2, p3, steps=24):
    pts = []
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)

    for t in np.linspace(0, 1, steps + 1):
        pt = (
            ((1 - t) ** 3) * p0
            + 3 * ((1 - t) ** 2) * t * p1
            + 3 * (1 - t) * (t ** 2) * p2
            + (t ** 3) * p3
        )
        pts.append((float(pt[0]), float(pt[1])))
    return pts


def path_to_polylines(path_obj, curve_steps=24):
    polylines = []
    current_poly = []

    for seg in path_obj.segments:
        if seg.type == "line":
            if not current_poly:
                current_poly.append(seg.p0)
            current_poly.append(seg.p1)

        elif seg.type == "quadratic_bezier":
            curve_pts = sample_quadratic_bezier(seg.p0, seg.p1, seg.p2, steps=curve_steps)
            if not current_poly:
                current_poly.extend(curve_pts)
            else:
                current_poly.extend(curve_pts[1:] if current_poly[-1] == curve_pts[0] else curve_pts)

        elif seg.type == "cubic_bezier":
            curve_pts = sample_cubic_bezier(seg.p0, seg.p1, seg.p2, seg.p3, steps=curve_steps)
            if not current_poly:
                current_poly.extend(curve_pts)
            else:
                current_poly.extend(curve_pts[1:] if current_poly[-1] == curve_pts[0] else curve_pts)

    if len(current_poly) >= 2:
        polylines.append(current_poly)

    return polylines


def export_one_raw_d_to_dxf(raw_d, output_dxf, layer_name="PATH_01", curve_steps=32):
    paths = parse_path_d_multi(raw_d)

    if not paths:
        print("解析后没有 path，无法导出")
        return

    os.makedirs(os.path.dirname(output_dxf), exist_ok=True)

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name, dxfattribs={"color": 1})

    for path_obj in paths:
        polylines = path_to_polylines(path_obj, curve_steps=curve_steps)

        for poly in polylines:
            if len(poly) < 2:
                continue

            pts = [(float(x), float(-y)) for x, y in poly]
            if path_obj.closed and pts[0] != pts[-1]:
                pts.append(pts[0])

            msp.add_lwpolyline(pts, dxfattribs={"layer": layer_name})

    doc.saveas(output_dxf)
    print(f"已导出: {output_dxf}")


def export_text_idx(local_idx, output_dir="./debug_hard_samples/text_mis"):
    with open("./data/dataset_raw_text.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    item = data[local_idx]
    file_id = item["file_id"]
    path_index = item["path_index"]
    raw_d = item["raw_d"]

    out = os.path.join(output_dir, f"text_idx_{local_idx}_{file_id}_p{path_index}.dxf")
    print(f"[TEXT] idx={local_idx} | file_id={file_id} | path_index={path_index}")
    export_one_raw_d_to_dxf(raw_d, out, layer_name=f"{file_id}_P{path_index}"[:255])


def export_nontext_global_idx(global_idx, text_count=847, output_dir="./debug_hard_samples/nontext_mis"):
    with open("./data/dataset_raw_nontext.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    local_idx = global_idx - text_count
    item = data[local_idx]
    file_id = item["file_id"]
    path_index = item["path_index"]
    raw_d = item["raw_d"]

    out = os.path.join(output_dir, f"nontext_gidx_{global_idx}_lidx_{local_idx}_{file_id}_p{path_index}.dxf")
    print(f"[NON] gidx={global_idx} | local_idx={local_idx} | file_id={file_id} | path_index={path_index}")
    export_one_raw_d_to_dxf(raw_d, out, layer_name=f"{file_id}_P{path_index}"[:255])


if __name__ == "__main__":
    # hardest TEXT
    for idx in [435, 444, 529, 532, 533, 81, 552, 374]:
        export_text_idx(idx)

    # hardest NON-TEXT
    for gidx in [1501, 1308, 1164, 1181, 1499, 1034, 1660, 1661]:
        export_nontext_global_idx(gidx, text_count=847)