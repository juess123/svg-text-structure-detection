import re
import numpy as np


def parse_commands(d: str):
    """
    把 SVG path 的 d 字符串解析成命令序列:
    返回格式:
        [
            ("M", [x, y]),
            ("L", [x, y]),
            ("C", [x1, y1, x2, y2, x, y]),
            ...
        ]

    当前支持:
        M/m, L/l, H/h, V/v, C/c, Q/q, Z/z
    """
    if not d or not isinstance(d, str):
        return []

    tokens = re.findall(
        r"[MLCQZHVmlcqzhv]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        d
    )

    commands = []
    i = 0
    current_cmd = None

    param_count = {
        "M": 2, "m": 2,
        "L": 2, "l": 2,
        "H": 1, "h": 1,
        "V": 1, "v": 1,
        "Q": 4, "q": 4,
        "C": 6, "c": 6,
        "Z": 0, "z": 0,
    }

    while i < len(tokens):
        token = tokens[i]

        if token in param_count:
            current_cmd = token
            i += 1

            if current_cmd in ("Z", "z"):
                commands.append((current_cmd, []))
                continue
        else:
            if current_cmd is None:
                i += 1
                continue

        need = param_count[current_cmd]

        if need == 0:
            continue

        while i + need - 1 < len(tokens):
            # 后面如果遇到新命令，就停止当前批次
            if tokens[i] in param_count:
                break

            vals = tokens[i:i + need]
            if len(vals) < need:
                break

            try:
                vals = [float(v) for v in vals]
            except ValueError:
                break

            commands.append((current_cmd, vals))
            i += need

            # M/m 后面连续坐标，按 L/l 处理
            if current_cmd == "M":
                current_cmd = "L"
                need = 2
            elif current_cmd == "m":
                current_cmd = "l"
                need = 2

            # 如果下一个 token 是新命令，结束
            if i < len(tokens) and tokens[i] in param_count:
                break

    return commands


def command_stats(commands):
    """
    统计命令信息，供特征提取使用
    """
    stats = {
        "num_commands": 0,
        "num_subpaths": 0,
        "num_curve_commands": 0,
        "num_line_commands": 0,
        "num_close_commands": 0,
        "num_move_commands": 0,
    }

    if not commands:
        return stats

    curve_cmds = {"C", "c", "Q", "q"}
    line_cmds = {"L", "l", "H", "h", "V", "v"}

    for cmd, _ in commands:
        stats["num_commands"] += 1

        if cmd in ("M", "m"):
            stats["num_subpaths"] += 1
            stats["num_move_commands"] += 1
        elif cmd in curve_cmds:
            stats["num_curve_commands"] += 1
        elif cmd in line_cmds:
            stats["num_line_commands"] += 1
        elif cmd in ("Z", "z"):
            stats["num_close_commands"] += 1

    return stats


def _quadratic_bezier(p0, p1, p2, t):
    return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2


def _cubic_bezier(p0, p1, p2, p3, t):
    return (
        ((1 - t) ** 3) * p0
        + 3 * ((1 - t) ** 2) * t * p1
        + 3 * (1 - t) * (t ** 2) * p2
        + (t ** 3) * p3
    )


def sample_path(commands, curve_steps=24):
    """
    把命令序列采样成点序列，供几何特征提取使用。
    返回:
        np.ndarray shape=(N, 2)

    当前支持:
        M/m, L/l, H/h, V/v, Q/q, C/c, Z/z
    """
    points = []
    current = np.array([0.0, 0.0], dtype=np.float64)
    start_point = None

    for cmd, values in commands:
        # MoveTo
        if cmd == "M":
            current = np.array(values[:2], dtype=np.float64)
            start_point = current.copy()
            points.append(current.copy())

        elif cmd == "m":
            current = current + np.array(values[:2], dtype=np.float64)
            start_point = current.copy()
            points.append(current.copy())

        # LineTo
        elif cmd == "L":
            target = np.array(values[:2], dtype=np.float64)
            points.append(target.copy())
            current = target

        elif cmd == "l":
            target = current + np.array(values[:2], dtype=np.float64)
            points.append(target.copy())
            current = target

        # Horizontal
        elif cmd == "H":
            target = np.array([values[0], current[1]], dtype=np.float64)
            points.append(target.copy())
            current = target

        elif cmd == "h":
            target = np.array([current[0] + values[0], current[1]], dtype=np.float64)
            points.append(target.copy())
            current = target

        # Vertical
        elif cmd == "V":
            target = np.array([current[0], values[0]], dtype=np.float64)
            points.append(target.copy())
            current = target

        elif cmd == "v":
            target = np.array([current[0], current[1] + values[0]], dtype=np.float64)
            points.append(target.copy())
            current = target

        # Quadratic Bezier
        elif cmd == "Q":
            p0 = current.copy()
            p1 = np.array(values[0:2], dtype=np.float64)
            p2 = np.array(values[2:4], dtype=np.float64)

            for t in np.linspace(0, 1, curve_steps + 1)[1:]:
                pt = _quadratic_bezier(p0, p1, p2, t)
                points.append(pt.copy())

            current = p2

        elif cmd == "q":
            p0 = current.copy()
            p1 = current + np.array(values[0:2], dtype=np.float64)
            p2 = current + np.array(values[2:4], dtype=np.float64)

            for t in np.linspace(0, 1, curve_steps + 1)[1:]:
                pt = _quadratic_bezier(p0, p1, p2, t)
                points.append(pt.copy())

            current = p2

        # Cubic Bezier
        elif cmd == "C":
            p0 = current.copy()
            p1 = np.array(values[0:2], dtype=np.float64)
            p2 = np.array(values[2:4], dtype=np.float64)
            p3 = np.array(values[4:6], dtype=np.float64)

            for t in np.linspace(0, 1, curve_steps + 1)[1:]:
                pt = _cubic_bezier(p0, p1, p2, p3, t)
                points.append(pt.copy())

            current = p3

        elif cmd == "c":
            p0 = current.copy()
            p1 = current + np.array(values[0:2], dtype=np.float64)
            p2 = current + np.array(values[2:4], dtype=np.float64)
            p3 = current + np.array(values[4:6], dtype=np.float64)

            for t in np.linspace(0, 1, curve_steps + 1)[1:]:
                pt = _cubic_bezier(p0, p1, p2, p3, t)
                points.append(pt.copy())

            current = p3

        # ClosePath
        elif cmd in ("Z", "z"):
            if start_point is not None:
                points.append(start_point.copy())
                current = start_point.copy()

    if not points:
        return np.zeros((0, 2), dtype=np.float64)

    return np.array(points, dtype=np.float64)