import numpy as np


def cubic_bezier(p0, p1, p2, p3, t):
    return (
        (1 - t) ** 3 * p0 +
        3 * (1 - t) ** 2 * t * p1 +
        3 * (1 - t) * t ** 2 * p2 +
        t ** 3 * p3
    )


def sample_path(commands, curve_steps=20):
    """
    把命令序列转为点序列
    """

    points = []
    current = np.array([0.0, 0.0])
    start_point = None

    for cmd, values in commands:

        cmd_lower = cmd.lower()

        # Move
        if cmd_lower == 'm':
            current = np.array(values[:2])
            start_point = current.copy()
            points.append(current.copy())

        # Line
        elif cmd_lower == 'l':
            target = np.array(values[:2])
            points.append(target)
            current = target

        # Horizontal line
        elif cmd_lower == 'h':
            target = np.array([current[0] + values[0], current[1]])
            points.append(target)
            current = target

        # Vertical line
        elif cmd_lower == 'v':
            target = np.array([current[0], current[1] + values[0]])
            points.append(target)
            current = target

        # Cubic Bezier
        elif cmd_lower == 'c':
            p0 = current
            p1 = current + np.array(values[0:2])
            p2 = current + np.array(values[2:4])
            p3 = current + np.array(values[4:6])

            for t in np.linspace(0, 1, curve_steps):
                pt = cubic_bezier(p0, p1, p2, p3, t)
                points.append(pt)

            current = p3

        # Close path
        elif cmd_lower == 'z':
            if start_point is not None:
                points.append(start_point.copy())
                current = start_point

    return np.array(points)