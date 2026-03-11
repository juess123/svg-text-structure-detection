def compute_path_size(path):

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return max(width, height)


def normalize_size(size, min_size, max_size):

    if max_size == min_size:
        return 5

    return (size - min_size) / (max_size - min_size) * 10


def classify_size(norm_size):

    if norm_size < 3.3:
        return "SMALL"

    elif norm_size < 6.6:
        return "MEDIUM"

    else:
        return "LARGE"