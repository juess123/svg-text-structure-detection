import re

SUPPORTED_COMMANDS = set("MmLlHhVvCcZz")

def parse_commands(raw_d):

    pattern = re.compile(r'([A-Za-z])|(-?\d+\.?\d*)')

    tokens = pattern.findall(raw_d)

    commands = []
    current_cmd = None
    current_numbers = []

    for token in tokens:
        letter, number = token

        # 如果是命令字母
        if letter:
            if current_cmd is not None:
                commands.append((current_cmd, current_numbers))
                current_numbers = []

            current_cmd = letter

            if letter not in SUPPORTED_COMMANDS:
                print(f"⚠ Warning: Unsupported SVG command '{letter}' detected.")

        # 如果是数字
        elif number:
            current_numbers.append(float(number))

    # 最后一条命令
    if current_cmd is not None:
        commands.append((current_cmd, current_numbers))

    return commands

def command_stats(commands):
    """
    统计命令层面的结构信息
    """

    num_commands = len(commands)

    # 曲线命令（目前只支持 C/c）
    num_curve_commands = sum(
        1 for cmd, _ in commands if cmd.lower() == 'c'
    )

    # 子路径数量（每个 M/m 表示一个新的子路径）
    num_subpaths = sum(
        1 for cmd, _ in commands if cmd.lower() == 'm'
    )

    return {
        "num_commands": num_commands,
        "num_curve_commands": num_curve_commands,
        "num_subpaths": num_subpaths
    }

