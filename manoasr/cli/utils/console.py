# coding=utf-8
"""控制台输出工具"""

from __future__ import annotations


def success(msg: str) -> str:
    return f"  ✓ {msg}"


def error(msg: str) -> str:
    return f"  ✗ {msg}"


def warning(msg: str) -> str:
    return f"  ! {msg}"


def info(msg: str) -> str:
    return f"  → {msg}"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def divider(width: int = 35) -> str:
    return "─" * width


def key_value(key: str, value: str, width: int = 10) -> str:
    return f"  {key}:{' ' * (width - len(key))}{value}"


def print_header(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {divider()}")


def print_footer() -> None:
    print(f"  {divider()}\n")


def interactive_select(title: str, options: list[dict], current: str | None = None) -> dict | None:
    """Arrow-key driven interactive selector.

    Each option is a dict with at least a ``key`` field.  Optional fields:
    ``label`` (display text, defaults to key) and ``hint``.

    Returns the chosen option dict, or None if the user pressed Ctrl-C / q.
    """
    import sys
    import tty
    import termios

    if not options:
        return None

    selected = 0
    for i, opt in enumerate(options):
        if opt["key"] == current:
            selected = i
            break

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    total_lines = len(options) + 5

    def _read_key() -> str:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("\x03", "\x04"):
            return "ctrl-c"
        return ch

    def _save_cursor() -> None:
        sys.stdout.write("\0337")
        sys.stdout.flush()

    def _restore_and_clear() -> None:
        sys.stdout.write("\0338")
        sys.stdout.write(f"\033[J")
        sys.stdout.flush()

    def _render(sel: int) -> None:
        lines = []
        lines.append(f"\r\n  {title}")
        lines.append(f"\r\n  {divider()}")
        for i, opt in enumerate(options):
            label = opt.get("label", opt["key"])
            marker = "›" if i == sel else " "
            tag = " (当前)" if opt["key"] == current else ""
            highlight = "\033[1m" if i == sel else ""
            reset = "\033[0m" if i == sel else ""
            lines.append(f"\r\n  {marker} {highlight}{label}{tag}{reset}")
        lines.append(f"\r\n  {divider()}")
        lines.append(f"\r\n  ↑↓ 选择  Enter 确认  q 取消")
        sys.stdout.write("".join(lines))
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)

        _save_cursor()
        _render(selected)

        while True:
            key = _read_key()
            if key == "up":
                selected = (selected - 1) % len(options)
            elif key == "down":
                selected = (selected + 1) % len(options)
            elif key == "enter":
                _restore_and_clear()
                return options[selected]
            elif key in ("ctrl-c", "esc", "q"):
                _restore_and_clear()
                return None

            _restore_and_clear()
            _render(selected)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
