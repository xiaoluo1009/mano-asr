# coding=utf-8
"""mano-asr doctor - 环境检查"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import click

from manoasr.cli.utils.config import load_config, config_exists
from manoasr.cli.utils.console import success, error, warning, print_header, print_footer
from manoasr.cli.utils.process import is_port_in_use


def check_python() -> tuple[bool, str]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        return True, f"Python {version}"
    return False, f"Python {version} (需要 3.10+)"


def check_ffmpeg() -> tuple[bool, str]:
    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version_line = result.stdout.split("\n")[0] if result.stdout else ""
            if "version" in version_line.lower():
                parts = version_line.split()
                for i, p in enumerate(parts):
                    if p.lower() == "version" and i + 1 < len(parts):
                        return True, f"ffmpeg {parts[i + 1]}"
            return True, "ffmpeg (已安装)"
        except Exception:
            return True, "ffmpeg (已安装)"
    return False, "ffmpeg (未安装)"


def check_ffprobe() -> tuple[bool, str]:
    if shutil.which("ffprobe"):
        return True, "ffprobe (已安装)"
    return False, "ffprobe (未安装)"


def check_mlx() -> tuple[bool, str]:
    try:
        import mlx

        version = getattr(mlx, "__version__", "未知版本")
        return True, f"MLX {version}"
    except ImportError:
        return False, "MLX (未安装)"


def check_config() -> tuple[bool, str]:
    if config_exists():
        return True, "配置文件存在"
    return False, "配置文件不存在"


def check_model(model_path: str, model_type: str) -> tuple[bool, str]:
    path = Path(model_path)
    if path.exists() and (path / "config.json").exists():
        return True, f"{model_type} 模型: {path.name}"
    return False, f"{model_type} 模型: {path.name} (不存在)"


def check_port(port: int) -> tuple[bool, str]:
    if is_port_in_use(port):
        return False, f"端口 {port} 已被占用"
    return True, f"端口 {port} 可用"


@click.command()
def doctor():
    """环境检查"""

    print_header("环境检查")

    all_passed = True

    checks = [
        check_python(),
        check_ffmpeg(),
        check_ffprobe(),
        check_mlx(),
        check_config(),
    ]

    if config_exists():
        config = load_config()
        if config.get("models", {}).get("asr"):
            checks.append(check_model(config["models"]["asr"], "ASR"))
        if config.get("models", {}).get("vad"):
            checks.append(check_model(config["models"]["vad"], "VAD"))
        port = config.get("server", {}).get("port", 8787)
        checks.append(check_port(port))

    for passed, message in checks:
        if passed:
            click.echo(success(message))
        else:
            click.echo(error(message))
            all_passed = False

    print_footer()

    from manoasr.cli.utils.update_checker import check_and_notify
    check_and_notify()

    if all_passed:
        click.echo(success("所有检查通过\n"))
    else:
        click.echo(warning("部分检查未通过，请根据提示修复\n"))
        raise SystemExit(1)
