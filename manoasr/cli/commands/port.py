# coding=utf-8
"""mano-asr port - 端口管理"""

from __future__ import annotations

import click

from manoasr.cli.utils.config import load_config, save_config, config_exists
from manoasr.cli.utils.console import success, error, warning, info
from manoasr.cli.utils.constants import DEFAULT_PORT


@click.command()
@click.argument("new_port", type=int, required=False)
def port(new_port: int):
    """端口管理

    \b
    查看当前端口:
      mano-asr port

    \b
    设置新端口:
      mano-asr port 9000
    """
    if not config_exists():
        click.echo(error("未初始化，请先运行: mano-asr init"))
        raise SystemExit(1)

    config = load_config()
    current_port = config.get("server", {}).get("port", DEFAULT_PORT)

    if new_port is None:
        click.echo(f"\n  当前服务端口: {current_port}\n")
        return

    if new_port < 1024 or new_port > 65535:
        click.echo(error("端口号必须在 1024-65535 之间"))
        raise SystemExit(1)

    if new_port == current_port:
        click.echo(info(f"端口未变化，仍为: {current_port}"))
        return

    if "server" not in config:
        config["server"] = {}
    config["server"]["port"] = new_port
    save_config(config)

    click.echo(success(f"端口已更改: {current_port} → {new_port}"))
    click.echo(warning("需要重启服务生效: mano-asr restart"))
