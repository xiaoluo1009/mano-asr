# coding=utf-8
"""mano-asr config - 配置管理"""

from __future__ import annotations

import click

from manoasr.cli.utils.config import (
    load_config,
    save_config,
    config_exists,
    get_default_config,
)
from manoasr.cli.utils.console import success, error, print_header, print_footer
from manoasr.cli.utils.constants import CONFIG_FILE

import yaml


@click.group()
def config():
    """配置管理"""
    pass


@config.command("show")
def config_show():
    """显示当前配置"""

    if not config_exists():
        click.echo(error("未初始化，请先运行: mano-asr start"))
        raise SystemExit(1)

    current_config = load_config()

    click.echo(f"\n  配置文件: {CONFIG_FILE}")
    print_header("配置内容")

    yaml_str = yaml.dump(current_config, allow_unicode=True, default_flow_style=False)
    for line in yaml_str.strip().split("\n"):
        click.echo(f"  {line}")

    print_footer()


@config.command("reset")
@click.option("--yes", "-y", is_flag=True, help="跳过确认")
def config_reset(yes: bool):
    """重置为默认配置"""

    if not yes:
        click.confirm("确定要重置配置吗？", abort=True)

    default_config = get_default_config()
    save_config(default_config)

    click.echo(success("配置已重置为默认值"))
