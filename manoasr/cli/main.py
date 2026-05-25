# coding=utf-8
"""mano-asr CLI 入口"""

import os
import warnings
import logging

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*model of type.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

import click

from manoasr import __version__
from manoasr.cli.commands import service, transcribe, port, model, config, logs, doctor


@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, help="显示版本号")
@click.pass_context
def cli(ctx, version):
    """mano-asr: 本地语音转写服务

    运行 mano-asr help 查看所有命令
    """
    if version:
        click.echo(f"mano-asr {__version__}")
        return

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(service.start)
cli.add_command(service.stop)
cli.add_command(service.restart)
cli.add_command(service.status)

cli.add_command(transcribe.transcribe)
cli.add_command(port.port)
cli.add_command(model.model)
cli.add_command(config.config)
cli.add_command(logs.logs)
cli.add_command(doctor.doctor)


@cli.command("help")
def help_cmd():
    """显示帮助信息"""
    help_text = """
  mano-asr - 本地语音转写服务

  使用方法:
    mano-asr <command> [options]

  服务管理:
    start         启动服务（首次运行自动初始化）
    stop          停止服务
    restart       重启服务
    status        查看服务状态

  功能:
    transcribe    转写音频文件
    port          端口管理（查看/设置）
    model         模型管理 (list/use/info)
    config        配置管理 (show/reset)
    logs          查看日志 (--errors/--stats)
    doctor        环境检查

  其他:
    help          显示此帮助信息
    --version     显示版本号

  示例:
    mano-asr start                    启动服务
    mano-asr transcribe audio.wav     转写音频
    mano-asr port 9000                设置端口
    mano-asr model use <name>         切换模型
    mano-asr logs --stats             查看日志统计
    mano-asr logs --errors            只看错误日志
"""
    click.echo(help_text)


def main():
    cli()


if __name__ == "__main__":
    main()
