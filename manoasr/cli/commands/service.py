# coding=utf-8
"""mano-asr 服务管理命令: start/stop/restart/status"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from manoasr.cli.utils.config import load_config, config_exists, save_config, get_default_config
from manoasr.cli.utils.console import (
    success,
    error,
    warning,
    info,
    bold,
    key_value,
    print_header,
    print_footer,
)
from manoasr.cli.utils.process import (
    get_pid,
    save_pid,
    stop_process,
    is_port_in_use,
    get_port_process,
    get_process_uptime,
)
from manoasr.cli.utils.constants import DEFAULT_PORT, LOG_FILE, CONFIG_DIR, LOG_DIR, MODEL_TYPES, DEFAULT_MODEL_TYPE
from manoasr.cli.utils.download import ensure_default_models, ensure_model, find_model_in_dirs


def get_configured_port() -> int:
    if config_exists():
        config = load_config()
        return config.get("server", {}).get("port", DEFAULT_PORT)
    return DEFAULT_PORT


def _do_init() -> dict:
    """执行初始化：下载模型（如需）并创建配置"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    asr_path, vad_path = ensure_default_models(DEFAULT_MODEL_TYPE)

    config = get_default_config()
    config["models"]["asr"] = str(asr_path)
    if vad_path:
        config["models"]["vad"] = str(vad_path)
    else:
        config["models"]["vad"] = None
    save_config(config)

    asr_name = Path(config["models"]["asr"]).name
    click.echo(success(f"自动初始化完成"))
    click.echo(f"    ASR 模型: {asr_name}")
    if config["models"].get("vad"):
        vad_name = Path(config["models"]["vad"]).name
        click.echo(f"    VAD 模型: {vad_name}")
    click.echo(f"    服务端口: {config['server']['port']}")

    return config


def _check_service_health(port: int, timeout: float = 2.0) -> tuple[bool, str]:
    """检查服务健康状态，返回 (是否健康, 状态描述)"""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            pass
        return True, "正常"
    except socket.timeout:
        return False, "响应超时"
    except ConnectionRefusedError:
        return False, "连接被拒绝"
    except Exception as e:
        return False, str(e)


@click.command()
@click.option("--foreground", "-f", is_flag=True, help="前台运行（调试用）")
@click.option("--debug", "-d", is_flag=True, help="调试模式（记录转写结果到日志）")
def start(foreground: bool, debug: bool):
    """启动 mano-asr 服务（首次运行自动初始化）"""

    if not config_exists():
        click.echo(info("首次运行，正在初始化..."))
        _do_init()
        click.echo("")

    config = load_config()
    port = config.get("server", {}).get("port", DEFAULT_PORT)

    asr_path = Path(config.get("models", {}).get("asr", ""))
    if not asr_path.exists() or not (asr_path / "config.json").exists():
        model_type_key = config.get("models", {}).get("type", DEFAULT_MODEL_TYPE)
        spec = MODEL_TYPES.get(model_type_key, {})
        model_name = spec.get("default_model", asr_path.name)
        click.echo(info("ASR 模型不存在，正在下载..."))
        asr_path = ensure_model(model_name, is_vad=False)
        config["models"]["asr"] = str(asr_path)
        save_config(config)

    vad_path_str = config.get("models", {}).get("vad")
    if vad_path_str:
        vad_path = Path(vad_path_str)
        if not vad_path.exists():
            click.echo(info("VAD 模型不存在，正在下载..."))
            try:
                from manoasr.cli.utils.constants import DEFAULT_VAD_MODEL
                vad_path = ensure_model(DEFAULT_VAD_MODEL, is_vad=True)
                config["models"]["vad"] = str(vad_path)
                save_config(config)
            except SystemExit:
                click.echo(warning("VAD 模型不可用，继续启动（不使用 VAD）"))
                config["models"]["vad"] = None
                save_config(config)

    pid = get_pid()
    if pid:
        healthy, health_msg = _check_service_health(port)
        if healthy:
            click.echo(warning(f"mano-asr 服务已在运行中 (PID: {pid})"))
            return
        else:
            port_info = get_port_process(port)
            if port_info and port_info[0] != pid:
                click.echo(warning(f"服务进程存在 (PID: {pid}) 但端口被占用"))
                click.echo(error(f"端口 {port} 被其他进程占用: {port_info[1]} (PID: {port_info[0]})"))
                click.echo(f"\n    解决方法:")
                click.echo(f"      1. 停止占用进程: kill {port_info[0]}")
                click.echo(f"      2. 然后重启: mano-asr restart")
                raise SystemExit(1)
            else:
                click.echo(warning(f"服务进程存在 (PID: {pid}) 但无响应，尝试重启..."))
                stop_process(pid)

    if is_port_in_use(port):
        process_info = get_port_process(port)
        click.echo(error(f"端口 {port} 已被占用"))
        if process_info:
            pid, name = process_info
            click.echo(f"    占用进程: {name} (PID: {pid})")
        click.echo(f"\n    解决方法:")
        click.echo(f"      1. 停止占用进程: kill {process_info[0] if process_info else '<PID>'}")
        click.echo(f"      2. 或更换端口: mano-asr port <新端口>")
        raise SystemExit(1)

    if foreground:
        _run_server(config, port, debug)
    else:
        _start_daemon(config, port, debug)

    from manoasr.cli.utils.update_checker import check_and_notify
    check_and_notify()


def _run_server(config: dict, port: int, debug: bool = False):
    """前台运行服务"""
    model_type_key = config.get("models", {}).get("type", DEFAULT_MODEL_TYPE)
    spec = MODEL_TYPES.get(model_type_key, {})
    asr_name = Path(config["models"]["asr"]).name

    click.echo(success("mano-asr 服务已启动（前台模式）"))
    click.echo(f"    地址: http://127.0.0.1:{port}")
    click.echo(f"    引擎: {model_type_key} ({spec.get('label', model_type_key)})")
    click.echo(f"    模型: {asr_name}")
    if debug:
        click.echo(f"    调试模式: 开启")
    click.echo(f"    按 Ctrl+C 停止\n")

    import uvicorn

    from manoasr.cli.utils.constants import PROJECT_ROOT

    sys.path.insert(0, str(PROJECT_ROOT))

    import server

    server.MODEL_PATH = config["models"]["asr"]
    server.VAD_MODEL_PATH = config["models"].get("vad")
    server.HOST = "0.0.0.0"
    server.PORT = port
    server.LOAD_ON_STARTUP = config.get("server", {}).get("load_on_startup", True)
    server.DEBUG_MODE = debug

    model_type_key = config.get("models", {}).get("type", DEFAULT_MODEL_TYPE)
    spec = MODEL_TYPES.get(model_type_key)
    if spec:
        server.MODEL_TYPE = spec["server_type"]

    uvicorn.run(server.app, host="0.0.0.0", port=port, log_level="info")


def _start_daemon(config: dict, port: int, debug: bool = False):
    """后台启动服务"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    daemon_path = Path(__file__).parent.parent / "daemon.py"

    cmd = [
        sys.executable,
        str(daemon_path),
        "--model",
        config["models"]["asr"],
        "--port",
        str(port),
    ]
    if config["models"].get("vad"):
        cmd.extend(["--vad", config["models"]["vad"]])

    model_type_key = config.get("models", {}).get("type", DEFAULT_MODEL_TYPE)
    spec = MODEL_TYPES.get(model_type_key)
    if spec:
        cmd.extend(["--model-type", spec["server_type"]])

    if config.get("server", {}).get("load_on_startup", True):
        cmd.append("--load-on-startup")
    if debug:
        cmd.append("--debug")

    with open(LOG_FILE, "a") as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )

    save_pid(process.pid)

    model_type_key = config.get("models", {}).get("type", DEFAULT_MODEL_TYPE)
    spec = MODEL_TYPES.get(model_type_key, {})
    asr_name = Path(config["models"]["asr"]).name

    click.echo(success("mano-asr 服务已启动"))
    click.echo(f"    地址: http://127.0.0.1:{port}")
    click.echo(f"    PID:  {process.pid}")
    click.echo(f"    引擎: {model_type_key} ({spec.get('label', model_type_key)})")
    click.echo(f"    模型: {asr_name}")
    if debug:
        click.echo(f"    调试模式: 开启")


@click.command()
def stop():
    """停止 mano-asr 服务"""

    pid = get_pid()
    if not pid:
        click.echo(warning("mano-asr 服务未在运行"))
        return

    click.echo(info("正在停止服务..."))

    if stop_process(pid):
        click.echo(success("mano-asr 服务已停止"))
    else:
        click.echo(error(f"无法停止进程 {pid}，请手动终止: kill -9 {pid}"))
        raise SystemExit(1)


@click.command()
@click.option("--debug", "-d", is_flag=True, help="调试模式（记录转写结果到日志）")
@click.pass_context
def restart(ctx, debug: bool):
    """重启 mano-asr 服务"""

    pid = get_pid()
    if pid:
        click.echo(info("停止服务..."))
        if not stop_process(pid):
            click.echo(error("无法停止服务"))
            raise SystemExit(1)
        click.echo(success("服务已停止"))

    click.echo(info("启动服务..."))
    ctx.invoke(start, foreground=False, debug=debug)


@click.command()
def status():
    """查看 mano-asr 服务状态"""

    print_header("mano-asr 服务状态")

    pid = get_pid()
    port = get_configured_port()

    if pid:
        uptime = get_process_uptime(pid) or "未知"
        healthy, health_msg = _check_service_health(port)

        if healthy:
            click.echo(key_value("状态", bold("运行中")))
        else:
            click.echo(key_value("状态", warning(f"异常 ({health_msg})")))
            port_info = get_port_process(port)
            if port_info and port_info[0] != pid:
                click.echo(warning(f"  ⚠ 端口 {port} 被其他进程占用: {port_info[1]} (PID: {port_info[0]})"))
                click.echo(info(f"    建议: mano-asr restart 或 kill {port_info[0]}"))

        click.echo(key_value("PID", str(pid)))
        click.echo(key_value("端口", str(port)))
        click.echo(key_value("运行时间", uptime))

        if config_exists():
            config = load_config()
            model_type_key = config.get("models", {}).get("type", DEFAULT_MODEL_TYPE)
            spec = MODEL_TYPES.get(model_type_key, {})
            click.echo(f"  {'─' * 35}")
            click.echo(key_value("引擎", f"{model_type_key} ({spec.get('label', model_type_key)})"))
            click.echo(key_value("ASR 模型", Path(config["models"]["asr"]).name))
            if config["models"].get("vad"):
                click.echo(key_value("VAD 模型", Path(config["models"]["vad"]).name))
    else:
        click.echo(key_value("状态", "未运行"))

    print_footer()

    from manoasr.cli.utils.update_checker import check_and_notify
    check_and_notify()
