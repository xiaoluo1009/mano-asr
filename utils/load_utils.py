import os
import numpy as np

import subprocess
from subprocess import CalledProcessError, run


def is_ffmpeg_installed():
    try:
        output = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        return "ffmpeg version" in output.decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


use_ffmpeg = False
if is_ffmpeg_installed():
    use_ffmpeg = True
else:
    print(
        "Notice: ffmpeg is not installed. Audio file loading requires ffmpeg.\n"
        "Please install it by:"
        "\n\tsudo apt install ffmpeg # ubuntu"
        "\n\t# brew install ffmpeg # mac"
    )


def load_audio(
    data_or_path_or_list,
    fs: int = 16000,
    audio_fs: int = 16000,
    **kwargs,
):
    """Load audio input as a numpy waveform.

    Supported inputs are local paths, file-like objects, ``numpy.ndarray``
    waveforms, and lists/tuples of those audio inputs.
    """
    if isinstance(data_or_path_or_list, (list, tuple)):
        return [
            load_audio(audio, fs=fs, audio_fs=audio_fs, **kwargs)
            for audio in data_or_path_or_list
        ]

    if (
        isinstance(data_or_path_or_list, str) and os.path.exists(data_or_path_or_list)
    ) or hasattr(data_or_path_or_list, "read"):  # local audio file or bytes io
        if hasattr(data_or_path_or_list, "read") and hasattr(data_or_path_or_list, "seek"):
            data_or_path_or_list.seek(0)
        data_or_path_or_list = _load_audio_ffmpeg(data_or_path_or_list, sr=fs)
        audio_fs = fs
        # if data_in is a file or url, set is_final=True
        if "cache" in kwargs:
            kwargs["cache"]["is_final"] = True
            kwargs["cache"]["is_streaming_input"] = False
    elif isinstance(data_or_path_or_list, np.ndarray):  # audio sample point
        data_or_path_or_list = data_or_path_or_list.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported audio input: {data_or_path_or_list!r}")

    data_or_path_or_list = _to_mono_numpy(
        data_or_path_or_list,
        reduce_channels=kwargs.get("reduce_channels", True),
    )
    if audio_fs != fs:
        data_or_path_or_list = _resample_numpy(data_or_path_or_list, audio_fs, fs)
    return data_or_path_or_list


def _to_mono_numpy(audio: np.ndarray, reduce_channels: bool = True) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 0:
        raise ValueError("Audio input must contain at least one sample.")
    if audio.ndim == 1:
        return audio
    if not reduce_channels:
        return np.squeeze(audio)

    if audio.shape[0] <= audio.shape[-1]:
        audio = audio.mean(axis=0)
    else:
        audio = audio.mean(axis=-1)
    return np.asarray(audio, dtype=np.float32)


def _resample_numpy(audio: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    if source_fs <= 0 or target_fs <= 0:
        raise ValueError(
            f"Invalid sample rate: source_fs={source_fs}, target_fs={target_fs}"
        )
    if audio.size == 0 or source_fs == target_fs:
        return audio.astype(np.float32, copy=False)

    target_length = max(int(round(audio.shape[-1] * target_fs / source_fs)), 1)
    source_positions = np.linspace(0.0, 1.0, num=audio.shape[-1], endpoint=True)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _load_audio_ffmpeg(file, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    input_data = None
    input_target = file
    pcm_params = []
    if hasattr(file, "read"):
        input_data = file.read()
        input_target = "pipe:0"
    elif str(file).lower().endswith('.pcm'):
        pcm_params = [
            "-f", "s16le",
            "-ar", str(sr),
            "-ac", "1"
        ]

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        *pcm_params,
        "-i", input_target,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, input=input_data, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    except FileNotFoundError as e:
        raise RuntimeError("Failed to load audio: ffmpeg is not installed.") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
