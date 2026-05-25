"""
FSMN-VAD 顶层 Model: 组装 frontend + encoder + postprocess.

用法:
    from mlx_audio.vad.models.fsmn.model import FSMNVADModel
    model = FSMNVADModel.from_pretrained("/path/to/fsmn-vad-mlx")
    segments = model.detect("test.wav")
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig, FSMNEncoderConfig
from .encoder import FSMNEncoder
from .frontend import extract_features
from .postprocess import VADPostProcess, VADXOptions


class Model(nn.Module):
    """FSMN-VAD: 完整的 VAD pipeline."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = FSMNEncoder(config.encoder)

        # 后处理
        opts = VADXOptions(
            sample_rate=config.sample_rate,
            frame_in_ms=config.frame_in_ms,
            frame_length_ms=config.frame_length,
            window_size_ms=config.window_size_ms,
            sil_to_speech_time_thres=config.sil_to_speech_time_thres,
            speech_to_sil_time_thres=config.speech_to_sil_time_thres,
            speech_noise_thres=config.speech_noise_thres,
            max_end_silence_time=config.max_end_silence_time,
            max_start_silence_time=config.max_start_silence_time,
            sil_pdf_ids=config.sil_pdf_ids,
        )
        self.postprocess = VADPostProcess(opts)

        # 前端参数
        self._cmvn_path: Optional[str] = None
        self._model_dir: Optional[str] = None

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "Model":
        """从目录加载模型."""
        model_path = Path(model_path)

        # 读 config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)

        # 实例化
        model = cls(config)

        # 加载权重
        weights_path = model_path / "model.safetensors"
        weights = mx.load(str(weights_path))
        model.encoder.load_weights(list(weights.items()))

        # CMVN 路径
        cmvn_path = model_path / "am.mvn"
        if cmvn_path.exists():
            model._cmvn_path = str(cmvn_path)
        model._model_dir = str(model_path)

        return model

    def generate(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: int = 16000,
    ) -> List[List[int]]:
        """
        检测音频中的语音段.

        Args:
            audio: wav 文件路径或 float32 numpy waveform
            sample_rate: 音频采样率 (仅当 audio 为 numpy 时使用)

        Returns:
            [[start_ms, end_ms], ...] 语音段时间戳
        """
        # 加载音频
        if isinstance(audio, str):
            import soundfile as sf
            waveform, sr = sf.read(audio, dtype="float32")
            if sr != self.config.sample_rate:
                from scipy.signal import resample
                waveform = resample(
                    waveform, int(len(waveform) * self.config.sample_rate / sr)
                ).astype(np.float32)
        else:
            waveform = audio.astype(np.float32)

        # 前端特征提取
        features = extract_features(
            waveform,
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            frame_length_ms=self.config.frame_length,
            frame_shift_ms=self.config.frame_shift,
            lfr_m=self.config.lfr_m,
            lfr_n=self.config.lfr_n,
            cmvn_path=self._cmvn_path,
        )

        # MLX encoder forward
        x = mx.array(features[np.newaxis, :, :])  # [1, T, 400]
        scores = self.encoder(x)
        mx.eval(scores)
        scores_np = np.array(scores)  # [1, T, 248]

        # 后处理
        cache = self.postprocess.init_cache()
        segments = self.postprocess.forward(
            scores=scores_np,
            waveform=waveform,
            cache=cache,
            is_final=True,
        )

        return segments
