from .config import FSMNEncoderConfig, ModelConfig
from .encoder import FSMNBlock, FSMNEncoder, FSMNLayer

DETECTION_HINTS = {
    "architectures": ["fsmn_vad", "FsmnVAD", "FsmnVADStreaming"],
    "config_keys": ["encoder", "fsmn_layers", "lorder"],
    "path_patterns": ["fsmn", "fsmn-vad", "fsmn_vad"],
}

__all__ = [
    "FSMNEncoderConfig",
    "ModelConfig",
    "FSMNBlock",
    "FSMNEncoder",
    "FSMNLayer",
    "DETECTION_HINTS",
]
