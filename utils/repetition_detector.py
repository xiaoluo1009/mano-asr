# coding=utf-8
import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9]+(?:[-_'][a-zA-Z0-9]+)*|[\u4e00-\u9fff]|[^\s\w]",
    re.UNICODE,
)
_PUNCT_RE = re.compile(r"^[^\w\u4e00-\u9fff]+$", re.UNICODE)


@dataclass
class RepetitionConfig:
    max_ngram_size: int = 20
    min_repeat_count: int = 5
    min_repeated_tokens: int = 20
    min_repeat_ratio: float = 0.3
    high_repeat_count: int = 10
    high_repeat_ratio: float = 0.5
    max_text_chars_per_second: float = 10.0


@dataclass
class RepetitionSpan:
    repeat_text: str
    repeat_count: int
    ngram_size: int
    start_token: int
    end_token: int
    repeated_tokens: int
    repeat_ratio: float


@dataclass
class RepetitionResult:
    has_repeat: bool
    risk_level: str
    confidence: float
    reason: str
    token_count: int
    text_char_count: int
    chars_per_second: Optional[float]
    best_repeat: Optional[RepetitionSpan]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        if self.best_repeat is None:
            data["best_repeat"] = None
        return data


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_asr_text(text: str) -> List[str]:
    tokens = []
    for token in _TOKEN_RE.findall(normalize_text(text)):
        if _PUNCT_RE.match(token):
            continue
        tokens.append(token)
    return tokens


def text_char_count(text: str) -> int:
    return len("".join(tokenize_asr_text(text)))


def join_tokens(tokens: Sequence[str]) -> str:
    parts = []
    for token in tokens:
        if re.fullmatch(r"[a-zA-Z0-9]+(?:[-_'][a-zA-Z0-9]+)*", token):
            if parts:
                parts.append(" ")
            parts.append(token)
        else:
            parts.append(token)
    return "".join(parts)


def _iter_repeated_spans(
    tokens: Sequence[str],
    max_ngram_size: int,
) -> Iterable[Tuple[int, int, int, int]]:
    token_count = len(tokens)
    max_ngram_size = min(max_ngram_size, token_count // 2)

    for ngram_size in range(1, max_ngram_size + 1):
        index = 0
        while index + ngram_size * 2 <= token_count:
            pattern = tuple(tokens[index : index + ngram_size])
            repeat_count = 1
            cursor = index + ngram_size

            while cursor + ngram_size <= token_count:
                current = tuple(tokens[cursor : cursor + ngram_size])
                if current != pattern:
                    break
                repeat_count += 1
                cursor += ngram_size

            if repeat_count > 1:
                yield index, cursor, ngram_size, repeat_count
                index = cursor
            else:
                index += 1


def find_best_repetition(
    text: str,
    config: Optional[RepetitionConfig] = None,
) -> Optional[RepetitionSpan]:
    cfg = config or RepetitionConfig()
    tokens = tokenize_asr_text(text)
    token_count = len(tokens)
    if token_count < 2:
        return None

    best: Optional[RepetitionSpan] = None
    for start, end, ngram_size, repeat_count in _iter_repeated_spans(
        tokens,
        cfg.max_ngram_size,
    ):
        repeated_tokens = end - start
        repeat_ratio = repeated_tokens / token_count
        span = RepetitionSpan(
            repeat_text=join_tokens(tokens[start : start + ngram_size]),
            repeat_count=repeat_count,
            ngram_size=ngram_size,
            start_token=start,
            end_token=end,
            repeated_tokens=repeated_tokens,
            repeat_ratio=repeat_ratio,
        )

        if best is None:
            best = span
            continue

        best_score = best.repeat_ratio * best.repeated_tokens
        span_score = span.repeat_ratio * span.repeated_tokens
        if span_score > best_score:
            best = span

    return best


def detect_repetition(
    text: str,
    audio_duration: Optional[float] = None,
    config: Optional[RepetitionConfig] = None,
) -> RepetitionResult:
    cfg = config or RepetitionConfig()
    tokens = tokenize_asr_text(text)
    char_count = text_char_count(text)
    best = find_best_repetition(text, cfg)

    chars_per_second = None
    too_fast = False
    if audio_duration is not None and audio_duration > 0:
        chars_per_second = char_count / audio_duration
        too_fast = chars_per_second > cfg.max_text_chars_per_second

    if best is None:
        return RepetitionResult(
            has_repeat=False,
            risk_level="none",
            confidence=0.0,
            reason="未发现连续重复片段",
            token_count=len(tokens),
            text_char_count=char_count,
            chars_per_second=chars_per_second,
            best_repeat=None,
        )

    meets_base_rule = (
        best.repeat_count >= cfg.min_repeat_count
        and best.repeated_tokens >= cfg.min_repeated_tokens
        and best.repeat_ratio >= cfg.min_repeat_ratio
    )
    meets_high_rule = (
        best.repeat_count >= cfg.high_repeat_count
        and best.repeat_ratio >= cfg.high_repeat_ratio
    )

    if meets_high_rule or (meets_base_rule and too_fast):
        risk_level = "high"
    elif meets_base_rule:
        risk_level = "medium"
    elif best.repeat_count >= 3:
        risk_level = "low"
    else:
        risk_level = "none"

    has_repeat = risk_level in {"medium", "high"}
    confidence = _estimate_confidence(best, too_fast, cfg) if has_repeat else 0.0
    reason = _build_reason(best, too_fast, risk_level)

    return RepetitionResult(
        has_repeat=has_repeat,
        risk_level=risk_level,
        confidence=confidence,
        reason=reason,
        token_count=len(tokens),
        text_char_count=char_count,
        chars_per_second=chars_per_second,
        best_repeat=best,
    )


def check_repetition(
    text: str,
    audio_duration: Optional[float] = None,
    *,
    min_repeat_count: int = 5,
    min_repeat_ratio: float = 0.3,
    min_repeated_tokens: int = 20,
    max_ngram_size: int = 20,
) -> Dict[str, Any]:
    """Return a plain dict for simple ASR repetition checks.

    Example:
        result = check_repetition("hello hello hello ...")
        if result["has_repeat"]:
            print(result["repeat_text"])
    """
    config = RepetitionConfig(
        max_ngram_size=max_ngram_size,
        min_repeat_count=min_repeat_count,
        min_repeated_tokens=min_repeated_tokens,
        min_repeat_ratio=min_repeat_ratio,
    )
    result = detect_repetition(text, audio_duration=audio_duration, config=config)
    best = result.best_repeat

    return {
        "has_repeat": result.has_repeat,
        "risk_level": result.risk_level,
        "confidence": result.confidence,
        "reason": result.reason,
        "repeat_text": best.repeat_text if best else "",
        "repeat_count": best.repeat_count if best else 0,
        "repeat_ratio": best.repeat_ratio if best else 0.0,
        "repeated_tokens": best.repeated_tokens if best else 0,
        "token_count": result.token_count,
        "text_char_count": result.text_char_count,
        "chars_per_second": result.chars_per_second,
        "detail": result.to_dict(),
    }


def has_repetition(
    text: str,
    audio_duration: Optional[float] = None,
    **kwargs: Any,
) -> bool:
    """Return only whether the ASR text contains abnormal repetition."""
    return bool(check_repetition(text, audio_duration=audio_duration, **kwargs)["has_repeat"])


def _estimate_confidence(
    repeat: RepetitionSpan,
    too_fast: bool,
    cfg: RepetitionConfig,
) -> float:
    count_score = min(repeat.repeat_count / max(cfg.high_repeat_count, 1), 1.0)
    ratio_score = min(repeat.repeat_ratio / max(cfg.high_repeat_ratio, 0.01), 1.0)
    confidence = 0.45 + 0.25 * count_score + 0.25 * ratio_score
    if too_fast:
        confidence += 0.05
    return round(min(confidence, 0.99), 4)


def _build_reason(
    repeat: RepetitionSpan,
    too_fast: bool,
    risk_level: str,
) -> str:
    if risk_level == "none":
        return "存在少量连续重复，但未达到异常重复阈值"

    reason = (
        f"片段“{repeat.repeat_text}”连续重复 {repeat.repeat_count} 次，"
        f"重复区域占比 {repeat.repeat_ratio:.2%}"
    )
    if too_fast:
        reason += "，且文本密度超过音频时长可解释范围"
    return reason


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect repeated degeneration in ASR transcription text."
    )
    parser.add_argument("--text", required=True, help="ASR transcription text.")
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=None,
        help="Optional audio duration in seconds.",
    )
    parser.add_argument("--max-ngram-size", type=int, default=20)
    parser.add_argument("--min-repeat-count", type=int, default=5)
    parser.add_argument("--min-repeated-tokens", type=int, default=20)
    parser.add_argument("--min-repeat-ratio", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = check_repetition(
        args.text,
        audio_duration=args.audio_duration,
        max_ngram_size=args.max_ngram_size,
        min_repeat_count=args.min_repeat_count,
        min_repeated_tokens=args.min_repeated_tokens,
        min_repeat_ratio=args.min_repeat_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
