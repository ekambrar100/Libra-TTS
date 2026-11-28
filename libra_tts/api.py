"""LibraTTS: text in, 24 kHz speech out."""
from __future__ import annotations

import platform
import sys
from typing import List, Optional

import numpy as np

from .config import Config, Sampling, model_dir
from .frontend import FrontEnd, language_runs
from .prompt import Tokenizer, build_prompt


def end_run_on(audio: np.ndarray, estimated_frames: float, sample_rate: int = 24000, frame_rate: float = 12.5,
               silence_db: float = -50.0, silent_s: float = 3.5, min_fraction: float = 0.3,
               tail_s: float = 0.35) -> np.ndarray:
    """An utterance the voice has finished but kept padding with near-silence instead of stopping: once speech has
    been heard, ``silent_s`` of consecutive near-silent frames (RMS at or below ``silence_db`` dBFS) ends it, but not
    before ``min_fraction`` of its estimated frames; ``tail_s`` of the silence after the last voiced frame is kept."""
    n = int(round(sample_rate / frame_rate))
    k = len(audio) // n
    loud = np.square(audio[:k * n].reshape(k, n).astype(np.float64)).mean(1) > 10 ** (silence_db / 10)
    need = max(1, int(round(silent_s * frame_rate)))
    voiced, run, last = False, 0, 0
    for i, v in enumerate(loud):
        if v:
            voiced, run, last = True, 0, i + 1
        elif voiced:
            run += 1
            if run >= need and i + 1 >= min_fraction * estimated_frames:
                return audio[:last * n + int(round(tail_s * sample_rate))]
    return audio


def _default_backend() -> str:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "onnx"


class LibraTTS:
    def __init__(self, model: Optional[str] = None, backend: str = "auto", threads: Optional[int] = None,
                 batch_size: int = 64, compile: Optional[bool] = None):
        """model: a local model folder or a Hugging Face repo id (default: the released model).
        backend: "mlx" (Apple silicon GPU), "cuda" (NVIDIA GPU, PyTorch), "onnx" (ONNX Runtime, CPU) or "auto" (the
        first of these that is installed).
        batch_size: utterances of a long text generated together on a GPU (64: ~4 GB peak).
        compile: on CUDA, compile each layer with torch.compile where Triton is available (Linux): ~15% faster one
        sentence at a time, ~6 s more to load (default on; off for one-shot use)."""
        self.backend_name = _default_backend() if backend == "auto" else backend
        self.batch_size = batch_size
        self.cfg = Config(model_dir(model, self.backend_name))
        self.tok = Tokenizer(self.cfg)
        self.front = FrontEnd(self.cfg, self.tok)
        if self.backend_name == "mlx":
            from .mlx_backend import MLXBackend
            self.backend = MLXBackend(self.cfg)
        elif self.backend_name == "cuda":
            from .torch_backend import TorchBackend
            self.backend = TorchBackend(self.cfg, compile=compile)
        elif self.backend_name == "onnx":
            from .onnx_backend import OnnxBackend
            self.backend = OnnxBackend(self.cfg, threads=threads)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.sample_rate = self.cfg.sample_rate

    @property
    def voices(self) -> List[str]:
        return list(self.cfg.voices)

    @property
    def languages(self) -> List[str]:
        return sorted(self.cfg.languages)

    @property
    def styles(self) -> List[str]:
        return list(self.cfg.styles)

    @property
    def tags(self) -> List[str]:
        return list(self.cfg.tags)

    def synthesize(self, text: str, voice: Optional[str] = None, lang: str = "auto", style: Optional[str] = None,
                   pace: Optional[str] = None, seed: int = 0, sampling: Optional[Sampling] = None,
                   gap: float = 0.3) -> np.ndarray:
        """Speech for ``text`` as float32 samples at ``sample_rate``. lang: an ISO code ("en", "zh", ...) or "auto"
        (each sentence by its script, so one text can mix languages; Latin script = English). style: "warm",
        "whispering", ... (``styles``); pace: "slow" or "fast", with a style. Inline tags in the text (``tags``:
        <laugh>, <sigh>, ...) are performed where they stand. Long text is read utterance by utterance, joined by
        ``gap`` s."""
        voice = voice or self.cfg.default_voice
        runs = language_runs(text) if lang == "auto" else [(lang, text)]
        s = sampling or self.cfg.sampling
        instruction = self.cfg.style_text(style, pace)
        style_ids = self.tok.encode_style(instruction) if instruction else None
        utts = [(lg, u) for lg, t in runs for u in self.front.utterances(t, lg)]
        if not utts:
            return np.zeros(0, np.float32)
        prompts = [build_prompt(self.cfg, self.front.encode(u, lg), voice, lg, style_ids) for lg, u in utts]
        est = [len(u) / self.cfg.language(lg)["chars_per_second"] * self.cfg.frame_rate for lg, u in utts]
        caps = [min(self.cfg.max_frames, int(e * 2.5) + 75) for e in est]
        seeds = [(seed * 1000003 + 7919 * i) & 0x7FFFFFFF for i in range(len(utts))]
        if self.backend_name in ("mlx", "cuda"):
            B = self.batch_size
            codes = [c for i in range(0, len(prompts), B)
                     for c in self.backend.generate(prompts[i:i + B], seeds[i:i + B], caps[i:i + B], s)]
        else:
            codes = [self.backend.generate(p, sd, cap, s) for p, sd, cap in zip(prompts, seeds, caps)]
        pause = np.zeros(int(gap * self.sample_rate), np.float32)
        parts: List[np.ndarray] = []
        for c, e in zip(codes, est):
            parts += [end_run_on(self.backend.decode(c), e, self.sample_rate, self.cfg.frame_rate), pause]
        return np.concatenate(parts[:-1])

    def save(self, path: str, audio: np.ndarray):
        import soundfile as sf
        sf.write(path, audio, self.sample_rate, subtype="PCM_16")
