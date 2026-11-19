<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.png">
  <img src="assets/banner.png" alt="Libra TTS: 16 languages, 8 voices, 0.3B parameters, faster than real time on a Mac or a PC" width="100%">
</picture>

**Multilingual text-to-speech model with 16 languages and 8 voices in 0.3B parameters, running faster than real time**

[![Try it live](https://img.shields.io/badge/Try%20it-live%20demo-2F64CF?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Ekam/Libra-TTS)
[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Ekam/Libra-TTS)
[![Listen to 13 samples](https://img.shields.io/badge/%E2%96%B6%20Listen-13%20samples-8B7BFF?style=for-the-badge)](https://huggingface.co/Ekam/Libra-TTS#samples)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2F80ED)](LICENSE)
![Apple silicon: MLX](https://img.shields.io/badge/Apple%20silicon-MLX-111111?logo=apple&logoColor=white)
![NVIDIA: CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/CPU-ONNX%20Runtime-005CED?logo=onnx&logoColor=white)
![Runs on macOS, Windows and Linux](https://img.shields.io/badge/runs%20on-macOS%20%7C%20Windows%20%7C%20Linux-555555)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![16 languages](https://img.shields.io/badge/languages-16-2EA44F)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FEkam%2FLibra-TTS&query=%24.downloads&label=downloads&color=FFD21E&logo=huggingface&logoColor=black)](https://huggingface.co/Ekam/Libra-TTS)

</div>

## Highlights

- **Lowest English error rate of the open models we tested**: 1.64% WER, ahead of Magpie-357M, Kokoro-82M and both Qwen3-TTS models (2.74% at 1.7B, 3.21% at 0.6B).
- **16 languages**, English, Chinese, Japanese, Korean,
German, French, Spanish, Italian, Portuguese, Russian, Arabic, Hindi, Punjabi, Bengali, Urdu and Indonesian.
- **Fast**: one sentence at a time 7.7x faster than real time on a MacBook Pro and 8.7x on an RTX 3090; about 70x
  reading long text on either; 2.4x on a CPU with ONNX.
- **Runs everywhere**: Apple silicon (MLX), NVIDIA GPUs on Windows and Linux (PyTorch with CUDA graphs) and any
  CPU (ONNX Runtime); the live demo reads on a GPU.

## Quick start (Mac, Apple silicon)

```bash
pip install "libra-tts[ja] @ git+https://github.com/ekambrar100/Libra-TTS"
libra-tts "Some days, the smallest moments are the ones we remember." -o hello.wav
```

```python
from libra_tts import LibraTTS

tts = LibraTTS()                     # downloads the weights once (~1 GB)
audio = tts.synthesize("Hello from Libra.", voice="nina")
tts.save("hello.wav", audio)

tts.synthesize("今天天气很好，我们去公园散步吧。", voice="aliya")   # language detected from the script
tts.synthesize("Hola, ¿qué tal?", voice="jon", lang="es")           # Latin-script languages: pass lang
```

Long text is split at sentence ends and read up to 64 utterances at a time. On Apple silicon Libra runs on the GPU through
[MLX](https://github.com/ml-explore/mlx).

## Windows and Linux

**With an NVIDIA graphics card** (GeForce RTX 20-series or newer). On Windows, first install PyTorch with GPU support
(a plain `pip install torch` on Windows has none):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then, on Windows or Linux:

```bash
pip install "libra-tts[cuda,ja] @ https://github.com/ekambrar100/Libra-TTS/archive/refs/heads/main.zip"
libra-tts "Hello from Libra." -o hello.wav
```

Libra finds the GPU by itself. On an RTX 3090 it reads one sentence at a time 8.7x faster than real time, and long
text 72x (11 minutes of audio in 9 seconds, 3.6 GB of GPU memory). On Linux, `LibraTTS()` also compiles its layers
when it loads (torch.compile: about 6 seconds more, cached on disk), which reads one sentence at a time about 15%
faster (RTX 4090: 10.9x to 12.6x); `compile=False` skips it, as the command line does.

**On the CPU** (any PC, Intel Macs), with ONNX Runtime:

```bash
pip install "libra-tts[onnx,ja] @ https://github.com/ekambrar100/Libra-TTS/archive/refs/heads/main.zip"
```

On Windows, use Python 3.10 or newer from python.org and run these commands in PowerShell.

## Voices and languages

| voice | | voice | |
|---|---|---|---|
| `nina` (default) | female, warm | `omar` | male, deep |
| `sofia` | female, gentle | `jon` | male, clear |
| `aliya` | female, rich | `bob` | male, storyteller |
| `maya` | female, cozy | `sam` | male |

Every voice speaks every language: `en zh ja ko de fr es it pt ru ar hi pa bn ur id`.

`lang="auto"` (the default) picks the language from the script; Latin-script text is read as English unless you pass
`lang`. Options: `seed` (same text + seed = same audio), `sampling=Sampling(temperature=...)`, `gap` (pause between
sentences, 0.3 s).

## Styles

```python
tts.synthesize("It can keep a secret.", voice="nina", style="whispering")
tts.synthesize("Okay, you win.", voice="jon", style="wry")
tts.synthesize("Tomorrow? Light rain.", voice="aliya", style="tense", pace="slow")
```

```bash
libra-tts "Good news, everyone!" --style excited -o news.wav
```

Styles: `warm cheerful excited sad tense frightened angry calm serious wry tender whispering dramatic documentary`, with
`pace="slow"` or `"fast"`. 


## Benchmarks

Error rate of a speech recognizer on the generated audio (Whisper-large-v3; Paraformer-zh for Chinese; Omnilingual
ASR for Punjabi, Bengali, Urdu), seed-tts-eval scoring, lower is better. Every model ran with its own default settings
and text processing on the same sentences, one read each. Parameters are each model's own weights as released, without a separate audio decoder (Libra and Qwen3-TTS use the
same Qwen3-TTS codec decoder; Kokoro and KittenTTS have theirs built in). Qwen3-TTS's names count only the language
model it grew from; its released weights are 1.9B (1.7B) and 0.9B (0.6B).

| model | parameters | English WER | 10-language average |
|---|---:|---:|---:|
| **Libra TTS** | 0.3B | **1.64%** | **1.77%** |
| Qwen3-TTS 1.7B | 1.9B | 2.74% | 2.75% |
| Qwen3-TTS 0.6B | 0.9B | 3.21% | 2.62% |
| Magpie-357M | 357M | 1.77% | – |
| Kokoro-82M | 82M | 1.80% | – |
| MOSS-TTS-Nano | 100M | 3.06% | 8.83% |
| Soprano-1.1 | 80M | 3.45% | – |
| PocketTTS | 100M | 3.81% | – |
| KittenTTS mini 0.8 | 74M | 4.33% | – |

English: Seed-TTS-eval test-en + LibriSpeech-PC test-clean, 2,215 sentences. Ten languages: the MiniMax multilingual
test set, 100 sentences per language ("–": the model does not cover all ten). Libra vs Qwen3-TTS on the average:
−0.98 points vs 1.7B and −0.85 vs 0.6B (95% CIs exclude zero).

| error % | parameters | en | zh | ja | ko | de | fr | es | it | pt | ru |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Libra TTS** | **0.3B** | 1.07 | 0.98 | 3.84 | 2.08 | **0.50** | **2.74** | **0.60** | **1.00** | **1.07** | **3.85** |
| Qwen3-TTS 1.7B | 1.9B | **0.94** | **0.75** | **3.47** | 4.87 | 2.39 | 4.71 | 1.47 | 2.05 | 2.18 | 4.69 |
| Qwen3-TTS 0.6B | 0.9B | 1.25 | 1.15 | 6.02 | **1.92** | 0.90 | 4.11 | 1.55 | 1.74 | 2.33 | 5.26 |



## Speed

| MacBook Pro, M5 Max | Libra TTS | Qwen3-TTS 0.6B size | Qwen3-TTS 1.7B size |
|---|---:|---:|---:|
| GPU (MLX), one sentence at a time | **7.7x** real time | 5.1x | 3.6x |
| peak memory | **2.0 GB** | 3.8 GB | 5.8 GB |

20 English sentences of 5-25 s, end to end (text in, 24 kHz audio out); all three on the same MLX engine. Long text
is read up to 64 utterances at a time: 15 minutes of audio in 13 seconds (69x real time, 4 GB peak). ONNX Runtime on the
same Mac's CPU: 2.4x real time.

| NVIDIA RTX 3090 (PyTorch, CUDA) | Libra TTS |
|---|---:|
| one sentence at a time (the same 20 sentences) | **8.7x** real time, 2.4 GB of GPU memory |
| long text, 64 utterances at a time | **72x** real time, 3.6 GB |

## How it was made

Libra TTS is distilled from a fine-tuned [Qwen3-TTS-12Hz-1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
teacher. It has a 14-layer low-rank talker and a 5-layer residual-code predictor (307M parameters) built from the teacher's own
weights through SVD (and then trained on over 9,000 hours of the teacher's speech) and the Qwen3-TTS 12 Hz codec decoder (114M). A final
round of online reinforcement learning (GRPO) on the model's own readings in all 16 languages and lowered its error rate (ten-language average 1.93% to 1.77%).

## License

Apache 2.0
