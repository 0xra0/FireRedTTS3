<div align="center">
    <h1>
    FireRedTTS3 (Optimized & Fast Edition)
    </h1>
    <p>
    Official PyTorch implementation with <b>SDPA Attention Support</b>, <b>Prompt Conditioning Caching</b>, and <b>Configurable Flow Steps</b> <br>
    <b><em>FireRedTTS3: Unified Speech Generation and Editing with Semantically Enriched Speech Representations</em></b>
    </p>
    <p>
    <img src="assets/firered_logo.png" alt="FireRedTTS Logo" width="248" height="68">
    </p>
    <p>
    <a href="https://arxiv.org/abs/2608.17492"><img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="technical report"></a>
    <a href="https://fireredteam.github.io/demos/firered_tts_3/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="version"></a>
    <a href="https://huggingface.co/FireRedTeam/FireRedTTS3"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="HF-model"></a>
    <a href="https://www.modelscope.cn/models/FireRedTeam/FireRedTTS3"><img src="https://img.shields.io/badge/ModelScope-Model%20Page-624AFF?logo=modelscope&logoColor=white" alt="ModelScope-model"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0"></a>
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Attention-SDPA%20Native-brightgreen" alt="SDPA">
    </p>
</div>

---

## Overview

**FireRedTTS3** is a state-of-the-art unified speech generation and editing system built upon **semantically enriched continuous speech representations**. It provides high-fidelity zero-shot voice cloning across 24 languages and 21 Chinese dialects, instruction-driven voice design, and fine-grained acoustic and semantic audio editing.

This repository includes **high-performance local enhancements** tailored for real-world production and batch dubbing workflows:
- 🚀 **PyTorch Native SDPA (Scaled Dot-Product Attention)**: Removed hard dependency on `flash-attn` compilation. Works out-of-the-box on modern PyTorch across CUDA architectures.
- ⚡ **Prompt Conditioning Caching**: When synthesizing multiple sentences with the same reference audio (e.g. video/audiobook dubbing), redundant RedAE latent encoding and CAM++ speaker embedding extraction are bypassed via fast waveform hashing, dramatically reducing GPU latency.
- ⏱️ **Configurable Flow-Matching Steps**: Easily adjust diffusion/flow-matching steps (`FIRERED_N_TIMESTEPS=6/8/10`) via environment variable for 1.5x–2x faster inference.
- 🛠️ **Clean Patching Tool**: Includes [`patch_firered_fast.py`](patch_firered_fast.py) for reproducible patching of vanilla codebases.

---

## Highlights ✨

* 🌍 **Multilingual — 24 Languages**: Best average WER/CER (avg 3.754%) and speaker similarity on MiniMax-MLS-Test (avg 84.8%), plus top cloning accuracy on Seed-TTS-eval (avg 3.04%).
  - Supported: `Arabic`, `Cantonese`, `Chinese`, `Czech`, `Dutch`, `English`, `Finnish`, `French`, `German`, `Greek`, `Hindi`, `Indonesian`, `Italian`, `Japanese`, `Korean`, `Polish`, `Portuguese`, `Romanian`, `Russian`, `Spanish`, `Thai`, `Turkish`, `Ukrainian`, `Vietnamese`.
* 🗣️ **Multi-Dialect — 21 Chinese Dialects**: High fidelity zero-shot cloning across major Chinese dialect regions:
  - Supported: `Anhui`, `Fujian`, `Gansu`, `Guizhou`, `Hebei`, `Henan`, `Hubei`, `Hunan`, `Jiangxi`, `Liaoning`, `Minnan`, `Ningxia`, `Shaanxi`, `Shandong`, `Shanghai`, `Shanxi`, `Sichuan`, `Tianjin`, `Wenzhou`, `Wu`, `Yunnan`.
* 🎨 **Instruction-Controlled Voice Design**: Synthesize new voices directly from natural language prompts describing gender, age, timbre, emotion, and pace without reference audio.
* ✂️ **Free-Form Speech Editing**: Semantic modification (word insertion, deletion, replacement) and acoustic adjustments (speed, pitch, volume) via text prompts.
* ⚡ **Dubbing Optimized**: Up to 40%+ faster sequential inference when reusing reference speaker clips.

---

## Contents

- [Key Optimizations](#key-optimizations-in-this-edition)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Model Download](#model-download)
- [Text Frontend & Normalization](#text-frontend--normalization)
- [Web UI (Gradio Demo)](#web-ui-gradio-demo)
- [Python API Usage](#python-api-usage)
  - [1. Zero-Shot Voice Cloning](#1-zero-shot-voice-cloning)
  - [2. Accelerated Dubbing / Batch Synthesis](#2-accelerated-dubbing--batch-synthesis)
  - [3. Voice Design (Prompt-based)](#3-voice-design-prompt-based)
  - [4. Semantic Speech Editing](#4-semantic-speech-editing)
  - [5. Acoustic Speech Editing](#5-acoustic-speech-editing)
- [Environment Variables](#environment-variables)
- [Performance Benchmarks](#performance-benchmarks)
- [Usage Disclaimer](#usage-disclaimer)
- [Citation & Acknowledgements](#citation--acknowledgements)
- [License](#license)

---

## Key Optimizations in this Edition

### 1. PyTorch Native SDPA
Vanilla FireRedTTS3 hardcoded `attn_implementation="flash_attention_2"`. This often led to installation failures when prebuilt wheels for `flash-attn` did not match specific CUDA/PyTorch configurations.
- Replaced with PyTorch native `sdpa` in `fireredtts3/llm/fireredtts3_base.py` and `fireredtts3/redae/redae.py`.
- Enables seamless out-of-the-box execution on modern GPUs with zero compilation hassle while maintaining fast memory-efficient attention.

### 2. Prompt Conditioning Cache
In dubbing and long-form narration, the reference voice clip remains constant while hundreds of sentences are synthesized. Re-encoding audio with RedAE and recomputing CAM++ speaker embeddings on every iteration is wasteful.
- Hashes CPU reference audio via SHA-1.
- Automatically caches `prompt_latents`, `spk_emb`, and `prompt_num_samples`.
- Yields immediate throughput gains for batch processing.

### 3. Configurable Flow Timesteps
Flow-matching timesteps can be adjusted dynamically via the environment variable `FIRERED_N_TIMESTEPS`:
```bash
# Default quality (10 steps)
python gradio_base.py --host 0.0.0.0 --port 7860

# Fast mode (6 steps, ~40% faster)
FIRERED_N_TIMESTEPS=6 python gradio_base.py --host 0.0.0.0 --port 7860

# Balanced mode (8 steps)
FIRERED_N_TIMESTEPS=8 python gradio_base.py --host 0.0.0.0 --port 7860
```

---

## System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 to 3.12
- **GPU**: NVIDIA GPU with >= 8GB VRAM (16GB+ recommended for batching & Instruct model)
- **CUDA**: 11.8 or 12.x
- **System Libraries**: `ffmpeg`, `libsndfile1`

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 git git-lfs curl
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/0xra0/FireRedTTS3.git
cd FireRedTTS3
```

### 2. Set Up Virtual Environment

```bash
# Using conda
conda create -n fireredtts3 python=3.10 -y
conda activate fireredtts3

# Or using venv
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch & Dependencies

Install PyTorch 2.x with your CUDA version:
```bash
# For CUDA 12.1+
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install repository requirements
pip install -r requirements.txt

# Install Gradio for Web UI
pip install gradio
```

> **Note on `flash_attn`**: Because this edition uses native PyTorch `sdpa`, installing `flash_attn` is **optional**. If you wish to build it anyway, run `pip install flash-attn --no-build-isolation`.

---

## Model Download

Pretrained weights must be placed inside the `pretrained_models/` directory.

### Option A: Using Hugging Face CLI (Recommended)

```bash
pip install -U "huggingface_hub[cli]"
hf download FireRedTeam/FireRedTTS3 --local-dir pretrained_models/
```

### Option B: Using ModelScope

```bash
pip install -U modelscope
modelscope download --model FireRedTeam/FireRedTTS3 --local_dir pretrained_models/
```

### Expected Directory Structure

```text
pretrained_models/
├── fireredtts3_base/
│   ├── config.json
│   └── model.safetensors
├── redae/
│   ├── config.json
│   └── model.safetensors
├── campp/
│   └── campplus_voxceleb.bin
└── text_tokenizer/
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```

---

## Text Frontend & Normalization

### 1. Language Identification (Optional)
If you wish to allow automatic language detection when passing `language=None`:
```bash
mkdir -p fireredtts3/utils/llm_tn/models
curl -L -o fireredtts3/utils/llm_tn/models/lid.176.ftz \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

### 2. Text Normalization (TN)
Text Normalization converts numbers, currencies, dates, and acronyms into speakable words.
- **Local Rule-based TN (`wetext`)**: Enabled by default (`use_wetext=True`). Supports Chinese and English.
- **LLM-based TN (`use_llm_tn=True`)**: Recommended for all 24 multilingual targets. Connects to any OpenAI-compatible endpoint.

```bash
cp .env.example .env
```
Edit `.env`:
```env
LLM_TN_API_URL=https://api.deepseek.com/chat/completions
LLM_TN_API_KEY=sk-your-api-key
LLM_TN_MODEL=deepseek-v4-flash
```

---

## Web UI (Gradio Demo)

Launch the interactive web interface:

### 1. FireRedTTS3-Base (Voice Cloning)
```bash
# Run default (10 steps)
python gradio_base.py --host "0.0.0.0" --port 7860

# Run accelerated (6 flow steps)
FIRERED_N_TIMESTEPS=6 python gradio_base.py --host "0.0.0.0" --port 7860
```

### 2. FireRedTTS3-Instruct (Voice Design & Editing)
```bash
python gradio_instruct.py --host "0.0.0.0" --port 7860
```

> **Reverse Proxy / Remote VS Code**: Add `--root-path https://your-domain/proxy/7860/` if hosting behind an HTTP proxy.

![Gradio UI](assets/tts3_gradio_demo.jpg)

---

## Python API Usage

### 1. Zero-Shot Voice Cloning

```python
import torch
import torchaudio
from fireredtts3.core import FireRedTTS3

# Initialize Base model
tts = FireRedTTS3(
    model_dir="pretrained_models",
    use_wetext=True,
    use_llm_tn=False,
)

# Reference audio (3 - 15 seconds recommended)
prompt_audio, prompt_audio_sr = torchaudio.load("prompt.wav")
prompt_text = "The prompt audio transcript in spoken words."

# Text to generate
text = "FireRedTTS3 provides state-of-the-art voice synthesis with unmatched speaker similarity."

gen_audio, gen_audio_sr = tts.generate(
    language="English",    # Or None for auto-detection
    prompt_text=prompt_text,
    prompt_audio=prompt_audio,
    prompt_audio_sr=prompt_audio_sr,
    text=text,
    do_tn=True,
)

torchaudio.save("output_cloned.wav", gen_audio.cpu(), gen_audio_sr)
print("Saved to output_cloned.wav")
```

### 2. Accelerated Dubbing / Batch Synthesis

When processing multiple lines of dialog with the same voice, FireRedTTS3 caches the prompt conditioning:

```python
import torchaudio
from fireredtts3.core import FireRedTTS3

tts = FireRedTTS3("pretrained_models", use_wetext=True)

prompt_audio, prompt_sr = torchaudio.load("character_voice.wav")
prompt_text = "Transcript of character voice sample."

lines = [
    "Welcome to our virtual tour.",
    "Notice the rich acoustic details and natural prosody.",
    "Subsequent sentences reuse cached prompt latents instantly."
]

for idx, line in enumerate(lines):
    # First iteration computes and caches RedAE latents & CAM++ embeddings
    # Iterations 2+ announce: [FAST] Reusing cached prompt latents + speaker embedding
    audio, sr = tts.generate(
        language="English",
        prompt_text=prompt_text,
        prompt_audio=prompt_audio,
        prompt_audio_sr=prompt_sr,
        text=line,
    )
    torchaudio.save(f"dub_line_{idx:03d}.wav", audio.cpu(), sr)
```

### 3. Voice Design (Prompt-based)

Generate brand-new character voices without reference audio:

```python
import torchaudio
from fireredtts3.core import FireRedTTS3Instruct

instruct = FireRedTTS3Instruct("pretrained_models", use_wetext=True)

instruction = "A middle-aged detective with a raspy, deep voice speaking slowly and thoughtfully."
text = "Something doesn't feel right about this scene."

gen_audio, sr, voice_plan = instruct.generate_voice_design(
    instruction=instruction,
    text=text,
)

torchaudio.save("voice_design.wav", gen_audio.cpu(), sr)
print(f"Generated Voice Plan:\n{voice_plan}")
```

### 4. Semantic Speech Editing

Modify existing audio content using natural language instructions:

```python
import torchaudio
from fireredtts3.core import FireRedTTS3Instruct

instruct = FireRedTTS3Instruct("pretrained_models")

audio_in, sr = torchaudio.load("original_sentence.wav")

# Instruction to modify words
gen_audio, gen_sr, rewritten_text = instruct.generate_semantic_edit(
    instruction="Replace 'cats' with 'playful dogs'.",
    audio_in=audio_in,
    audio_in_sr=sr,
)

torchaudio.save("semantic_edit.wav", gen_audio.cpu(), gen_sr)
print(f"Rewritten Text with Edit Masks: {rewritten_text}")
```

### 5. Acoustic Speech Editing

Adjust speed, pitch, or volume of input speech using formal instruction templates:

```python
import torchaudio
from fireredtts3.core import FireRedTTS3Instruct

instruct = FireRedTTS3Instruct("pretrained_models")
audio_in, sr = torchaudio.load("original_speech.wav")

# Speed: "adjust the speed to X" (X in [0.5, 2.0], step 0.1)
# Pitch: "shift the pitch by N step(s)" (N in {-6, ..., -1, 1, ..., +6})
# Volume: "adjust the volume to X" (X in [0.3, 2.0], step 0.1)

gen_audio, gen_sr = instruct.generate_acoustic_edit(
    instruction="adjust the speed to 1.3x",
    audio_in=audio_in,
    audio_in_sr=sr,
)

torchaudio.save("acoustic_edit.wav", gen_audio.cpu(), gen_sr)
```

---

## Environment Variables

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `FIRERED_N_TIMESTEPS` | Integer | `10` | Number of flow-matching inference steps (e.g. `6`, `8`, `10`). Lower values accelerate generation. |
| `LLM_TN_API_URL` | String | – | OpenAI-compatible endpoint for LLM text normalization. |
| `LLM_TN_API_KEY` | String | – | API key for text normalization service. |
| `LLM_TN_MODEL` | String | `deepseek-v4-flash` | Target LLM model name. |

---

## Performance Benchmarks

### Zero-Shot Voice Cloning — Seed-TTS-eval

Best in **bold**, second best in <ins>underline</ins>. Evaluated using [Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval):

| Model | Test-EN<br>WER/SIM | Test-ZH<br>CER/SIM | Test-Hard<br>CER/SIM | Avg<br>WER/SIM |
| :--- | :---: | :---: | :---: | :---: |
| CosyVoice3-1.5B | 2.22 / 72.0 | 1.12 / 78.1 | **5.83** / 75.8 | <ins>3.06</ins> / 75.3 |
| DiTAR | 1.69 / 73.5 | 1.02 / 75.3 | – / – | – / – |
| F5-TTS | 2.00 / 67.0 | 1.53 / 76.0 | 8.67 / 71.3 | 4.10 / 71.4 |
| FireRedTTS-2 | 1.95 / 66.5 | 1.14 / 73.6 | 8.98 / 70.3 | 4.02 / 70.1 |
| IndexTTS2 | 2.23 / 70.6 | 1.03 / 76.5 | 7.12 / 75.5 | 3.46 / 74.2 |
| MegaTTS3 | 2.79 / <ins>77.1</ins> | 1.52 / 79.0 | – / – | – / – |
| MiniMax-Speech | 1.65 / 69.2 | **0.83** / 78.3 | – / – | – / – |
| Qwen3-TTS | **1.23** / 71.7 | 1.22 / 77.0 | 6.76 / 74.8 | 3.07 / 74.5 |
| Seed-TTS | 2.25 / 76.2 | 1.12 / 79.6 | 7.59 / 77.6 | 3.65 / 77.8 |
| VibeVoice | 3.04 / 68.9 | 1.16 / 74.4 | – / – | – / – |
| VoxCPM2 | 1.84 / 75.3 | <ins>0.97</ins> / 79.5 | 8.13 / 75.3 | 3.65 / 76.7 |
| dots.tts (Pretrain) | 1.80 / 77.0 | <ins>0.97</ins> / <ins>80.4</ins> | 6.65 / **78.8** | 3.14 / <ins>78.7</ins> |
| **FireRedTTS3-Base** | <ins>1.64</ins> / **77.2** | 1.01 / **80.9** | <ins>6.50</ins> / <ins>78.4</ins> | **3.04** / **78.8** |

### Multilingual Evaluation — MiniMax-MLS-Test

FireRedTTS3 achieves an average WER/CER of **3.75%** and speaker similarity of **84.8%** across 24 languages, outperforming leading proprietary and open-source models:

<details>
<summary><b>WER / CER (↓) across 24 Languages (click to expand)</b></summary>

| Language | Minimax | ElevenLabs | VoxCPM2 | FishAudio S2 | dots.tts (Pretrain) | **FireRedTTS3** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Arabic | **1.67** | **1.67** | 13.05 | 3.50 | 37.91 | <ins>1.75</ins> |
| Cantonese | <ins>34.11</ins> | 51.51 | 38.58 | **30.67** | 37.91 | 40.32 |
| Chinese | 2.25 | 16.03 | 1.14 | **0.73** | 1.08 | <ins>0.91</ins> |
| Czech | 3.88 | **2.11** | 24.13 | <ins>2.84</ins> | 5.05 | 3.17 |
| Dutch | 1.14 | **0.80** | <ins>0.91</ins> | 0.99 | 1.20 | 1.15 |
| English | 2.16 | 2.34 | 2.29 | 1.62 | **1.06** | <ins>2.12</ins> |
| Finnish | 4.67 | <ins>2.96</ins> | **2.63** | 3.33 | 3.44 | 3.10 |
| French | 4.10 | 5.22 | 4.53 | **3.05** | <ins>3.82</ins> | 5.28 |
| German | 1.91 | <ins>0.57</ins> | 0.68 | **0.55** | 1.03 | 0.69 |
| Greek | 2.02 | **0.99** | 2.84 | 5.74 | 2.97 | <ins>1.24</ins> |
| Hindi | <ins>6.96</ins> | **5.83** | 19.70 | 14.64 | 14.32 | 7.02 |
| Indonesian | 1.24 | **1.06** | <ins>1.08</ins> | 1.46 | 2.71 | 1.42 |
| Italian | <ins>1.54</ins> | 1.74 | 1.56 | **1.27** | 3.16 | 2.28 |
| Japanese | <ins>3.52</ins> | 10.65 | 4.63 | **2.76** | 7.16 | 3.60 |
| Korean | <ins>1.75</ins> | 1.87 | 1.96 | **1.18** | 5.30 | 2.42 |
| Polish | 1.42 | **0.77** | <ins>1.14</ins> | 1.26 | 2.72 | 1.22 |
| Portuguese | 1.88 | <ins>1.33</ins> | 1.94 | **1.14** | 1.64 | 1.79 |
| Romanian | 2.88 | **1.35** | 21.58 | 10.74 | 3.36 | <ins>1.93</ins> |
| Russian | 4.28 | 3.88 | 3.63 | **2.40** | 3.64 | <ins>3.28</ins> |
| Spanish | 1.03 | 1.08 | 1.44 | **0.91** | <ins>0.96</ins> | 1.21 |
| Thai | <ins>2.70</ins> | 73.94 | 2.96 | 4.23 | 7.45 | **1.87** |
| Turkish | 1.52 | **0.70** | <ins>0.82</ins> | 0.87 | 5.45 | 0.92 |
| Ukrainian | 1.08 | <ins>1.00</ins> | 6.32 | 2.30 | 1.61 | **0.55** |
| Vietnamese | <ins>0.88</ins> | 73.42 | 3.31 | 7.41 | 3.85 | **0.86** |
| **Average** | <ins>3.77</ins> | 10.95 | 6.79 | 4.40 | 6.60 | **3.75** |

</details>

<details>
<summary><b>Speaker Similarity (SIM ↑) across 24 Languages (click to expand)</b></summary>

| Language | Minimax | ElevenLabs | VoxCPM2 | FishAudio S2 | dots.tts (Pretrain) | **FireRedTTS3** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Arabic | 73.6 | 70.6 | **79.1** | 75.0 | 77.5 | <ins>78.9</ins> |
| Cantonese | 77.8 | 67.0 | 83.5 | 80.5 | **84.7** | <ins>83.9</ins> |
| Chinese | 78.0 | 67.7 | <ins>82.5</ins> | 81.6 | 82.3 | **84.2** |
| Czech | 79.6 | 68.5 | 78.3 | 79.8 | <ins>83.8</ins> | **86.1** |
| Dutch | 73.8 | 68.0 | 80.8 | 73.0 | <ins>81.4</ins> | **84.3** |
| English | 75.6 | 61.3 | 85.4 | 79.7 | **86.9** | <ins>86.8</ins> |
| Finnish | 83.5 | 75.9 | <ins>89.0</ins> | 81.9 | 88.0 | **89.9** |
| French | 62.8 | 53.5 | 73.5 | 69.8 | <ins>78.2</ins> | **81.0** |
| German | 73.3 | 61.4 | <ins>80.3</ins> | 76.7 | 79.5 | **83.3** |
| Greek | 82.6 | 73.3 | 86.0 | 79.5 | <ins>87.6</ins> | **89.3** |
| Hindi | 81.8 | 73.0 | <ins>85.6</ins> | 82.1 | 84.5 | **87.2** |
| Indonesian | 72.9 | 66.0 | 80.0 | 76.3 | <ins>80.8</ins> | **83.3** |
| Italian | 69.9 | 57.9 | 78.0 | 74.7 | **84.5** | <ins>83.6</ins> |
| Japanese | 77.6 | 73.8 | <ins>82.8</ins> | 79.6 | **83.1** | <ins>82.8</ins> |
| Korean | 77.6 | 70.0 | 83.3 | 81.7 | <ins>84.3</ins> | **86.6** |
| Polish | 80.2 | 72.9 | <ins>88.4</ins> | 81.9 | 87.3 | **89.8** |
| Portuguese | 80.5 | 71.1 | <ins>83.7</ins> | 78.1 | 83.1 | **86.3** |
| Romanian | <ins>80.9</ins> | 69.9 | 79.7 | 73.3 | **86.2** | **86.2** |
| Russian | 76.1 | 67.6 | 81.1 | 79.0 | <ins>83.0</ins> | **84.7** |
| Spanish | 76.2 | 61.5 | 83.1 | 77.6 | <ins>83.9</ins> | **86.3** |
| Thai | 80.0 | 58.8 | **84.0** | 78.6 | <ins>83.8</ins> | 83.3 |
| Turkish | 77.9 | 59.6 | <ins>87.1</ins> | 83.5 | **87.4** | 86.6 |
| Ukrainian | 73.0 | 64.7 | <ins>79.8</ins> | 74.7 | **80.5** | <ins>79.8</ins> |
| Vietnamese | 74.3 | 36.9 | 80.6 | 74.0 | <ins>80.7</ins> | **81.3** |
| **Average** | 76.6 | 65.5 | 82.3 | 78.0 | <ins>83.5</ins> | **84.8** |

</details>

---

## Usage Disclaimer

- This software includes zero-shot voice cloning capabilities. It is released **solely for academic research and authorized creative purposes**.
- **DO NOT** use this software for any fraudulent, deceptive, or illegal activities.
- The authors and maintainers assume no liability for misuse.

---

## Citation & Acknowledgements

If you find FireRedTTS3 helpful in your research or project, please cite:

```bibtex
@article{fireredtts3,
  title   = {FireRedTTS3: Unified Speech Generation and Editing with Semantically Enriched Speech Representations},
  author  = {Shen, Feiyu and Xie, Kun and Wu, Yichen and Dai, Ziqi and Han, Yichen and Li, Junjie and Geng, Xuelong and Xie, Fenglong and Xie, Lei and Tang, Xu and others},
  journal = {arXiv preprint arXiv:2608.17492},
  year    = {2026},
}
```

Special thanks to the open-source projects foundational to this work:
- [Qwen3](https://github.com/QwenLM/Qwen3) & [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [DiTAR](https://arxiv.org/abs/2502.03930) for patch-level diffusion autoregression
- [CAM++](https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k) for speaker embedding extraction
- [fastText](https://fasttext.cc/docs/en/language-identification.html)
- [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)

---

## License

Released under the [Apache-2.0](LICENSE) License.
