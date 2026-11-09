#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def backup_once(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".before_fast_patch")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"backup: {bak}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"{label}: already patched")
        return text
    if old not in text:
        raise RuntimeError(f"{label}: expected source block not found; file version differs")
    print(f"{label}: patched")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "repo",
        nargs="?",
        default=".",
        help="FireRedTTS3 repository root (default: current directory)",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    core = repo / "fireredtts3" / "llm" / "fireredtts3_base.py"
    gradio = repo / "gradio_base.py"

    for p in (core, gradio):
        if not p.is_file():
            raise SystemExit(f"not found: {p}")
        backup_once(p)

    # ------------------------------------------------------------------
    # 1) Cache RedAE prompt latents + CAM++ speaker embedding.
    #    The dubbing workflow reuses the exact same reference audio for
    #    hundreds/thousands of subtitle lines, so recomputing these on
    #    every request is wasted GPU work.
    # ------------------------------------------------------------------
    s = core.read_text(encoding="utf-8")

    s = replace_once(
        s,
        "import os\nimport torch\n",
        "import os\nimport hashlib\nimport torch\n",
        "hashlib import",
    )

    old_init = """        self.spk_extractor = CamppEmbedding(spk_ckpt_path)
        self.spk_extractor.to(self.device)
"""
    new_init = """        self.spk_extractor = CamppEmbedding(spk_ckpt_path)
        self.spk_extractor.to(self.device)

        # Fixed-reference dubbing optimization.
        # Cache expensive prompt conditioning (RedAE encode + CAM++ speaker
        # embedding) when the same reference waveform is used repeatedly.
        self._prompt_cache_key = None
        self._prompt_cache_latents = None
        self._prompt_cache_spk_emb = None
        self._prompt_cache_num_samples = None
        self._prompt_cache_hit_announced = False
"""
    s = replace_once(s, old_init, new_init, "prompt cache state")

    old_prompt = """        # Prompt audio
        prompt_audio = prompt_audio[:1]
        prompt_audio = torchaudio.functional.resample(prompt_audio, prompt_audio_sr, self.redae.sample_rate)
        prompt_audio_sr = self.redae.sample_rate
        prompt_audio = self.redae.pad_to_multiple_of(prompt_audio, self.redae.downsample_rate*self.tts_core.patch_size)
        prompt_audio = prompt_audio.to(self.device)
        prompt_latents = self.redae.encode(prompt_audio, prompt_audio_sr)
        prompt_latents = prompt_latents.to(torch.float32)
        # Spk emb
        spk_emb = self.spk_extractor.forward(prompt_audio, prompt_audio_sr)
        spk_emb = spk_emb.to(self.device)
"""
    new_prompt = """        # Prompt audio
        prompt_audio = prompt_audio[:1]

        # Hash the small reference waveform on CPU. This is cheap compared with
        # repeatedly running RedAE + CAM++ over a 5-20 s reference clip.
        _prompt_cpu = prompt_audio.detach().to("cpu").contiguous()
        _prompt_hash = hashlib.sha1(_prompt_cpu.numpy().tobytes()).digest()
        _prompt_key = (
            int(prompt_audio_sr),
            tuple(_prompt_cpu.shape),
            str(_prompt_cpu.dtype),
            _prompt_hash,
        )

        if _prompt_key == self._prompt_cache_key:
            prompt_latents = self._prompt_cache_latents
            spk_emb = self._prompt_cache_spk_emb
            prompt_num_samples = self._prompt_cache_num_samples
            if not self._prompt_cache_hit_announced:
                print("[FAST] Reusing cached prompt latents + speaker embedding", flush=True)
                self._prompt_cache_hit_announced = True
        else:
            prompt_audio = torchaudio.functional.resample(
                prompt_audio, prompt_audio_sr, self.redae.sample_rate
            )
            prompt_audio_sr = self.redae.sample_rate
            prompt_audio = self.redae.pad_to_multiple_of(
                prompt_audio,
                self.redae.downsample_rate * self.tts_core.patch_size,
            )
            prompt_audio = prompt_audio.to(self.device)

            prompt_latents = self.redae.encode(prompt_audio, prompt_audio_sr)
            prompt_latents = prompt_latents.to(torch.float32)

            # Speaker embedding
            spk_emb = self.spk_extractor.forward(prompt_audio, prompt_audio_sr)
            spk_emb = spk_emb.to(self.device)

            prompt_num_samples = prompt_audio.shape[1]

            self._prompt_cache_key = _prompt_key
            self._prompt_cache_latents = prompt_latents
            self._prompt_cache_spk_emb = spk_emb
            self._prompt_cache_num_samples = prompt_num_samples
            self._prompt_cache_hit_announced = False
            print("[FAST] Cached prompt latents + speaker embedding", flush=True)
"""
    s = replace_once(s, old_prompt, new_prompt, "prompt conditioning cache")

    s = replace_once(
        s,
        "        gen_audio = gen_audio[:, prompt_audio.shape[1]:]\n",
        "        gen_audio = gen_audio[:, prompt_num_samples:]\n",
        "cached prompt length",
    )

    core.write_text(s, encoding="utf-8")

    # ------------------------------------------------------------------
    # 2) Make flow-matching steps configurable without changing the Gradio
    #    API schema. Default remains official 10. Run with:
    #       FIRERED_N_TIMESTEPS=6 python gradio_base.py ...
    # ------------------------------------------------------------------
    g = gradio.read_text(encoding="utf-8")

    old_call = """            inference_cfg=cfg,
        )
"""
    new_call = """            inference_cfg=cfg,
            n_timesteps=int(os.environ.get("FIRERED_N_TIMESTEPS", "10")),
        )
"""
    g = replace_once(g, old_call, new_call, "FIRERED_N_TIMESTEPS")

    # Print effective step count once at startup, if this exact launch section exists.
    marker_old = """    tts = FireRedTTS3(
        'pretrained_models',
        use_wetext=False,
        use_llm_tn=False,
    )
"""
    marker_new = """    print(
        f"[FAST] FireRed flow steps: {int(os.environ.get('FIRERED_N_TIMESTEPS', '10'))}",
        flush=True,
    )
    tts = FireRedTTS3(
        'pretrained_models',
        use_wetext=False,
        use_llm_tn=False,
    )
"""
    if marker_new not in g and marker_old in g:
        g = g.replace(marker_old, marker_new, 1)
        print("startup step display: patched")
    else:
        print("startup step display: already patched or source differs")

    gradio.write_text(g, encoding="utf-8")

    print()
    print("Done.")
    print("Default quality (10 steps):")
    print("  python gradio_base.py --host 127.0.0.1 --port 7860")
    print("Faster test (6 steps):")
    print("  FIRERED_N_TIMESTEPS=6 python gradio_base.py --host 127.0.0.1 --port 7860")
    print("Safer compromise (8 steps):")
    print("  FIRERED_N_TIMESTEPS=8 python gradio_base.py --host 127.0.0.1 --port 7860")


if __name__ == "__main__":
    main()
