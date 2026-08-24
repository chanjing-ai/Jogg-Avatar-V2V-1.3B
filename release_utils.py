"""Lightweight validation helpers that do not initialize CUDA or load models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_MODEL_FILES = (
    "config.json",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "xlm-roberta-large/tokenizer.json",
    "google/umt5-xxl/spiece.model",
)


def split_model_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _require_path(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing {label}: {path}")


def _load_jobs(input_json: str, errors: list[str]) -> list[dict[str, Any]]:
    path = Path(input_json)
    _require_path(path, "input JSON", errors)
    if not path.is_file():
        return []
    try:
        jobs = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid input JSON {path}: {exc}")
        return []
    if not isinstance(jobs, list) or not jobs:
        errors.append("input JSON must contain a non-empty list")
        return []
    return jobs


def validate_inference_inputs(args: Any) -> list[dict[str, Any]]:
    errors: list[str] = []
    base_dir = Path(args.ckpt_dir)
    for relative_path in BASE_MODEL_FILES:
        _require_path(base_dir / relative_path, "base-model file", errors)

    model_paths = split_model_paths(args.infinitetalk_dir)
    if not model_paths:
        errors.append("infinitetalk_dir must contain at least one safetensors path")
    for model_path in model_paths:
        _require_path(model_path, "Jogg-Avatar checkpoint", errors)

    _require_path(Path(args.wav2vec_dir), "audio encoder", errors)
    jobs = _load_jobs(args.input_json, errors)
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            errors.append(f"job {index} must be an object")
            continue
        if not isinstance(job.get("prompt"), str) or not job["prompt"].strip():
            errors.append(f"job {index} must define a non-empty prompt")
        video_value = job.get("cond_video")
        audio_value = job.get("cond_audio")
        if not isinstance(video_value, str) or not video_value:
            errors.append(f"job {index} must define cond_video")
            continue
        if not isinstance(audio_value, dict) or set(audio_value) != {"person1"}:
            errors.append(f"job {index} cond_audio must contain only person1")
            continue
        person1 = audio_value["person1"]
        if not isinstance(person1, str) or not person1:
            errors.append(f"job {index} cond_audio.person1 must be a path")
            continue
        video_path = Path(video_value)
        _require_path(video_path, f"job {index} video", errors)
        _require_path(Path(person1), f"job {index} audio", errors)
        _require_path(video_path.with_suffix(".npy"), f"job {index} landmarks", errors)
        if not args.scene_seg:
            bbox_path = video_path.with_name(f"{video_path.stem}_bbox.json")
            _require_path(bbox_path, f"job {index} face box", errors)

    if errors:
        raise ValueError("Inference validation failed:\n- " + "\n- ".join(errors))
    return jobs
