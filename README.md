# Jogg-Avatar V2V 1.3B

[English](README.md) | [简体中文](README_zh.md)

Jogg-Avatar V2V 1.3B is an audio-driven avatar video generation model based
on [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and the
[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) approach. Given a
source video and a driving audio track, it preserves the source body,
background, and camera motion while regenerating a speech-synchronized face.

This first release contains the InfiniteTalk inference, preprocessing, and
training paths. FlashHead is a separate model variant and is not included yet.

## Requirements

- Linux with an NVIDIA GPU and a CUDA 12.8-compatible driver
- `ffmpeg` available on `PATH`
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.13, PyTorch 2.8.0, and CUDA 12.8 wheels pinned by this project

```bash
git clone https://github.com/chanjing-ai/Jogg-Avatar-V2V-1.3B.git
cd Jogg-Avatar-V2V-1.3B

# Inference
uv sync

# Face preprocessing
uv sync --extra preprocess

# Training and tests
uv sync --extra preprocess --extra train --extra test
```

FlashAttention is optional but recommended. Install it after the environment
has been synchronized:

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Models

Download the Wan base model and Chinese Wav2Vec2 encoder:

```bash
mkdir -p models
uv run hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir models/Wan2.1-T2V-1.3B
uv run hf download TencentGameMate/chinese-wav2vec2-base \
  --local-dir models/chinese-wav2vec2-base

# T5 tokenizer files used to build prompt embeddings locally.
uv run hf download google/umt5-xxl \
  --include "spiece.model" "tokenizer*" "special_tokens_map.json" \
  --local-dir models/Wan2.1-T2V-1.3B/google/umt5-xxl
```

Download the Jogg-Avatar InfiniteTalk checkpoint from Hugging Face:

```bash
uv run hf download cicada-ai/Jogg-Avatar-V2V-Infinite \
  --local-dir models/Jogg-Avatar-V2V-1.3B
```

The face preprocessing models `landmark.onnx` and `scrfd_500m_bnkps.onnx`
are required separately.

The expected layout is:

```text
models/
|-- Wan2.1-T2V-1.3B/
|   |-- config.json
|   |-- diffusion_pytorch_model.safetensors
|   |-- models_t5_umt5-xxl-enc-bf16.pth
|   |-- models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
|   |-- Wan2.1_VAE.pth
|   |-- google/umt5-xxl/
|   `-- xlm-roberta-large/
|-- chinese-wav2vec2-base/
|-- Jogg-Avatar-V2V-1.3B/
|   |-- model-00001-of-00002.safetensors
|   |-- model-00002-of-00002.safetensors
|   `-- training_init/audio_proj.safetensors
|-- landmark.onnx
`-- scrfd_500m_bnkps.onnx
```

Set `JOGG_AVATAR_MODEL_DIR` before running preprocessing or inference to use a
different model root. Model files, training data, and generated media are
intentionally excluded from Git.

## Inference

Prepare a JSON file from `examples/inference.json`. Each source video needs one
driving audio file under `cond_audio.person1`. Preprocess the video once to
create a same-name landmark array and face-box file:

```bash
uv run python preprocess_video.py \
  --data_dir /path/to/source_videos \
  --num_gpus 1
```

For `/path/to/source.mp4`, preprocessing creates `/path/to/source.npy` and
`/path/to/source_bbox.json`. Validate all paths without loading the 1.3B model:

```bash
uv run python inference.py \
  --input_json examples/inference.json \
  --validate_only
```

Run single-GPU inference:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference.py \
  --input_json examples/inference.json \
  --sample_steps 8 \
  --mode streaming
```

Outputs are written under `results/`. For multiple GPUs, launch with
`torchrun` and set `--ulysses_size` so that `ulysses_size * ring_size` equals
the world size. `--tea_cache_l1_thresh` enables feature caching; start with
`0` for a correctness baseline.

## Training

Training uses a separate `training/wan` package because its model definition
differs from the inference package. Run all commands from the repository root.

First extract audio and landmarks from the raw videos:

```bash
uv run python training/preprocess/landmark_pipeline.py \
  --data_dir /path/to/raw_videos \
  --aux_dir /path/to/aux_data \
  --num_gpus 1
```

Then encode 81-frame clips, audio features, reference features, and latents:

```bash
uv run python training/preprocess/data_preprocess_pipeline.py \
  --data_dir /path/to/raw_videos \
  --aux_dir /path/to/aux_data \
  --save_dir /path/to/processed_data \
  --num_gpus 1
```

Create the shared prompt embedding used by the dataset:

```bash
uv run python training/create_context.py \
  --prompt "A person is talking." \
  --output data/share/context.pt
```

Start training. The initialization audio projection is distributed with the
Jogg model release; a released Jogg checkpoint can also be supplied through
`--resume_ckpt` as a comma-separated shard list.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python training/train.py \
  --data_dir /path/to/processed_data \
  --share_dir data/share \
  --infinitetalk_pred_model_path \
    models/Jogg-Avatar-V2V-1.3B/training_init/audio_proj.safetensors \
  --output_path outputs/train \
  --training_strategy auto
```

Install the `deepspeed` extra and select a DeepSpeed strategy only when that
runtime is required:

```bash
uv sync --extra train --extra deepspeed
```

DeepSpeed checkpoints include a generated `zero_to_fp32.py`. Convert a
checkpoint to sharded safetensors for inference or publication with:

```bash
uv run python /path/to/checkpoint/zero_to_fp32.py \
  /path/to/checkpoint /path/to/export \
  --safe_serialization --max_shard_size 5GB
```

The resulting shard names and index layout match `huggingface/`.

## Responsible Use

Obtain consent for source videos and voices, follow applicable privacy and
publicity laws, and clearly disclose synthetic media. Do not use this project
for impersonation, fraud, harassment, or deceptive content.

## Acknowledgments

This project builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1) and
[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk). The release layout
and tooling follow the Jogg-Avatar and Jogg-Avatar-V2V projects.

## License

The code is released under the [Apache License 2.0](LICENSE). Wan2.1,
InfiniteTalk, Wav2Vec2, face models, model checkpoints, and training data remain
subject to their own licenses and terms.
