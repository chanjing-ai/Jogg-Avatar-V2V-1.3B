# 蝉镜数字人 V2V 1.3B

[English](README.md) | [简体中文](README_zh.md)

蝉镜数字人 V2V 1.3B（Chanjing-Avatar V2V 1.3B）是一个音频驱动的数字人视频生成模型，基于
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)，并参考
[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) 技术路线。模型输入源视频和
驱动音频，保留原视频中的身体、背景与镜头运动，同时重新生成与语音同步的人脸区域。

本开源版本包含 InfiniteTalk 的推理、预处理和训练链路。


https://github.com/user-attachments/assets/ef8ac635-782b-476a-be3a-5522c98dc457

## Chanjing-Avatar 模型家族

- [Chanjing-Avatar 14B](https://github.com/chanjing-ai/Chanjing-Avatar)（[权重](https://huggingface.co/cicada-ai/Chanjing-Avatar-14B)）：输入参考图与驱动音频，生成 720p 数字人视频。
- [Chanjing-Avatar V2V 5B](https://github.com/chanjing-ai/Chanjing-Avatar-V2V-5B)（[权重](https://huggingface.co/cicada-ai/Chanjing-Avatar-V2V-5B)）：保留源视频动作、镜头与背景，重新生成说话人脸。
- [Chanjing-Avatar V2V 1.3B](https://github.com/chanjing-ai/Chanjing-Avatar-V2V-1.3B)（[权重](https://huggingface.co/cicada-ai/Chanjing-Avatar-V2V-1.3B)）：更轻量的音频驱动人脸视频生成模型。

## 环境要求

- Linux、NVIDIA GPU，以及兼容 CUDA 12.8 的驱动
- `PATH` 中可用的 `ffmpeg`
- [`uv`](https://docs.astral.sh/uv/)
- 项目锁定 Python 3.13、PyTorch 2.8.0 和 CUDA 12.8 wheel

```bash
git clone https://github.com/chanjing-ai/Chanjing-Avatar-V2V-1.3B.git
cd Chanjing-Avatar-V2V-1.3B

uv sync
uv sync --extra preprocess
uv sync --extra preprocess --extra train --extra test
```

FlashAttention 是推荐的可选依赖，需要在基础环境同步完成后单独安装：

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## 模型准备

先下载 Wan 基础模型和中文 Wav2Vec2：

```bash
mkdir -p models
uv run hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir models/Wan2.1-T2V-1.3B
uv run hf download TencentGameMate/chinese-wav2vec2-base \
  --local-dir models/chinese-wav2vec2-base

uv run hf download google/umt5-xxl \
  --include "spiece.model" "tokenizer*" "special_tokens_map.json" \
  --local-dir models/Wan2.1-T2V-1.3B/google/umt5-xxl
```

从 Hugging Face 下载 Chanjing-Avatar V2V 1.3B 权重：

```bash
uv run hf download cicada-ai/Chanjing-Avatar-V2V-1.3B \
  --local-dir models/Chanjing-Avatar-V2V-1.3B
```

人脸预处理还需要另外准备 `landmark.onnx` 和 `scrfd_500m_bnkps.onnx`。

默认模型目录如下：

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
|-- Chanjing-Avatar-V2V-1.3B/
|   |-- model-00001-of-00002.safetensors
|   |-- model-00002-of-00002.safetensors
|   `-- training_init/audio_proj.safetensors
|-- landmark.onnx
`-- scrfd_500m_bnkps.onnx
```

通过 `CHANJING_AVATAR_MODEL_DIR` 可以修改预处理和推理使用的模型根目录。模型、训练数据、
checkpoint 和生成结果均不会提交到 Git。

## 推理

按 `examples/inference.json` 准备任务文件。每个源视频只接受
`cond_audio.person1` 中的一条驱动音频。首次推理前运行一次预处理：

```bash
uv run python preprocess_video.py \
  --data_dir /path/to/source_videos \
  --num_gpus 1
```

对于 `/path/to/source.mp4`，预处理会生成 `/path/to/source.npy` 和
`/path/to/source_bbox.json`。加载模型前可以先检查所有路径：

```bash
uv run python inference.py \
  --input_json examples/inference.json \
  --validate_only
```

单卡推理命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python inference.py \
  --input_json examples/inference.json \
  --sample_steps 8 \
  --mode streaming
```

结果写入 `results/`。多卡运行使用 `torchrun`，并保证
`ulysses_size * ring_size` 等于进程总数。首次验证建议保持
`--tea_cache_l1_thresh 0`，确认基线结果后再开启特征缓存。

## 训练

训练版模型定义与推理版不同，因此放在独立的 `training/wan` 包中。以下命令都从仓库
根目录运行。

先为原始视频提取音频和关键点：

```bash
uv run python training/preprocess/landmark_pipeline.py \
  --data_dir /path/to/raw_videos \
  --aux_dir /path/to/aux_data \
  --num_gpus 1
```

再编码 81 帧片段、音频特征、参考图特征和 latent：

```bash
uv run python training/preprocess/data_preprocess_pipeline.py \
  --data_dir /path/to/raw_videos \
  --aux_dir /path/to/aux_data \
  --save_dir /path/to/processed_data \
  --num_gpus 1
```

生成训练数据共用的提示词 embedding：

```bash
uv run python training/create_context.py \
  --prompt "A person is talking." \
  --output data/share/context.pt
```

启动训练。音频投影初始化权重随 Chanjing-Avatar 模型权重发布；也可以通过
`--resume_ckpt` 传入以英文逗号分隔的已发布权重分片。

正式训练前可先检查一个预处理样本和全部 checkpoint 张量形状。该命令不会分配 1.3B
模型参数，也不会启动训练：

```bash
uv run python training/train.py \
  --data_dir /path/to/processed_data \
  --share_dir data/share \
  --resume_ckpt \
    models/Chanjing-Avatar-V2V-1.3B/model-00001-of-00002.safetensors,models/Chanjing-Avatar-V2V-1.3B/model-00002-of-00002.safetensors \
  --validate_only
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python training/train.py \
  --data_dir /path/to/processed_data \
  --share_dir data/share \
  --infinitetalk_pred_model_path \
    models/Chanjing-Avatar-V2V-1.3B/training_init/audio_proj.safetensors \
  --output_path outputs/train \
  --training_strategy auto
```

默认启用 gradient checkpointing。需要把激活卸载到 CPU 时添加
`--use_gradient_checkpointing_offload`；只有显存充足时才使用
`--disable_gradient_checkpointing`。

确实需要 DeepSpeed 时再安装对应 extra：

```bash
uv sync --extra train --extra deepspeed
```

DeepSpeed checkpoint 中会生成 `zero_to_fp32.py`，可用它导出推理或发布所需的分片
safetensors：

```bash
uv run python /path/to/checkpoint/zero_to_fp32.py \
  /path/to/checkpoint /path/to/export \
  --safe_serialization --max_shard_size 5GB
```

导出后的分片名和索引结构与 `huggingface/` 中的模板一致。

## 使用边界

使用视频和声音前应取得授权，遵守隐私权、肖像权等适用规则，并明确标注合成内容。请勿将
本项目用于冒充、欺诈、骚扰或误导性内容。

## 致谢

本项目基于 [Wan2.1](https://github.com/Wan-Video/Wan2.1) 与
[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)。

## 许可证

代码使用 [Apache License 2.0](LICENSE)。Wan2.1、InfiniteTalk、Wav2Vec2、人脸模型、
模型权重和训练数据仍分别受其自身许可证与使用条款约束。
