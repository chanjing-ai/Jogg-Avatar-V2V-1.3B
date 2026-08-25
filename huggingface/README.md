---
license: apache-2.0
base_model: Wan-AI/Wan2.1-T2V-1.3B
pipeline_tag: video-to-video
tags:
  - audio-driven
  - avatar
  - talking-head
  - video-to-video
  - wan2.1
  - infinitetalk
---

# Chanjing-Avatar V2V 1.3B (InfiniteTalk)

Chanjing-Avatar V2V 1.3B is an audio-driven video-to-video avatar model based on
Wan2.1-T2V-1.3B and the InfiniteTalk approach. It preserves source body,
background, and camera motion while regenerating a face synchronized to a
driving audio track.

This model card is prepared for the InfiniteTalk checkpoint release. Training,
preprocessing, and inference code is available at
[chanjing-ai/Chanjing-Avatar-V2V-1.3B](https://github.com/chanjing-ai/Chanjing-Avatar-V2V-1.3B).

The checkpoint directory is expected to contain:

```text
Chanjing-Avatar-V2V-1.3B/
|-- model-00001-of-00002.safetensors
|-- model-00002-of-00002.safetensors
|-- model.safetensors.index.json
`-- training_init/audio_proj.safetensors
```

Wan2.1-T2V-1.3B, TencentGameMate/chinese-wav2vec2-base, and the face
preprocessing models are required separately. Review all upstream licenses and
terms before use.

Users are responsible for obtaining consent for source videos and voices and
for clearly disclosing synthetic media. Do not use this model for impersonation,
fraud, harassment, or deceptive content.
