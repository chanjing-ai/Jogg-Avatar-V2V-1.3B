# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import repo_paths

repo_paths.ensure_project_src_path()

import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

MODEL_ROOT = os.environ.get("CHANJING_AVATAR_MODEL_DIR", "models")

import random

import torch
import torch.distributed as dist
import subprocess

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import str2bool, is_video, split_wav_librosa, composite_face_back_to_video
from wan.utils.multitalk_utils import save_video_ffmpeg
from transformers import Wav2Vec2FeatureExtractor
from audio_analysis.wav2vec2 import Wav2Vec2Model
from wan.utils.segvideo import shot_detect
from release_utils import validate_inference_inputs

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf


def _require_single_cond_audio(input_data):
    ca = input_data.get("cond_audio")
    if not isinstance(ca, dict) or "person1" not in ca:
        raise ValueError("cond_audio 必须为对象且包含 person1 音频路径")
    extra = set(ca.keys()) - {"person1"}
    if extra:
        raise ValueError(
            "cond_audio 仅保留 person1，请移除: " + ", ".join(sorted(extra))
        )


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        if args.size == 'infinitetalk-480':
            args.sample_shift = 7
        elif args.size == 'infinitetalk-720':
            args.sample_shift = 11
        else:
            raise NotImplementedError(f'Not supported size')

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, 99999999)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an audio-driven avatar video with InfiniteTalk 1.3B"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="infinitetalk-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="infinitetalk-480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The size bucket used for generation."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to be generated in one clip. The number should be 4n+1"
    )
    parser.add_argument(
        "--max_frame_num",
        type=int,
        default=1000,
        help="The maximum frame length of the generated video."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.path.join(MODEL_ROOT, "Wan2.1-T2V-1.3B"),
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--infinitetalk_dir",
        type=str,
        default=",".join([
            os.path.join(MODEL_ROOT, "Chanjing-Avatar-V2V-1.3B", "model-00001-of-00002.safetensors"),
            os.path.join(MODEL_ROOT, "Chanjing-Avatar-V2V-1.3B", "model-00002-of-00002.safetensors"),
        ]),
        help="Comma-separated InfiniteTalk safetensors paths.")
    parser.add_argument(
        "--quant_dir",
        type=str,
        default=None,
        help="The path to the Wan quant checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=os.path.join(MODEL_ROOT, "chinese-wav2vec2-base"),
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--lora_dir",
        type=str,
        nargs='+',
        default=None,
        help="The paths to the LoRA checkpoint files."
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        nargs='+',
        default=[1.2],
        help="Controls how much to influence the outputs with the LoRA parameters. Accepts multiple float values."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=True,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Optional output basename; input JSON save_file takes precedence.")
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        default=os.path.join("data", "temp"),
        help="The path to save the audio embedding.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples/inference.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--motion_frame",
        type=int,
        default=9,
        help="Driven frame length used in the mode of long video genration.")
    parser.add_argument(
        "--mode",
        type=str,
        default="streaming",
        choices=['clip', 'streaming'],
        help="clip: generate one video chunk, streaming: long video generation")
    parser.add_argument(
        "--sample_steps", type=int, default=8, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_text_guide_scale",
        type=float,
        # default=3.0,
        default=1.0,
        help="Classifier free guidance scale for text control.")
    parser.add_argument(
        "--sample_audio_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        required=False,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    parser.add_argument(
        "--tea_cache_l1_thresh",
        type=float,
        default=0.,
        help="TeaCache/CustomCache L1 threshold (0=off). Same as wan2.2-5b-v2v inference_5B.yaml. 50-step: 0.1-0.14."
    )
    parser.add_argument(
        "--feature_caching",
        type=str,
        default="Custom",
        choices=["Tea", "Custom"],
        help="Feature caching: Tea=TeaCache, Custom=CustomCache (Tea + Taylor compensation, same as wan2.2)."
    )
    parser.add_argument(
        "--use_apg",
        action="store_true",
        default=False,
        help="Enable adaptive projected guidance for video generation (APG)."
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="Momentum used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55,
        help="Norm threshold used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="strength for color correction [0.0 -- 1.0]."
    )
    parser.add_argument(
        "--scene_seg",
        action="store_true",
        default=False,
        help="Enable scene segmentation for input video."
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Quantization type, must be 'int8' or 'fp8'."
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate model and input paths without loading the model.",
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def custom_init(device, wav2vec):
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0] + '.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array


def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array


def generate(args):
    input_data_list = validate_inference_inputs(args)
    if args.validate_only:
        print("Validation passed.")
        return

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
                args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
                args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    assert args.task == "infinitetalk-1.3B", 'You should choose infinitetalk in args.task.'

    logging.info("Creating infinitetalk-1.3B pipeline.")
    wan_i2v = wan.InfiniteTalkPipeline_v6(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        quant_dir=args.quant_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        lora_dir=args.lora_dir,
        lora_scales=args.lora_scale,
        quant=args.quant,
        dit_path=args.dit_path,
        infinitetalk_dir=args.infinitetalk_dir
    )
    if args.num_persistent_param_in_dit is not None:
        wan_i2v.vram_management = True
        wan_i2v.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )

    base_audio_save_dir = args.audio_save_dir
    default_save_file = args.save_file
    wav2vec_feature_extractor, audio_encoder = custom_init(
        'cpu', args.wav2vec_dir
    )
    for input_data in input_data_list:
        generated_list = []
        args.save_file = input_data.get('save_file') or default_save_file
        if args.save_file is not None and os.path.exists(os.path.join("results", args.save_file + ".mp4")):
            continue
        cond_video_path = input_data['cond_video']
        if not os.path.exists(cond_video_path):
            raise FileNotFoundError(
                f"cond_video 文件不存在: {cond_video_path}\n"
                "请填原视频路径，并先运行: python preprocess_video.py --data_dir '<目录>' 生成同名 .npy 与 _bbox.json"
            )
        if not args.scene_seg:
            bbox_required = os.path.splitext(cond_video_path)[0] + "_bbox.json"
            if not os.path.exists(bbox_required):
                raise FileNotFoundError(
                    f"缺少预处理结果: {bbox_required}\n"
                    "请先对原视频运行: python preprocess_video.py --data_dir '<含该视频的目录>'"
                )
        _require_single_cond_audio(input_data)
        _stem = os.path.splitext(os.path.basename(input_data['cond_video']))[0]
        audio_save_dir = os.path.join(base_audio_save_dir, _stem)
        os.makedirs(audio_save_dir, exist_ok=True)

        person1_path = input_data["cond_audio"]["person1"]
        if args.scene_seg and is_video(input_data['cond_video']):
            time_list, cond_list = shot_detect(input_data['cond_video'], audio_save_dir)
            if len(time_list) == 0:
                clip_jobs = [(input_data['cond_video'], person1_path)]
            else:
                audio_clip_paths = split_wav_librosa(
                    person1_path, time_list, audio_save_dir)
                clip_jobs = list(zip(cond_list, audio_clip_paths))
        else:
            clip_jobs = [(input_data['cond_video'], person1_path)]

        human_speech_full = audio_prepare_single(person1_path)
        sum_audio_all = os.path.join(audio_save_dir, 'sum_all.wav')
        sf.write(sum_audio_all, human_speech_full, 16000)
        input_data['video_audio'] = sum_audio_all
        logging.info("Generating video ...")

        for idx, (clip_video, clip_audio_path) in enumerate(clip_jobs):
            logging.info("Clip %s: cond_video=%s audio=%s", idx, clip_video, clip_audio_path)
            input_clip = {}
            input_clip['prompt'] = input_data['prompt']
            input_clip['cond_video'] = clip_video

            cond_video_path_clip = input_clip['cond_video']
            base = os.path.splitext(cond_video_path_clip)[0]
            bbox_path = base + "_bbox.json"
            if args.scene_seg:
                if os.path.exists(bbox_path):
                    with open(bbox_path, "r", encoding="utf-8") as f:
                        input_clip["bbox_orig"] = json.load(f)
            else:
                with open(bbox_path, "r", encoding="utf-8") as f:
                    input_clip["bbox_orig"] = json.load(f)

            human_speech = audio_prepare_single(clip_audio_path)
            audio_embedding = get_embedding(
                human_speech, wav2vec_feature_extractor, audio_encoder)
            emb_path = os.path.join(audio_save_dir, '1.pt')
            sum_audio = os.path.join(audio_save_dir, 'sum.wav')
            sf.write(sum_audio, human_speech, 16000)
            torch.save(audio_embedding, emb_path)
            input_clip['cond_audio'] = {'person1': emb_path}
            input_clip['video_audio'] = sum_audio
            video = wan_i2v.generate_infinitetalk(
                input_clip,
                size_buckget=args.size,
                motion_frame=args.motion_frame,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sampling_steps=args.sample_steps,
                text_guide_scale=args.sample_text_guide_scale,
                audio_guide_scale=args.sample_audio_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                max_frames_num=args.frame_num if args.mode == 'clip' else args.max_frame_num,
                color_correction_strength=args.color_correction_strength,
                use_target_size=True,
                extra_args=args,
            )

            generated_list.append(video)

        if rank == 0:

            if args.save_file is None:
                # 用输入视频名和音频名组合
                video_basename = os.path.splitext(os.path.basename(input_data["cond_video"]))[0]
                audio_basename = os.path.splitext(os.path.basename(input_data["cond_audio"]["person1"]))[0]
                args.save_file = f"{video_basename}_{audio_basename}"

            os.makedirs("results", exist_ok=True)
            save_path = os.path.join("results", args.save_file)
            sum_video = torch.cat(generated_list, dim=1)
            if input_clip.get("bbox_orig") and len(generated_list) == 1:
                out_mp4 = save_path + ".mp4"
                composite_face_back_to_video(
                    input_clip["cond_video"],
                    sum_video,
                    input_clip["bbox_orig"],
                    out_mp4,
                    [input_data["video_audio"]],
                    fps=25,
                )
            else:
                save_video_ffmpeg(sum_video, save_path, [input_data['video_audio']], high_quality_save=False)

        logging.info(f"Saving generated video to {os.path.join('results', args.save_file)}.mp4")
        logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
