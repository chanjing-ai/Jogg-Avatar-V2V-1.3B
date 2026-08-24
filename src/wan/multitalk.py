# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import hashlib
# from inspect import ArgSpec
import logging
import json
import math
import importlib
import os
import random
import sys
import types
import copy
from contextlib import contextmanager
from functools import partial
from PIL import Image

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from diffusers.models.modeling_utils import no_init_weights, ContextManagers
import accelerate

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.multitalk_model import WanModel, WanLayerNorm, WanRMSNorm
from .cache import TeaCache, CustomCache
from .modules.t5 import T5EncoderModel, T5LayerNorm, T5RelativeEmbedding
from .modules.vae import WanVAE, CausalConv3d, RMS_norm, Upsample
from .utils.multitalk_utils import MomentumBuffer, adaptive_projected_guidance, match_and_blend_colors, match_and_blend_colors_torch
from vram_management import AutoWrappedQLinear, AutoWrappedLinear, AutoWrappedModule, enable_vram_management
from wan.utils.utils import convert_video_to_h264, extract_specific_frames, extract_fragment_video, get_video_codec, resize_fit_letterbox
from wan.wan_lora import WanLoraWrapper

from safetensors.torch import load_file
from optimum.quanto import quantize, freeze, qint8,requantize
import optimum.quanto.nn.qlinear as qlinear

def torch_gc():
    # if not force:
    #     return
    # torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def to_param_dtype_fp32only(model, param_dtype):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32 and param.__class__.__name__ not in ['WeightQBytesTensor']:
                param.data = param.data.to(param_dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype == torch.float32 and buf.__class__.__name__ not in ['WeightQBytesTensor']:
                module._buffers[name] = buf.to(param_dtype)

def resize_and_centercrop(cond_image, target_size):
        """
        Resize image or tensor to the target size without padding.
        """

        # Get the original size
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size

        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)

        # Resize
        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous()
            # crop
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
            resized_image = np.array(resized_image)
            # tensor and crop
            resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor[:, :, None, :, :]

        return cropped_tensor


def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


def build_fms_timesteps(num_steps, device, shift=5.0, sigma_min=0.0):
    sigmas = torch.linspace(1.0, sigma_min, num_steps + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * 1000
    ts = [t.reshape(1).to(device) for t in timesteps]
    ts.append(torch.tensor([0.0], device=device))
    return ts


class InfiniteTalkPipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        quant_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        num_timesteps=1000,
        use_timestep_transform=True,
        lora_dir=None,
        lora_scales=None,
        quant = None,
        dit_path = None,
        infinitetalk_dir=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            quant (`str`, *optional*, defaults to None):
                Quantization type, must be 'int8' or 'fp8'.
        """
        if quant is not None and quant not in ("int8", "fp8"):
            raise ValueError("quant must be 'int8', 'fp8', or None(default fp32 model)")
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)

        # self.text_encoder = T5EncoderModel(
        #     text_len=config.text_len,
        #     dtype=config.t5_dtype,
        #     device=torch.device('cpu'),
        #     checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
        #     tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        #     shard_fn=shard_fn if t5_fsdp else None,
        #     quant=quant,
        #     quant_dir=os.path.dirname(quant_dir) if quant_dir is not None else None,
        # )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        if quant is not None:
            logging.info(f"Loading Quantized MultiTalk from {quant_dir}")
            with torch.device('meta'):
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                # MultiTalk 使用的是 i2v 结构，这里强制覆盖为 i2v，并使用与权重一致的 in_dim
                wan_config["model_type"] = "i2v"
                wan_config["in_dim"] = 36
                self.model = WanModel(weight_init=False, **wan_config)
                torch_gc()
            model_state_dict = load_file(quant_dir)
            map_json_path = os.path.join(quant_dir.replace('safetensors', 'json'))
            self.model.init_freqs()
            with open(map_json_path, "r") as f:
                quantization_map = json.load(f)
            requantize(self.model, model_state_dict, quantization_map, device='cpu')
        else:
            if dit_path is None:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                # MultiTalk backbone 需要 i2v 模式，并且 patch_embedding 的输入通道数为 36（与 ckpt 中权重一致）
                wan_config["model_type"] = "i2v"
                wan_config["in_dim"] = 36
                self.model = WanModel(weight_init=False, **wan_config).to(dtype=self.param_dtype)
                if checkpoint_dir is not None:
                    weight_files = [f"{checkpoint_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                                    f"{checkpoint_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                                    f"{checkpoint_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                                    f"{checkpoint_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                                    f"{checkpoint_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                                    f"{checkpoint_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                                    f"{checkpoint_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                                    f"{infinitetalk_dir}"]
                # else:
                # weight_files = [f"{infinitetalk_dir}"]
                # weight_files = infinitetalk_dir.split(',')

                merged_state_dict = {}
                for weight_file in weight_files:
                    sd = load_file(weight_file)
                    # sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
                    merged_state_dict.update(sd)
                self.model.load_state_dict(merged_state_dict)

            else:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                with ContextManagers(init_contexts):
                    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                    self.model = WanModel(weight_init=False,**wan_config)
                checkpoint_weights = torch.load(dit_path, map_location='cpu')
                self.model.load_state_dict(checkpoint_weights['state_dict'])
                logging.info(f"loading infinitetalk weights {checkpoint_dir}")

        self.model.eval().requires_grad_(False)

        to_param_dtype_fp32only(self.model, self.param_dtype)
        if lora_dir is not None and quant is None :
            lora_wrapper = WanLoraWrapper(self.model)
            for lora_path, lora_scale in zip(lora_dir, lora_scales):
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, lora_scale, param_dtype=self.param_dtype, device=self.device)




        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_dit_forward_multitalk,
                usp_attn_forward_multitalk,
                usp_crossattn_multi_forward_multitalk
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn)
            self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1



        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.cpu_offload = False
        self.model_names = ["model"]
        self.vram_management = False

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.model.parameters())).dtype
        enable_vram_management(
            self.model,
            module_map={
                qlinear.QLinear: AutoWrappedQLinear,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)

                if not isinstance(model, nn.Module):
                    model = model.model

                if model is not None:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if not isinstance(model, nn.Module):
                model = model.model
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()


    def generate_infinitetalk(self,
                 input_data,
                 size_buckget='infinitetalk-480',
                 motion_frame=25,
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=4.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 max_frames_num=1000,
                 face_scale=0.05,
                 progress=True,
                 color_correction_strength=0.0,
                 use_target_size=False,
                 extra_args=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
        """

        # init teacache（v6 仅用 CustomCache，其他 pipeline 用 use_teacache/teacache_thresh）
        if getattr(self, "_is_v6", False):
            self.model.disable_teacache()
        else:
            if extra_args.use_teacache:
                self.model.teacache_init(
                    sample_steps=sampling_steps,
                    teacache_thresh=extra_args.teacache_thresh,
                    model_scale=extra_args.size,
                )
            else:
                self.model.disable_teacache()

        input_prompt = input_data['prompt']
        cond_file_path = input_data['cond_video']
        codec = get_video_codec(cond_file_path)
        if codec == 'av1':
            output_video_path = 'tmp/' + '_input_h264.mp4'
            print(f"Converting {cond_file_path} from AV1 to H.264...")
            convert_video_to_h264(cond_file_path, output_video_path)
            print(f"Conversion complete! Saved as {output_video_path}")
            cond_file_path = output_video_path
        else:
            print("No conversion needed.")
        bbox_orig = input_data.get('bbox_orig')
        cond_image = extract_specific_frames(cond_file_path, 0, bbox=bbox_orig)
        # cond_image = Image.fromarray(cond_image)


        # decide a proper size
        bucket_config_module = importlib.import_module("wan.utils.multitalk_utils")
        if size_buckget == 'infinitetalk-480':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_627')
        elif size_buckget == 'infinitetalk-720':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_960')

        src_h, src_w = cond_image.height, cond_image.width
        ratio = src_h / src_w
        # 在线 bbox 裁剪时：模型输入必须与训练一致，固定为 (512,512)/(448,576)/(576,448) 之一，按 crop 长宽比选取
        if bbox_orig is not None:
            TRAIN_SIZE_CANDIDATES = [(512, 512), (448, 576), (576, 448)]  # (H, W)，与老版 crop 视频流程一致
            target_h, target_w = min(
                TRAIN_SIZE_CANDIDATES,
                key=lambda hw: abs((hw[0] / hw[1]) - ratio)
            )
        elif use_target_size:
            target_h = (src_h // 16) * 16
            target_w = (src_w // 16) * 16
        else:
            closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x)-ratio))[0]
            target_h, target_w = bucket_config[closest_bucket][0]

        if bbox_orig is not None:
            cond_image = resize_fit_letterbox(cond_image, (target_h, target_w))
            cond_image = torch.from_numpy(np.array(cond_image)).permute(2, 0, 1).float()[None, :, None, :, :]
        else:
            cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2 # normalization
        cond_image = cond_image.to(self.device)  # 1 C 1 H W

        # Store the original image for color reference if strength > 0
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()


        # read audio embeddings
        audio_embedding_path_1 = input_data['cond_audio']['person1']
        if len(input_data['cond_audio']) == 1:
            HUMAN_NUMBER = 1
            audio_embedding_path_2 = None
        else:
            HUMAN_NUMBER = 2
            audio_embedding_path_2 = input_data['cond_audio']['person2']


        full_audio_embs = []
        audio_embedding_paths = [audio_embedding_path_1, audio_embedding_path_2]
        for human_idx in range(HUMAN_NUMBER):
            audio_embedding_path = audio_embedding_paths[human_idx]
            if not os.path.exists(audio_embedding_path):
                continue
            full_audio_emb = torch.load(audio_embedding_path)
            if torch.isnan(full_audio_emb).any():
                continue
            if full_audio_emb.shape[0] <= frame_num:
                continue
            full_audio_embs.append(full_audio_emb)

        assert len(full_audio_embs) == HUMAN_NUMBER, f"Aduio file not exists or length not satisfies frame nums."

        # preprocess text embedding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        torch_gc()
        # prepare params for video generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        torch_gc()

        # set random seed and init noise
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # start video generation iteratively
        while True:
            audio_embs = []
            # split audio with window size
            for human_idx in range(HUMAN_NUMBER):
                center_indices = torch.arange(
                    audio_start_idx,
                    audio_end_idx,
                    1,
                ).unsqueeze(
                    1
                ) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
                audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                audio_embs.append(audio_emb)
            audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)
            torch_gc()

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size



            noise = torch.randn(
                16, (frame_num - 1) // 4 + 1,
                lat_h,
                lat_w,
                dtype=torch.float32,
                device=self.device)

            # get mask
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ],
                            dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2).to(self.param_dtype) # B 4 T H W

            with torch.no_grad():
                # get clip embedding
                self.clip.model.to(self.device)
                clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype)
                if offload_model:
                    self.clip.model.cpu()
                torch_gc()

                # zero padding and vae encode
                video_frames = torch.zeros(1, cond_image.shape[1], frame_num-cond_image.shape[2], target_h, target_w).to(self.device)
                padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
                y = self.vae.encode(padding_frames_pixels_values)
                y = torch.stack(y).to(self.param_dtype) # B C T H W
                cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)

                if is_first_clip:
                    latent_motion_frames = self.vae.encode(cond_image)[0]
                else:
                    latent_motion_frames = self.vae.encode(cond_frame)[0]

                y = torch.concat([msk, y], dim=1) # B 4+C T H W
                torch_gc()


            # construct human mask
            human_masks = []
            if HUMAN_NUMBER==1:
                background_mask = torch.ones([src_h, src_w])
                human_mask1 = torch.ones([src_h, src_w])
                human_mask2 = torch.ones([src_h, src_w])
                human_masks = [human_mask1, human_mask2, background_mask]
                # background_mask = torch.zeros([src_h, src_w])
                # x_min, y_min, x_max, y_max = 1286, 556, 2040, 1310
                # human_mask1 = torch.zeros([src_h, src_w])
                # human_mask2 = torch.zeros([src_h, src_w])
                # human_mask1[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                # # human_mask2[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                # background_mask += human_mask1
                # # background_mask += human_mask2
                # background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
                # human_masks = [human_mask1, human_mask2, background_mask]


            elif HUMAN_NUMBER==2:
                if 'bbox' in input_data:
                    assert len(input_data['bbox']) == len(input_data['cond_audio']), f"The number of target bbox should be the same with cond_audio"
                    background_mask = torch.zeros([src_h, src_w])
                    for _, person_bbox in input_data['bbox'].items():
                        x_min, y_min, x_max, y_max = person_bbox
                        human_mask = torch.zeros([src_h, src_w])
                        human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                        background_mask += human_mask
                        human_masks.append(human_mask)
                else:
                    x_min, x_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
                    background_mask = torch.zeros([src_h, src_w])
                    background_mask = torch.zeros([src_h, src_w])
                    human_mask1 = torch.zeros([src_h, src_w])
                    human_mask2 = torch.zeros([src_h, src_w])
                    lefty_min, lefty_max = int((src_w//2) * face_scale), int((src_w//2) * (1 - face_scale))
                    righty_min, righty_max = int((src_w//2) * face_scale + (src_w//2)), int((src_w//2) * (1 - face_scale) + (src_w//2))
                    human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                    human_mask2[x_min:x_max, righty_min:righty_max] = 1
                    background_mask += human_mask1
                    background_mask += human_mask2
                    human_masks = [human_mask1, human_mask2]
                background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
                human_masks.append(background_mask)

            ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
            # resize and centercrop for ref_target_masks
            ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))

            _, _, _,lat_h, lat_w = y.shape
            ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode='nearest').squeeze()
            ref_target_masks = (ref_target_masks > 0)
            ref_target_masks = ref_target_masks.float().to(self.device)

            torch_gc()

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)

            # evaluation mode
            with torch.no_grad(), no_sync():

                # prepare timesteps
                timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
                timesteps.append(0.)
                timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                if self.use_timestep_transform:
                    timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

                # sample videos
                latent = noise

                # prepare condition and uncondition configs
                arg_c = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }


                arg_null_text = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }

                arg_null_audio = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }


                arg_null = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }

                torch_gc()
                if not self.vram_management:
                    self.model.to(self.device)
                else:
                    self.load_models_to_device(["model"])

                # injecting motion frames
                if not is_first_clip:
                    latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                    motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                    add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                    _, T_m, _, _ = add_latent.shape
                    latent[:, :T_m] = add_latent

                # infer with APG
                # refer https://arxiv.org/abs/2410.02416
                if extra_args.use_apg:
                    text_momentumbuffer  = MomentumBuffer(extra_args.apg_momentum)
                    audio_momentumbuffer = MomentumBuffer(extra_args.apg_momentum)


                progress_wrap = partial(tqdm, total=len(timesteps)-1) if progress else (lambda x: x)
                for i in progress_wrap(range(len(timesteps)-1)):
                    timestep = timesteps[i]
                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                    latent_model_input = [latent.to(self.device)]

                    # inference with CFG strategy
                    noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                    torch_gc()

                    if math.isclose(text_guide_scale, 1.0):
                        noise_pred_drop_audio = self.model(
                            latent_model_input, t=timestep, **arg_null_audio)[0]
                        torch_gc()
                    else:
                        noise_pred_drop_text = self.model(
                            latent_model_input, t=timestep, **arg_null_text)[0]
                        torch_gc()
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0]
                        torch_gc()

                    if extra_args.use_apg:
                        # correct update direction
                        if math.isclose(text_guide_scale, 1.0):
                            diff_uncond_audio  = noise_pred_cond - noise_pred_drop_audio
                            noise_pred = noise_pred_cond + (audio_guide_scale - 1)* adaptive_projected_guidance(diff_uncond_audio,
                                                                                            noise_pred_cond,
                                                                                            momentum_buffer=audio_momentumbuffer,
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                        else:
                            diff_uncond_text  = noise_pred_cond - noise_pred_drop_text
                            diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                            noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_text,
                                                                                                                noise_pred_cond,
                                                                                                                momentum_buffer=text_momentumbuffer,
                                                                                                                norm_threshold=extra_args.apg_norm_threshold) \
                                + (audio_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_audio,
                                                                                            noise_pred_cond,
                                                                                            momentum_buffer=audio_momentumbuffer,
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                    else:
                        # vanilla CFG strategy
                        if math.isclose(text_guide_scale, 1.0):
                            noise_pred = noise_pred_drop_audio + audio_guide_scale* (noise_pred_cond - noise_pred_drop_audio)
                        else:
                            noise_pred = noise_pred_uncond + text_guide_scale * (
                                noise_pred_cond - noise_pred_drop_text) + \
                                audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                    noise_pred = -noise_pred

                    # update latent
                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / self.num_timesteps
                    latent = latent + noise_pred * dt[:, None, None, None]

                    # injecting motion frames
                    if not is_first_clip:
                        latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                        motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                        add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                        _, T_m, _, _ = add_latent.shape
                        latent[:, :T_m] = add_latent

                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                    x0 = [latent.to(self.device)]
                    del latent_model_input, timestep

                if offload_model:
                    if not self.vram_management:
                        self.model.cpu()
                torch_gc()

                videos = self.vae.decode(x0)

            # cache generated samples
            videos = torch.stack(videos).cpu() # B C T H W
            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(videos, original_color_reference, color_correction_strength)
            # >>> END OF COLOR CORRECTION STEP <<<

            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # decide whether is done
            if arrive_last_frame: break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frame

            cond_frame = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            audio_start_idx += (frame_num - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            # [优化] cond_image 已在预计算阶段提取，无需重复读取视频

            # Repeat audio emb
            if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(HUMAN_NUMBER):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length   = audio_end_idx - len(full_audio_embs[human_inx]) + 3
                        add_audio_emb = torch.flip(full_audio_embs[human_inx][-1*miss_length:], dims=[0])
                        full_audio_embs[human_inx] = torch.cat([full_audio_embs[human_inx], add_audio_emb], dim=0)
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)


            if max_frames_num <= frame_num: break

            clip_idx += 1
            if offload_model:
                torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

        gen_video_samples = torch.cat(gen_video_list, dim=2)[:, :, :int(max_frames_num)]
        gen_video_samples = gen_video_samples.to(torch.float32)
        if max_frames_num > frame_num and sum(miss_lengths) > 0:
            # split video frames
            # gen_video_samples = gen_video_samples[:, :, :-1*miss_lengths[0]]
            gen_video_samples = gen_video_samples[:, :, :full_audio_emb.shape[0]]

        if dist.is_initialized():
            dist.barrier()

        del noise, latent
        torch_gc()

        return gen_video_samples[0] if self.rank == 0 else None


class InfiniteTalkPipeline_v5:

    def __init__(
        self,
        config,
        checkpoint_dir,
        quant_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        num_timesteps=1000,
        use_timestep_transform=True,
        lora_dir=None,
        lora_scales=None,
        quant = None,
        dit_path = None,
        infinitetalk_dir=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            quant (`str`, *optional*, defaults to None):
                Quantization type, must be 'int8' or 'fp8'.
        """
        if quant is not None and quant not in ("int8", "fp8"):
            raise ValueError("quant must be 'int8', 'fp8', or None(default fp32 model)")
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
            quant=quant,
            quant_dir=os.path.dirname(quant_dir) if quant_dir is not None else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        if quant is not None:
            logging.info(f"Loading Quantized MultiTalk from {quant_dir}")
            with torch.device('meta'):
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                # MultiTalk 使用的是 i2v 结构，这里强制覆盖为 i2v，并使用与权重一致的 in_dim
                wan_config["model_type"] = "i2v"
                wan_config["in_dim"] = 36
                self.model = WanModel(weight_init=False, **wan_config)
                torch_gc()
            model_state_dict = load_file(quant_dir)
            map_json_path = os.path.join(quant_dir.replace('safetensors', 'json'))
            self.model.init_freqs()
            with open(map_json_path, "r") as f:
                quantization_map = json.load(f)
            requantize(self.model, model_state_dict, quantization_map, device='cpu')
        else:
            if dit_path is None:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                # MultiTalk backbone 需要 i2v 模式，并且 patch_embedding 的输入通道数为 36（与 ckpt 中权重一致）
                wan_config["model_type"] = "i2v"
                wan_config["in_dim"] = 36
                self.model = WanModel(weight_init=False, **wan_config).to(dtype=self.param_dtype)
                # if checkpoint_dir is not None:
                #     weight_files = [f"{checkpoint_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                #                     f"{infinitetalk_dir}"]
                # else:
                # weight_files = [f"{infinitetalk_dir}"]
                weight_files = infinitetalk_dir.split(',')

                merged_state_dict = {}
                for weight_file in weight_files:
                    sd = load_file(weight_file)
                    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
                    merged_state_dict.update(sd)
                self.model.load_state_dict(merged_state_dict)

            else:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                with ContextManagers(init_contexts):
                    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                    self.model = WanModel(weight_init=False,**wan_config)
                checkpoint_weights = torch.load(dit_path, map_location='cpu')
                self.model.load_state_dict(checkpoint_weights['state_dict'])
                logging.info(f"loading infinitetalk weights {checkpoint_dir}")

        self.model.eval().requires_grad_(False)

        to_param_dtype_fp32only(self.model, self.param_dtype)
        if lora_dir is not None and quant is None :
            lora_wrapper = WanLoraWrapper(self.model)
            for lora_path, lora_scale in zip(lora_dir, lora_scales):
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, lora_scale, param_dtype=self.param_dtype, device=self.device)




        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_dit_forward_multitalk,
                usp_attn_forward_multitalk,
                usp_crossattn_multi_forward_multitalk
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn)
            self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1



        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.cpu_offload = False
        self.model_names = ["model"]
        self.vram_management = False

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.model.parameters())).dtype
        enable_vram_management(
            self.model,
            module_map={
                qlinear.QLinear: AutoWrappedQLinear,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)

                if not isinstance(model, nn.Module):
                    model = model.model

                if model is not None:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if not isinstance(model, nn.Module):
                model = model.model
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()


    def generate_infinitetalk(self,
                 input_data,
                 size_buckget='infinitetalk-480',
                 motion_frame=25,
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=4.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 max_frames_num=1000,
                 face_scale=0.05,
                 progress=True,
                 color_correction_strength=0.0,
                 use_target_size=False,
                 extra_args=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
        """

        # init teacache（v6 仅用 CustomCache，其他 pipeline 用 use_teacache/teacache_thresh）
        if getattr(self, "_is_v6", False):
            self.model.disable_teacache()
        else:
            if extra_args.use_teacache:
                self.model.teacache_init(
                    sample_steps=sampling_steps,
                    teacache_thresh=extra_args.teacache_thresh,
                    model_scale=extra_args.size,
                )
            else:
                self.model.disable_teacache()

        input_prompt = input_data['prompt']
        cond_file_path = input_data['cond_video']
        codec = get_video_codec(cond_file_path)
        if codec == 'av1':
            output_video_path = 'tmp/' + '_input_h264.mp4'
            print(f"Converting {cond_file_path} from AV1 to H.264...")
            convert_video_to_h264(cond_file_path, output_video_path)
            print(f"Conversion complete! Saved as {output_video_path}")
            cond_file_path = output_video_path
        else:
            print("No conversion needed.")
        bbox_orig = input_data.get('bbox_orig')
        cond_image = extract_specific_frames(cond_file_path, 0, bbox=bbox_orig)
        # cond_image = Image.fromarray(cond_image)


        # decide a proper size
        bucket_config_module = importlib.import_module("wan.utils.multitalk_utils")
        if size_buckget == 'infinitetalk-480':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_627')
        elif size_buckget == 'infinitetalk-720':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_960')

        src_h, src_w = cond_image.height, cond_image.width
        ratio = src_h / src_w
        # 在线 bbox 裁剪时：模型输入必须与训练一致，固定为 (512,512)/(448,576)/(576,448) 之一，按 crop 长宽比选取
        if bbox_orig is not None:
            TRAIN_SIZE_CANDIDATES = [(512, 512), (448, 576), (576, 448)]  # (H, W)，与老版 crop 视频流程一致
            target_h, target_w = min(
                TRAIN_SIZE_CANDIDATES,
                key=lambda hw: abs((hw[0] / hw[1]) - ratio)
            )
        elif use_target_size:
            target_h = (src_h // 16) * 16
            target_w = (src_w // 16) * 16
        else:
            closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x)-ratio))[0]
            target_h, target_w = bucket_config[closest_bucket][0]

        if bbox_orig is not None:
            cond_image = resize_fit_letterbox(cond_image, (target_h, target_w))
            cond_image = torch.from_numpy(np.array(cond_image)).permute(2, 0, 1).float()[None, :, None, :, :]
        else:
            cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2 # normalization
        cond_image = cond_image.to(self.device)  # 1 C 1 H W

        # Store the original image for color reference if strength > 0
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()


        # read audio embeddings
        audio_embedding_path_1 = input_data['cond_audio']['person1']
        if len(input_data['cond_audio']) == 1:
            HUMAN_NUMBER = 1
            audio_embedding_path_2 = None
        else:
            HUMAN_NUMBER = 2
            audio_embedding_path_2 = input_data['cond_audio']['person2']


        full_audio_embs = []
        audio_embedding_paths = [audio_embedding_path_1, audio_embedding_path_2]
        for human_idx in range(HUMAN_NUMBER):
            audio_embedding_path = audio_embedding_paths[human_idx]
            if not os.path.exists(audio_embedding_path):
                continue
            full_audio_emb = torch.load(audio_embedding_path)
            if torch.isnan(full_audio_emb).any():
                continue
            if full_audio_emb.shape[0] <= frame_num:
                continue
            full_audio_embs.append(full_audio_emb)

        assert len(full_audio_embs) == HUMAN_NUMBER, f"Aduio file not exists or length not satisfies frame nums."

        # preprocess text embedding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        torch_gc()
        # prepare params for video generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        torch_gc()

        # set random seed and init noise
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # start video generation iteratively
        while True:
            audio_embs = []
            # split audio with window size
            for human_idx in range(HUMAN_NUMBER):
                center_indices = torch.arange(
                    audio_start_idx,
                    audio_end_idx,
                    1,
                ).unsqueeze(
                    1
                ) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
                audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                audio_embs.append(audio_emb)
            audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)
            torch_gc()

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

            noise = torch.randn(
                16, (frame_num - 1) // 4 + 1,
                lat_h-6,
                lat_w-6,
                dtype=torch.float32,
                device=self.device)

            # get mask
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ],
                            dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2).to(self.param_dtype) # B 4 T H W

            with torch.no_grad():
                # get clip embedding
                self.clip.model.to(self.device)
                clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype)
                if offload_model:
                    self.clip.model.cpu()
                torch_gc()

                # zero padding and vae encode
                video_frames = torch.zeros(1, cond_image.shape[1], frame_num-cond_image.shape[2], target_h, target_w).to(self.device)
                padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
                y = self.vae.encode(padding_frames_pixels_values)
                y = torch.stack(y).to(self.param_dtype) # B C T H W
                cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)

                clip_frames = extract_fragment_video(cond_file_path, audio_start_idx, audio_end_idx, (target_h, target_w), bbox=bbox_orig)
                clip_frames = clip_frames.to(dtype=torch.float32, device=self.device)
                clip_frames = clip_frames.unsqueeze(0)

                if is_first_clip:
                    latent_motion_frames = self.vae.encode(cond_image)[0]
                    clip_frames[:, :, :cur_motion_frames_num] = cond_image
                else:
                    if cond_frame.shape[-2:] != (target_h, target_w):
                        # 先对齐尺寸，再 encode，保证 latent_motion_frames 与 latents 空间一致
                        if cond_frame.dim() == 5:
                            n, c, t, h, w = cond_frame.shape
                            cond_frame = cond_frame.reshape(n * t, c, h, w)
                            cond_frame = F.interpolate(cond_frame, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            cond_frame = cond_frame.reshape(n, c, t, target_h, target_w)
                        else:
                            cond_frame = F.interpolate(cond_frame, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    latent_motion_frames = self.vae.encode(cond_frame)[0]
                    clip_frames[:, :, :cur_motion_frames_num] = cond_frame
                latents = self.vae.encode(clip_frames)[0]
                latents[:, cur_motion_frames_latent_num:, 3:-3, 3:-3] = noise[:, cur_motion_frames_latent_num:]
                y = torch.concat([msk, y], dim=1) # B 4+C T H W
                torch_gc()


            # construct human mask
            human_masks = []
            if HUMAN_NUMBER==1:
                background_mask = torch.ones([src_h, src_w])
                human_mask1 = torch.ones([src_h, src_w])
                human_mask2 = torch.ones([src_h, src_w])
                human_masks = [human_mask1, human_mask2, background_mask]
            elif HUMAN_NUMBER==2:
                if 'bbox' in input_data:
                    assert len(input_data['bbox']) == len(input_data['cond_audio']), f"The number of target bbox should be the same with cond_audio"
                    background_mask = torch.zeros([src_h, src_w])
                    for _, person_bbox in input_data['bbox'].items():
                        x_min, y_min, x_max, y_max = person_bbox
                        human_mask = torch.zeros([src_h, src_w])
                        human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                        background_mask += human_mask
                        human_masks.append(human_mask)
                else:
                    x_min, x_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
                    background_mask = torch.zeros([src_h, src_w])
                    background_mask = torch.zeros([src_h, src_w])
                    human_mask1 = torch.zeros([src_h, src_w])
                    human_mask2 = torch.zeros([src_h, src_w])
                    lefty_min, lefty_max = int((src_w//2) * face_scale), int((src_w//2) * (1 - face_scale))
                    righty_min, righty_max = int((src_w//2) * face_scale + (src_w//2)), int((src_w//2) * (1 - face_scale) + (src_w//2))
                    human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                    human_mask2[x_min:x_max, righty_min:righty_max] = 1
                    background_mask += human_mask1
                    background_mask += human_mask2
                    human_masks = [human_mask1, human_mask2]
                background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
                human_masks.append(background_mask)

            ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
            # resize and centercrop for ref_target_masks
            ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))

            _, _, _,lat_h, lat_w = y.shape
            ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode='nearest').squeeze()
            ref_target_masks = (ref_target_masks > 0)
            ref_target_masks = ref_target_masks.float().to(self.device)

            torch_gc()

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)

            # evaluation mode
            with torch.no_grad(), no_sync():

                # prepare timesteps
                timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
                timesteps.append(0.)
                timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                if self.use_timestep_transform:
                    timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

                # sample videos
                latent = latents

                # prepare condition and uncondition configs
                arg_c = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }


                arg_null_text = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }

                arg_null_audio = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }


                arg_null = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }

                torch_gc()
                if not self.vram_management:
                    self.model.to(self.device)
                else:
                    self.load_models_to_device(["model"])

                # # injecting motion frames
                # if not is_first_clip:
                #     latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                #     motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                #     add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                #     _, T_m, _, _ = add_latent.shape
                #     latent[:, :T_m] = add_latent

                # infer with APG
                # refer https://arxiv.org/abs/2410.02416
                if extra_args.use_apg:
                    text_momentumbuffer  = MomentumBuffer(extra_args.apg_momentum)
                    audio_momentumbuffer = MomentumBuffer(extra_args.apg_momentum)


                progress_wrap = partial(tqdm, total=len(timesteps)-1) if progress else (lambda x: x)
                for i in progress_wrap(range(len(timesteps)-1)):
                    timestep = timesteps[i]
                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                    latent_model_input = [latent.to(self.device)]

                    # inference with CFG strategy
                    noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                    torch_gc()

                    if math.isclose(text_guide_scale, 1.0):
                        noise_pred_drop_audio = self.model(
                            latent_model_input, t=timestep, **arg_null_audio)[0]
                        torch_gc()
                    else:
                        noise_pred_drop_text = self.model(
                            latent_model_input, t=timestep, **arg_null_text)[0]
                        torch_gc()
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0]
                        torch_gc()

                    if extra_args.use_apg:
                        # correct update direction
                        if math.isclose(text_guide_scale, 1.0):
                            diff_uncond_audio  = noise_pred_cond - noise_pred_drop_audio
                            noise_pred = noise_pred_cond + (audio_guide_scale - 1)* adaptive_projected_guidance(diff_uncond_audio,
                                                                                            noise_pred_cond,
                                                                                            momentum_buffer=audio_momentumbuffer,
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                        else:
                            diff_uncond_text  = noise_pred_cond - noise_pred_drop_text
                            diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                            noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_text,
                                                                                                                noise_pred_cond,
                                                                                                                momentum_buffer=text_momentumbuffer,
                                                                                                                norm_threshold=extra_args.apg_norm_threshold) \
                                + (audio_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_audio,
                                                                                            noise_pred_cond,
                                                                                            momentum_buffer=audio_momentumbuffer,
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                    else:
                        # vanilla CFG strategy
                        if math.isclose(text_guide_scale, 1.0):
                            noise_pred = noise_pred_drop_audio + audio_guide_scale* (noise_pred_cond - noise_pred_drop_audio)
                        else:
                            noise_pred = noise_pred_uncond + text_guide_scale * (
                                noise_pred_cond - noise_pred_drop_text) + \
                                audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                    noise_pred = -noise_pred

                    # update latent
                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / self.num_timesteps
                    latent = latent + noise_pred * dt[:, None, None, None]

                    # # injecting motion frames
                    # if not is_first_clip:
                    #     latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                    #     motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                    #     add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                    #     _, T_m, _, _ = add_latent.shape
                    #     latent[:, :T_m] = add_latent

                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                    x0 = [latent.to(self.device)]
                    del latent_model_input, timestep

                if offload_model:
                    if not self.vram_management:
                        self.model.cpu()
                torch_gc()

                videos = self.vae.decode(x0)

            # cache generated samples
            videos = torch.stack(videos).cpu() # B C T H W
            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(videos, original_color_reference, color_correction_strength)
            # >>> END OF COLOR CORRECTION STEP <<<

            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # decide whether is done
            if arrive_last_frame: break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frame

            cond_frame = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            audio_start_idx += (frame_num - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            # [优化] cond_image 已在预计算阶段提取，无需重复读取视频

            # Repeat audio emb
            if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(HUMAN_NUMBER):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length   = audio_end_idx - len(full_audio_embs[human_inx]) + 3
                        add_audio_emb = torch.flip(full_audio_embs[human_inx][-1*miss_length:], dims=[0])
                        full_audio_embs[human_inx] = torch.cat([full_audio_embs[human_inx], add_audio_emb], dim=0)
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)


            if max_frames_num <= frame_num: break

            clip_idx += 1
            if offload_model:
                torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

        gen_video_samples = torch.cat(gen_video_list, dim=2)[:, :, :int(max_frames_num)]
        gen_video_samples = gen_video_samples.to(torch.float32)
        if max_frames_num > frame_num and sum(miss_lengths) > 0:
            # split video frames
            # gen_video_samples = gen_video_samples[:, :, :-1*miss_lengths[0]]
            gen_video_samples = gen_video_samples[:, :, :full_audio_emb.shape[0]]

        if dist.is_initialized():
            dist.barrier()

        del noise, latent
        torch_gc()

        return gen_video_samples[0] if self.rank == 0 else None


class InfiniteTalkPipeline_v6:

    def __init__(
        self,
        config,
        checkpoint_dir,
        quant_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=False,
        num_timesteps=1000,
        use_timestep_transform=True,
        lora_dir=None,
        lora_scales=None,
        quant = None,
        dit_path = None,
        infinitetalk_dir=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            quant (`str`, *optional*, defaults to None):
                Quantization type, must be 'int8' or 'fp8'.
        """
        if quant is not None and quant not in ("int8", "fp8"):
            raise ValueError("quant must be 'int8', 'fp8', or None(default fp32 model)")
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self._is_v6 = True

        shard_fn = partial(shard_model, device_id=device_id)

        self.text_encoder = None
        self._text_encoder_kwargs = {
            "text_len": config.text_len,
            "dtype": config.t5_dtype,
            "device": torch.device("cpu"),
            "checkpoint_path": os.path.join(
                checkpoint_dir, config.t5_checkpoint
            ),
            "tokenizer_path": os.path.join(
                checkpoint_dir, config.t5_tokenizer
            ),
            "shard_fn": shard_fn if t5_fsdp else None,
            "quant": quant,
            "quant_dir": (
                os.path.dirname(quant_dir) if quant_dir is not None else None
            ),
        }

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        if quant is not None:
            logging.info(f"Loading Quantized MultiTalk from {quant_dir}")
            with torch.device('meta'):
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                # MultiTalk 使用 i2v 结构，这里在不改动原始配置文件的前提下，强制覆盖为 i2v，并与权重保持一致的 in_dim
                wan_config["model_type"] = "i2v"
                wan_config["in_dim"] = 36
                self.model = WanModel(weight_init=False, **wan_config)
                torch_gc()
            model_state_dict = load_file(quant_dir)
            map_json_path = os.path.join(quant_dir.replace('safetensors', 'json'))
            self.model.init_freqs()
            with open(map_json_path, "r") as f:
                quantization_map = json.load(f)
            requantize(self.model, model_state_dict, quantization_map, device='cpu')
        else:
            if dit_path is None:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                # 同上：MultiTalk backbone 需要 i2v 模式，并且 patch_embedding 的输入通道数为 36（与 ckpt 一致）
                wan_config["model_type"] = "i2v"
                wan_config["in_dim"] = 36
                self.model = WanModel(weight_init=False, **wan_config).to(dtype=self.param_dtype)
                # if checkpoint_dir is not None:
                #     weight_files = [f"{checkpoint_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                #                     f"{checkpoint_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                #                     f"{infinitetalk_dir}"]
                # else:
                # weight_files = [f"{infinitetalk_dir}"]
                weight_files = infinitetalk_dir.split(',')

                merged_state_dict = {}
                for weight_file in weight_files:
                    sd = load_file(weight_file)
                    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
                    merged_state_dict.update(sd)
                self.model.load_state_dict(merged_state_dict)

            else:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                with ContextManagers(init_contexts):
                    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                    self.model = WanModel(weight_init=False,**wan_config)
                checkpoint_weights = torch.load(dit_path, map_location='cpu')
                self.model.load_state_dict(checkpoint_weights['state_dict'])
                logging.info(f"loading infinitetalk weights {checkpoint_dir}")

        self.model.eval().requires_grad_(False)

        to_param_dtype_fp32only(self.model, self.param_dtype)
        if lora_dir is not None and quant is None :
            lora_wrapper = WanLoraWrapper(self.model)
            for lora_path, lora_scale in zip(lora_dir, lora_scales):
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, lora_scale, param_dtype=self.param_dtype, device=self.device)




        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_dit_forward_multitalk,
                usp_attn_forward_multitalk,
                usp_crossattn_multi_forward_multitalk
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn)
            self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1



        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.cpu_offload = False
        self.model_names = ["model"]
        self.vram_management = False

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.model.parameters())).dtype
        enable_vram_management(
            self.model,
            module_map={
                qlinear.QLinear: AutoWrappedQLinear,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)

                if not isinstance(model, nn.Module):
                    model = model.model

                if model is not None:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if not isinstance(model, nn.Module):
                model = model.model
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def segment_bbox_robust(
        self,
        landmarks_seg: np.ndarray,
        q_low: float = 0.03,
        q_high: float = 0.97):
        """基于 quantile 计算鲁棒的人脸框"""
        xs = landmarks_seg[:, :, 0].reshape(-1)
        ys = landmarks_seg[:, :, 1].reshape(-1)

        xmin = np.quantile(xs, q_low)
        xmax = np.quantile(xs, q_high)
        ymin = np.quantile(ys, q_low)
        ymax = np.quantile(ys, q_high)

        return float(xmin), float(ymin), float(xmax), float(ymax)

    def temporal_smooth(self, landmarks: np.ndarray, win: int = 5) -> np.ndarray:
        """时序平滑，landmarks: (T, N, 2)"""
        kernel = np.ones(win) / win
        out = landmarks.copy()
        for i in range(landmarks.shape[1]):
            for d in range(2):
                out[:, i, d] = np.convolve(landmarks[:, i, d], kernel, mode="same")
        return out

    def get_target_mask(self, lmk_path, target_height, target_width):
        """
        基于所有帧的人脸关键点，计算一个全局人脸框(global face box)，再缩小到latent尺度(//8)，
        构造 human1/human2/background 三张mask，并返回 ref_target_masks = stack([..], dim=0)
        """

        lmk203_list = np.load(lmk_path)

        # 1) 提取人脸关键点 (114-137, 202)，构建 landmarks_seg: (T, 25, 2)
        face_indices = list(range(114, 138)) + [202]
        landmarks_seg = np.array([
            [lmk203_list[i][j] for j in face_indices]
            for i in range(len(lmk203_list))
        ], dtype=np.float64)

        # 2) 时序平滑 + quantile 计算全局 bbox
        landmarks_smoothed = self.temporal_smooth(landmarks_seg, win=5)
        g_min_x, g_min_y, g_max_x, g_max_y = self.segment_bbox_robust(landmarks_smoothed)

        # 3) 应用扩展
        width = g_max_x - g_min_x
        height = g_max_y - g_min_y
        width_extension = width * 0.125
        height_extension = height * 0.375
        g_min_x = g_min_x - width_extension
        g_max_x = g_max_x + width_extension
        g_min_y = g_min_y - height_extension
        g_max_y = g_max_y + height_extension

        # 边界保护：确保扩展后的 bbox 不超出目标尺寸（支持小 bbox 场景）
        g_min_x = max(0.0, min(g_min_x, float(target_width)))
        g_max_x = max(0.0, min(g_max_x, float(target_width)))
        g_min_y = max(0.0, min(g_min_y, float(target_height)))
        g_max_y = max(0.0, min(g_max_y, float(target_height)))

        # 4) 缩小到 latent 尺度 (//8) 并做边界保护
        lat_h = target_height // 8
        lat_w = target_width // 8

        x0 = max(0, min(int(g_min_x) // 8, lat_w - 1))
        y0 = max(0, min(int(g_min_y) // 8, lat_h - 1))
        x1 = max(0, min(int(g_max_x) // 8, lat_w - 1))
        y1 = max(0, min(int(g_max_y) // 8, lat_h - 1))

        # 防止出现空框/反向
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        min_padding_len = x0 - 0
        min_padding_len = min(min_padding_len, y0 - 0)
        min_padding_len = min(min_padding_len, lat_w - x1)
        min_padding_len = min(min_padding_len, lat_h - y1)
        min_padding_len = max(min_padding_len, 2)
        min_padding_len = min(min_padding_len, 3)

        # 3) 构造三个二维mask
        human1_mask = torch.zeros((lat_h, lat_w), dtype=torch.float32)
        human1_mask[y0:y1 + 1, x0:x1 + 1] = 1.0

        human2_mask = torch.zeros((lat_h, lat_w), dtype=torch.float32)  # 全0不改

        background_mask = 1.0 - human1_mask  # 与human1_mask刚好相反

        human_masks = [human1_mask, human2_mask, background_mask]
        ref_target_masks = torch.stack(human_masks, dim=0)  # [3, lat_h, lat_w]
        return min_padding_len, ref_target_masks

    def generate_infinitetalk(self,
                 input_data,
                 size_buckget='infinitetalk-480',
                 motion_frame=25,
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=4.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 max_frames_num=1000,
                 face_scale=0.05,
                 progress=True,
                 color_correction_strength=0.0,
                 use_target_size=False,
                 use_fms_schedule=False,
                 extra_args=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
        """

        # init teacache（v6 仅用 CustomCache，其他 pipeline 用 use_teacache/teacache_thresh）
        if getattr(self, "_is_v6", False):
            self.model.disable_teacache()
        else:
            if extra_args.use_teacache:
                self.model.teacache_init(
                    sample_steps=sampling_steps,
                    teacache_thresh=extra_args.teacache_thresh,
                    model_scale=extra_args.size,
                )
            else:
                self.model.disable_teacache()

        input_prompt = input_data['prompt']
        cond_file_path = input_data['cond_video']
        codec = get_video_codec(cond_file_path)
        if codec == 'av1':
            output_video_path = 'tmp/' + '_input_h264.mp4'
            print(f"Converting {cond_file_path} from AV1 to H.264...")
            convert_video_to_h264(cond_file_path, output_video_path)
            print(f"Conversion complete! Saved as {output_video_path}")
            cond_file_path = output_video_path
        else:
            print("No conversion needed.")
        bbox_orig = input_data.get('bbox_orig')
        cond_image = extract_specific_frames(cond_file_path, 0, bbox=bbox_orig)
        # cond_image = Image.fromarray(cond_image)


        # decide a proper size
        bucket_config_module = importlib.import_module("wan.utils.multitalk_utils")
        if size_buckget == 'infinitetalk-480':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_627')
        elif size_buckget == 'infinitetalk-720':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_960')

        src_h, src_w = cond_image.height, cond_image.width
        ratio = src_h / src_w
        # 在线 bbox 裁剪时：模型输入必须与训练一致，固定为 (512,512)/(448,576)/(576,448) 之一，按 crop 长宽比选取
        if bbox_orig is not None:
            TRAIN_SIZE_CANDIDATES = [(512, 512), (448, 576), (576, 448)]  # (H, W)，与老版 crop 视频流程一致
            target_h, target_w = min(
                TRAIN_SIZE_CANDIDATES,
                key=lambda hw: abs((hw[0] / hw[1]) - ratio)
            )
        elif use_target_size:
            target_h = (src_h // 16) * 16
            target_w = (src_w // 16) * 16
        else:
            closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x)-ratio))[0]
            target_h, target_w = bucket_config[closest_bucket][0]

        if bbox_orig is not None:
            cond_image = resize_fit_letterbox(cond_image, (target_h, target_w))
            cond_image = torch.from_numpy(np.array(cond_image)).permute(2, 0, 1).float()[None, :, None, :, :]
        else:
            cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2 # normalization
        cond_image = cond_image.to(self.device)  # 1 C 1 H W

        # Store the original image for color reference if strength > 0
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()


        # read audio embeddings
        audio_embedding_path_1 = input_data['cond_audio']['person1']
        if len(input_data['cond_audio']) == 1:
            HUMAN_NUMBER = 1
            audio_embedding_path_2 = None
        else:
            HUMAN_NUMBER = 2
            audio_embedding_path_2 = input_data['cond_audio']['person2']


        full_audio_embs = []
        audio_embedding_paths = [audio_embedding_path_1, audio_embedding_path_2]
        for human_idx in range(HUMAN_NUMBER):
            audio_embedding_path = audio_embedding_paths[human_idx]
            if not os.path.exists(audio_embedding_path):
                continue
            full_audio_emb = torch.load(audio_embedding_path)
            if torch.isnan(full_audio_emb).any():
                continue
            if full_audio_emb.shape[0] <= frame_num:
                continue
            full_audio_embs.append(full_audio_emb)

        assert len(full_audio_embs) == HUMAN_NUMBER, f"Aduio file not exists or length not satisfies frame nums."

        # preprocess text embedding：优先使用本地缓存，不存在时在线跑 T5 并保存
        _runtime_temp = os.path.join("data", "temp")
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        _context_key = hashlib.sha256(
            json.dumps([input_prompt, n_prompt], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        context_path = os.path.join(_runtime_temp, f"context_{_context_key}.pt")
        context_null_path = os.path.join(
            _runtime_temp, f"context_null_{_context_key}.pt"
        )
        if os.path.exists(context_path) and os.path.exists(context_null_path):
            context = torch.load(context_path).to(self.device)
            context_null = torch.load(context_null_path).to(self.device)
        else:
            if self.text_encoder is None:
                self.text_encoder = T5EncoderModel(**self._text_encoder_kwargs)
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder(
                    [input_prompt], torch.device('cpu')
                )[0].to(self.device)
                context_null = self.text_encoder(
                    [n_prompt], torch.device('cpu')
                )[0].to(self.device)
            os.makedirs(_runtime_temp, exist_ok=True)
            torch.save(context, context_path)
            torch.save(context_null, context_null_path)

        # prepare params for video generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []

        # set random seed and init noise
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # ===== [优化] 预计算跨 clip 不变的量（msk, human mask, ref_target_masks）=====
        h, w = cond_image.shape[-2], cond_image.shape[-1]
        # Wan VAE 空间下采样为像素 //8（与 get_target_mask 中 //8 一致）；config.vae_stride[1:3]=16
        # 表示 patch 后的等效步长，不能直接用于 msk / VAE latent 的 H×W。
        vae_lat_h, vae_lat_w = h // 8, w // 8

        # 预计算 human mask（每个 clip 完全相同）
        human_masks = []
        if HUMAN_NUMBER == 1:
            lmk_path = os.path.splitext(cond_file_path)[0] + '.npy'
            if bbox_orig is not None:
                import tempfile
                crop_w = bbox_orig['x_max'] - bbox_orig['x_min']
                crop_h = bbox_orig['y_max'] - bbox_orig['y_min']
                lmk_crop = np.load(lmk_path)
                # 修复：裁剪超出边界的关键点坐标，防止缩放后超出目标尺寸
                lmk_crop[..., 0] = np.clip(lmk_crop[..., 0], 0, crop_w)
                lmk_crop[..., 1] = np.clip(lmk_crop[..., 1], 0, crop_h)
                scale_x = target_w / max(crop_w, 1)
                scale_y = target_h / max(crop_h, 1)
                lmk_scaled = lmk_crop.copy()
                lmk_scaled[..., 0] *= scale_x
                lmk_scaled[..., 1] *= scale_y
                # 再次裁剪，确保缩放后不超出目标尺寸
                lmk_scaled[..., 0] = np.clip(lmk_scaled[..., 0], 0, target_w)
                lmk_scaled[..., 1] = np.clip(lmk_scaled[..., 1], 0, target_h)
                fd, tmp_path = tempfile.mkstemp(suffix='.npy')
                os.close(fd)
                np.save(tmp_path, lmk_scaled)
                try:
                    min_padding_len, ref_target_masks = self.get_target_mask(tmp_path, target_h, target_w)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            else:
                min_padding_len, ref_target_masks = self.get_target_mask(lmk_path, src_h, src_w)
        elif HUMAN_NUMBER == 2:
            if 'bbox' in input_data:
                assert len(input_data['bbox']) == len(input_data['cond_audio']), \
                    f"The number of target bbox should be the same with cond_audio"
                background_mask = torch.zeros([src_h, src_w])
                for _, person_bbox in input_data['bbox'].items():
                    x_min, y_min, x_max, y_max = person_bbox
                    human_mask = torch.zeros([src_h, src_w])
                    human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                    background_mask += human_mask
                    human_masks.append(human_mask)
            else:
                x_min, x_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
                background_mask = torch.zeros([src_h, src_w])
                background_mask = torch.zeros([src_h, src_w])
                human_mask1 = torch.zeros([src_h, src_w])
                human_mask2 = torch.zeros([src_h, src_w])
                lefty_min, lefty_max = int((src_w // 2) * face_scale), int((src_w // 2) * (1 - face_scale))
                righty_min, righty_max = int((src_w // 2) * face_scale + (src_w // 2)), int((src_w // 2) * (1 - face_scale) + (src_w // 2))
                human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                human_mask2[x_min:x_max, righty_min:righty_max] = 1
                background_mask += human_mask1
                background_mask += human_mask2
                human_masks = [human_mask1, human_mask2]
            background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
            human_masks.append(background_mask)
            ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
            ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))

        ref_target_masks = ref_target_masks.float()
        ref_target_masks = F.interpolate(
            ref_target_masks.unsqueeze(0),
            size=(vae_lat_h, vae_lat_w),
            mode='nearest',
        ).squeeze(0)
        ref_target_masks = (ref_target_masks > 0).float().to(self.device)

        lat_h, lat_w = vae_lat_h, vae_lat_w
        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # 预计算 mask（每个 clip 完全相同；空间尺寸与 VAE latent 一致）
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2).to(self.param_dtype)  # B 4 T H W

        # ===== [优化] 预计算所有 clip 的 CLIP 特征 + VAE y + clip_frames =====
        clip_positions = []
        _a_start = 0
        _a_end = frame_num
        _is_first = True
        _full_audio_len = min(max_frames_num, min(len(e) for e in full_audio_embs))
        while True:
            clip_positions.append((_a_start, _a_end, _is_first))
            if _a_end >= _full_audio_len:
                break
            if max_frames_num <= frame_num:
                break
            _is_first = False
            _a_start += (frame_num - motion_frame)
            _a_end = _a_start + frame_num

        print(f"[Precompute] {len(clip_positions)} clip(s), pre-computing CLIP + VAE ...")
        precomputed_clips = []
        with torch.no_grad():
            for _cidx, (_a_start, _a_end, _is_first) in enumerate(clip_positions):
                # 提取该 clip 位置的 cond_image
                if _cidx == 0:
                    _ci = cond_image
                else:
                    _ci = extract_specific_frames(cond_file_path, _a_start, bbox=bbox_orig)
                    if bbox_orig is not None:
                        _ci = resize_fit_letterbox(_ci, (target_h, target_w))
                        _ci = torch.from_numpy(np.array(_ci)).permute(2, 0, 1).float()[None, :, None, :, :]
                    else:
                        _ci = resize_and_centercrop(_ci, (target_h, target_w))
                    _ci = _ci / 255
                    _ci = (_ci - 0.5) * 2
                    _ci = _ci.to(self.device)

                # CLIP visual
                _clip_ctx = self.clip.visual(_ci[:, :, -1:, :, :]).to(self.param_dtype)

                # y = VAE encode of [cond_image + zeros] → concat msk
                _vf_zero = torch.zeros(1, _ci.shape[1], frame_num - _ci.shape[2],
                                       target_h, target_w, device=self.device)
                _padding = torch.concat([_ci, _vf_zero], dim=2)
                _y_raw = self.vae.encode(_padding)
                _y_enc = torch.stack(_y_raw).to(self.param_dtype)
                _y_enc = torch.concat([msk, _y_enc], dim=1)  # B 4+C T H W

                # 提取 clip_frames
                _clip_frames = extract_fragment_video(
                    cond_file_path, _a_start, _a_end, (target_h, target_w), bbox=bbox_orig)
                _clip_frames = _clip_frames.to(dtype=torch.float32, device=self.device).unsqueeze(0)

                # 首个 clip：还可以预计算 latent_motion_frames 和 latents
                if _is_first:
                    _lmf = self.vae.encode(_ci)[0]
                    _clip_frames[:, :, :1] = _ci
                    _latents_enc = self.vae.encode(_clip_frames)[0]
                else:
                    _lmf = None       # 后续 clip 需在线计算（依赖生成结果）
                    _latents_enc = None

                precomputed_clips.append({
                    'cond_image': _ci,
                    'clip_context': _clip_ctx,
                    'y': _y_enc,
                    'latent_motion_frames': _lmf,
                    'latents': _latents_enc,
                    'clip_frames': _clip_frames,
                })
        print(f"[Precompute] Done. Saved {len(precomputed_clips)} clip pre-computations.")

        # 预计算 timesteps（每个 clip 完全相同）
        if use_fms_schedule:
            timesteps = build_fms_timesteps(sampling_steps, self.device, shift=shift)
        else:
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

        @contextmanager
        def noop_no_sync():
            yield
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # start video generation iteratively
        clip_idx = 0
        while True:
            pc = precomputed_clips[clip_idx]

            # split audio with window size
            audio_embs = []
            for human_idx in range(HUMAN_NUMBER):
                center_indices = torch.arange(
                    audio_start_idx, audio_end_idx, 1,
                ).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
                audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                audio_embs.append(audio_emb)
            audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)

            noise = torch.randn(
                16, (frame_num - 1) // 4 + 1,
                lat_h - min_padding_len * 2,
                lat_w - min_padding_len * 2,
                dtype=torch.float32,
                device=self.device)

            with torch.no_grad():
                # [优化] 使用预计算的 CLIP 和 y（省去每 clip 的 CLIP forward + VAE encode y）
                clip_context = pc['clip_context']
                y = pc['y']
                cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)

                if is_first_clip:
                    # 首个 clip：所有 latent 已预计算
                    latent_motion_frames = pc['latent_motion_frames']
                    latents = pc['latents']
                else:
                    # 后续 clip：latent_motion_frames 来自上一段生成的 cond_frame（无法预计算）
                    if cond_frame.shape[-2:] != (target_h, target_w):
                        if cond_frame.dim() == 5:
                            n, c, t, _h, _w = cond_frame.shape
                            cond_frame = cond_frame.reshape(n * t, c, _h, _w)
                            cond_frame = F.interpolate(cond_frame, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            cond_frame = cond_frame.reshape(n, c, t, target_h, target_w)
                        else:
                            cond_frame = F.interpolate(cond_frame, size=(target_h, target_w), mode='bilinear', align_corners=False)

                    latent_motion_frames = self.vae.encode(cond_frame)[0]

                    # [优化] Latent 融合：将生成的最后一帧 latent 与真实的 latent 进行加权融合，作为下一段的锚点
                    # 融合比例：0.7 * 生成 + 0.3 * 真实，平滑过渡且防止长相崩坏
                    real_cond_image = pc['cond_image']
                    real_latent = self.vae.encode(real_cond_image)[0]
                    latent_motion_frames[:, :, -1:] = 0.7 * latent_motion_frames[:, :, -1:] + 0.3 * real_latent[:, :, -1:]

                    _cf = pc['clip_frames'].clone()
                    _cf[:, :, :cur_motion_frames_num] = cond_frame
                    latents = self.vae.encode(_cf)[0]

                    # 同样把融合后的 latent 更新到 latents 的对应位置
                    latents[:, :cur_motion_frames_latent_num] = latent_motion_frames

                if min_padding_len > 0:
                    latents[:, cur_motion_frames_latent_num:, min_padding_len:-min_padding_len, min_padding_len:-min_padding_len] = noise[:, cur_motion_frames_latent_num:]
                else:
                    latents[:, cur_motion_frames_latent_num:] = noise[:, cur_motion_frames_latent_num:]

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)

            # evaluation mode
            with torch.no_grad(), no_sync():

                # prepare timesteps
                if use_fms_schedule:
                    timesteps = build_fms_timesteps(sampling_steps, self.device, shift=shift)
                else:
                    timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
                    timesteps.append(0.)
                    timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                    if self.use_timestep_transform:
                        timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]

                # sample videos
                latent = latents
                or_latent = latent.clone()

                # prepare condition and uncondition configs
                arg_c = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }


                arg_null_text = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }

                arg_null_audio = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }


                arg_null = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }

                # infer with APG
                # refer https://arxiv.org/abs/2410.02416
                if extra_args.use_apg:
                    text_momentumbuffer  = MomentumBuffer(extra_args.apg_momentum)
                    audio_momentumbuffer = MomentumBuffer(extra_args.apg_momentum)

                # v6: CustomCache（与 wan2.2 一致），取消稀疏 CFG
                _n_steps = len(timesteps) - 1
                tea_cache_l1_thresh = getattr(extra_args, "tea_cache_l1_thresh", 0.12)
                feature_caching = getattr(extra_args, "feature_caching", "Custom")
                use_custom_cache = tea_cache_l1_thresh is not None and tea_cache_l1_thresh > 0
                if use_custom_cache:
                    self.model.disable_teacache()
                    cache_cls = CustomCache if (str(feature_caching).lower() == "custom") else TeaCache
                    model_id = getattr(extra_args, "size", "infinitetalk-480") or "infinitetalk-480"
                    tea_cache_cond = cache_cls(num_inference_steps=_n_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=model_id)
                    tea_cache_null_audio = cache_cls(num_inference_steps=_n_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=model_id)
                    tea_cache_cond.reset()
                    tea_cache_null_audio.reset()
                else:
                    tea_cache_cond = None
                    tea_cache_null_audio = None

                progress_wrap = partial(tqdm, total=len(timesteps)-1) if progress else (lambda x: x)
                for i in progress_wrap(range(len(timesteps)-1)):
                    timestep = timesteps[i]
                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                    latent[:,cur_motion_frames_latent_num:, :min_padding_len, :] = or_latent[:,cur_motion_frames_latent_num:, :min_padding_len, :]
                    latent[:,cur_motion_frames_latent_num:, -min_padding_len:, :] = or_latent[:,cur_motion_frames_latent_num:, -min_padding_len:, :]
                    latent[:,cur_motion_frames_latent_num:, :, :min_padding_len] = or_latent[:,cur_motion_frames_latent_num:, :, :min_padding_len]
                    latent[:,cur_motion_frames_latent_num:, :, -min_padding_len:] = or_latent[:,cur_motion_frames_latent_num:, :, -min_padding_len:]

                    latent_model_input = [latent.to(self.device)]

                    # inference with CFG strategy（每步都算 cond + null_audio，用 CustomCache 加速）
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c, tea_cache=tea_cache_cond)[0]
                    if i % 4 == 0 or i == len(timesteps)-2:
                        noise_pred_drop_audio = self.model(
                            latent_model_input, t=timestep, **arg_null_audio, tea_cache=tea_cache_null_audio)[0]
                        noise_pred = noise_pred_drop_audio + audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
                        noise_pred = -noise_pred
                    else:
                        noise_pred = -noise_pred_cond

                    # if math.isclose(text_guide_scale, 1.0):
                    #     noise_pred_drop_audio = self.model(
                    #         latent_model_input, t=timestep, **arg_null_audio)[0]
                    #     torch_gc()
                    # else:
                    #     noise_pred_drop_text = self.model(
                    #         latent_model_input, t=timestep, **arg_null_text)[0]
                    #     torch_gc()
                    #     noise_pred_uncond = self.model(
                    #         latent_model_input, t=timestep, **arg_null)[0]
                    #     torch_gc()

                    # if extra_args.use_apg:
                    #     # correct update direction
                    #     if math.isclose(text_guide_scale, 1.0):
                    #         diff_uncond_audio  = noise_pred_cond - noise_pred_drop_audio
                    #         noise_pred = noise_pred_cond + (audio_guide_scale - 1)* adaptive_projected_guidance(diff_uncond_audio,
                    #                                                                         noise_pred_cond,
                    #                                                                         momentum_buffer=audio_momentumbuffer,
                    #                                                                         norm_threshold=extra_args.apg_norm_threshold)
                    #     else:
                    #         diff_uncond_text  = noise_pred_cond - noise_pred_drop_text
                    #         diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                    #         noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_text,
                    #                                                                                             noise_pred_cond,
                    #                                                                                             momentum_buffer=text_momentumbuffer,
                    #                                                                                             norm_threshold=extra_args.apg_norm_threshold) \
                    #             + (audio_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_audio,
                    #                                                                         noise_pred_cond,
                    #                                                                         momentum_buffer=audio_momentumbuffer,
                    #                                                                         norm_threshold=extra_args.apg_norm_threshold)
                    # else:
                    #     # vanilla CFG strategy
                    #     if math.isclose(text_guide_scale, 1.0):
                    #         noise_pred = noise_pred_drop_audio + audio_guide_scale* (noise_pred_cond - noise_pred_drop_audio)
                    #     else:
                    #         noise_pred = noise_pred_uncond + text_guide_scale * (
                    #             noise_pred_cond - noise_pred_drop_text) + \
                    #             audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                    # noise_pred = -noise_pred

                    # update latent
                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / self.num_timesteps
                    latent = latent + noise_pred * dt[:, None, None, None]

                    ## latent[:, :cur_motion_frames_latent_num] = latent_motion_frames

                    x0 = [latent.to(self.device)]
                    del latent_model_input, timestep

                videos = self.vae.decode(x0)

            # cache generated samples
            videos = torch.stack(videos).cpu() # B C T H W
            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(videos, original_color_reference, color_correction_strength)
            # >>> END OF COLOR CORRECTION STEP <<<

            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # decide whether is done
            if arrive_last_frame: break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frame

            cond_frame = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            audio_start_idx += (frame_num - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            # [优化] cond_image 已在预计算阶段提取，无需重复读取视频

            # Repeat audio emb
            if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(HUMAN_NUMBER):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length   = audio_end_idx - len(full_audio_embs[human_inx]) + 3
                        add_audio_emb = torch.flip(full_audio_embs[human_inx][-1*miss_length:], dims=[0])
                        full_audio_embs[human_inx] = torch.cat([full_audio_embs[human_inx], add_audio_emb], dim=0)
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)


            if max_frames_num <= frame_num: break

            clip_idx += 1
            if offload_model:
                torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

        gen_video_samples = torch.cat(gen_video_list, dim=2)[:, :, :int(max_frames_num)]
        gen_video_samples = gen_video_samples.to(torch.float32)
        if max_frames_num > frame_num and sum(miss_lengths) > 0:
            # split video frames
            # gen_video_samples = gen_video_samples[:, :, :-1*miss_lengths[0]]
            gen_video_samples = gen_video_samples[:, :, :full_audio_emb.shape[0]]

        if dist.is_initialized():
            dist.barrier()

        del noise, latent
        torch_gc()

        return gen_video_samples[0] if self.rank == 0 else None
