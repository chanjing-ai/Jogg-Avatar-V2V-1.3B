import os
import math
import torch
import random
import argparse
import lightning as pl
import numpy as np

from pathlib import Path
from typing import Tuple
from lightning.pytorch.loggers import TensorBoardLogger
from safetensors import safe_open
from safetensors.torch import load_file
from wan.modules.multitalk_model import WanModel
from wan.schedulers.flow_match import FlowMatchScheduler
from wan.configs.wan_t2v_1_3B_train import t2v_1_3B_train

from torch.utils.data import Dataset, DataLoader


VAE_TEMPORAL_STRIDE = 4
LATENT_CHANNELS = 16
CONDITION_CHANNELS = 20


def _split_checkpoint_paths(value):
    return [path.strip() for path in value.split(",") if path.strip()]


def _require_shape(name, tensor, ndim, suffix=None):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim:
        shape = getattr(tensor, "shape", None)
        raise ValueError(f"{name} must be a {ndim}D tensor, got {shape}")
    if suffix is not None and tuple(tensor.shape[-len(suffix):]) != tuple(suffix):
        raise ValueError(
            f"{name} must end with shape {tuple(suffix)}, got {tuple(tensor.shape)}"
        )


def validate_training_batch(batch):
    required = {
        "latents", "context", "clip_fea", "y", "audio",
        "min_padding_len", "ref_target_masks",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"training batch is missing: {', '.join(missing)}")

    latents = batch["latents"]
    _require_shape("latents", latents, 5)
    batch_size, channels, latent_frames, height, width = latents.shape
    if batch_size != 1:
        raise ValueError(
            "InfiniteTalk training currently requires batch_size=1; "
            f"got {batch_size}"
        )
    if channels != LATENT_CHANNELS:
        raise ValueError(
            f"latents must have {LATENT_CHANNELS} channels, got {channels}"
        )

    y = batch["y"]
    _require_shape("y", y, 5)
    expected_y = (batch_size, CONDITION_CHANNELS, latent_frames, height, width)
    if tuple(y.shape) != expected_y:
        raise ValueError(f"y must have shape {expected_y}, got {tuple(y.shape)}")

    context = batch["context"]
    _require_shape("context", context, 3)
    if (
        context.shape[0] != batch_size
        or context.shape[1] > 512
        or context.shape[-1] != 4096
    ):
        raise ValueError(
            f"context must have shape [1, <=512, 4096], got {tuple(context.shape)}"
        )

    clip_fea = batch["clip_fea"]
    _require_shape("clip_fea", clip_fea, 3)
    if tuple(clip_fea.shape) != (batch_size, 257, 1280):
        raise ValueError(
            f"clip_fea must have shape [1, 257, 1280], got {tuple(clip_fea.shape)}"
        )

    expected_audio_frames = 1 + (latent_frames - 1) * VAE_TEMPORAL_STRIDE
    audio = batch["audio"]
    _require_shape("audio", audio, 5)
    expected_audio = (batch_size, expected_audio_frames, 5, 12, 768)
    if tuple(audio.shape) != expected_audio:
        raise ValueError(f"audio must have shape {expected_audio}, got {tuple(audio.shape)}")

    masks = batch["ref_target_masks"]
    _require_shape("ref_target_masks", masks, 4)
    expected_masks = (batch_size, 3, height, width)
    if tuple(masks.shape) != expected_masks:
        raise ValueError(
            f"ref_target_masks must have shape {expected_masks}, got {tuple(masks.shape)}"
        )

    padding = batch["min_padding_len"]
    if not isinstance(padding, torch.Tensor) or padding.numel() != batch_size:
        raise ValueError("min_padding_len must contain one value per sample")
    return batch_size, channels, latent_frames, height, width


class InfiniteTalkDataset(Dataset):
    def __init__(self, data_dir: str, share_dir: str):
        """
        Args:
            data_dir: 数据根目录，递归查找样本
            share_dir: 存放全局共享张量（context.pt / padding_latent.pth）
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
        if not os.path.isdir(share_dir):
            raise ValueError(f"共享目录不存在: {share_dir}")

        self.data_dir = data_dir
        self.share_dir = share_dir

        # 共享资源：保持在 CPU，Lightning/训练循环会负责迁移到设备
        self.prompt = torch.load(
            os.path.join(share_dir, "context.pt"),
            map_location="cpu",
        )
        _require_shape("context.pt", self.prompt, 2)
        if self.prompt.shape[-1] != 4096:
            raise ValueError(
                f"context.pt must have shape [tokens, 4096], got {tuple(self.prompt.shape)}"
            )

        # 收集样本（要求 4 个配套文件同时存在）
        self.paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if (
                    file.startswith("latents_")
                    and not file.startswith("latents_y_")
                    and file.endswith(".pth")
                ):
                    latent_path = os.path.join(root, file)
                    audio_path = os.path.join(
                        root, file.replace("latents_", "audio_")
                    )
                    latent_y_path = os.path.join(
                        root, file.replace("latents_", "latent_y_")
                    )
                    ref_context_path = os.path.join(
                        root, file.replace("latents_", "ref_context_")
                    )
                    lmk203_path = os.path.join(
                        root,
                        file.replace("latents_", "lmk203_").replace(".pth", ".npy"),
                    )
                    companions = (
                        audio_path,
                        latent_y_path,
                        ref_context_path,
                        lmk203_path,
                    )
                    if all(os.path.exists(path) for path in companions):
                        self.paths.append(latent_path)

        self.paths.sort()
        if not self.paths:
            raise ValueError(f"在 {data_dir} 中没有找到符合条件的样本文件")

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _build_mask(latent_frames: int, H: int, W: int) -> torch.Tensor:
        """
        Build the four mask channels consumed beside the 16-channel latent.
        All four channels mark the first latent frame as the reference frame.
        """
        if latent_frames < 1 or H < 1 or W < 1:
            raise ValueError(
                f"invalid latent mask shape: frames={latent_frames}, H={H}, W={W}"
            )
        msk = torch.zeros(4, latent_frames, H, W, dtype=torch.float32)
        msk[:, 0] = 1
        return msk

    def segment_bbox_robust(
        self,
        landmarks_seg: np.ndarray,
        q_low: float = 0.03,
        q_high: float = 0.97,
    ) -> Tuple[float, float, float, float]:
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

    def get_target_mask(self, lmk203_list, target_height, target_width):
        """
        基于所有帧的人脸关键点，计算一个全局人脸框(global face box)，再缩小到latent尺度(//8)，
        构造 human1/human2/background 三张mask，并返回 ref_target_masks = stack([..], dim=0)
        """
        landmarks = np.asarray(lmk203_list)
        if landmarks.ndim != 3 or landmarks.shape[1] < 203 or landmarks.shape[2] != 2:
            raise ValueError(
                "landmarks must have shape [frames, >=203, 2], "
                f"got {landmarks.shape}"
            )

        # 1) 提取人脸关键点 (114-137, 202)，构建 landmarks_seg: (T, 25, 2)
        face_indices = list(range(114, 138)) + [202]
        landmarks_seg = np.array([
            [landmarks[i][j] for j in face_indices]
            for i in range(len(landmarks))
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

        min_padding_len = min(
            x0,
            y0,
            (lat_w - 1) - x1,
            (lat_h - 1) - y1,
        )
        min_padding_len = max(min_padding_len, 0)

        # 3) 构造三个二维mask
        human1_mask = torch.zeros((lat_h, lat_w), dtype=torch.float32)
        human1_mask[y0:y1 + 1, x0:x1 + 1] = 1.0

        human2_mask = torch.zeros((lat_h, lat_w), dtype=torch.float32)

        background_mask = 1.0 - human1_mask  # 与human1_mask刚好相反

        human_masks = [human1_mask, human2_mask, background_mask]
        ref_target_masks = torch.stack(human_masks, dim=0)  # [3, lat_h, lat_w]
        return min_padding_len, ref_target_masks


    def __getitem__(self, index):
        latent_path = self.paths[index]
        audio_path = latent_path.replace("latents_", "audio_")
        latent_y_path = latent_path.replace("latents_", "latent_y_")
        ref_context_path = latent_path.replace("latents_", "ref_context_")
        lmk203_path = latent_path.replace("latents_", "lmk203_").replace(".pth", ".npy")

        # 加载样本相关张量（保持 CPU）
        audio = torch.load(audio_path, map_location="cpu")
        latents = torch.load(latent_path, map_location="cpu")
        latent_y = torch.load(latent_y_path, map_location="cpu")
        lmk203_raw = np.load(lmk203_path, allow_pickle=True)
        if lmk203_raw.ndim == 0:
            lmk203_data = lmk203_raw.item()["lmk203_list"]
        else:
            lmk203_data = lmk203_raw
        ref_context = torch.load(ref_context_path, map_location="cpu")

        y = latent_y["y_latent"] if isinstance(latent_y, dict) else latent_y
        clip_fea = ref_context["ref_context"] if isinstance(ref_context, dict) else ref_context

        _require_shape("latents", latents, 4)
        _require_shape("latent_y", y, 4)
        if latents.shape[0] != LATENT_CHANNELS:
            raise ValueError(f"latents must have {LATENT_CHANNELS} channels, got {latents.shape[0]}")
        if tuple(y.shape) != tuple(latents.shape):
            raise ValueError(
                f"latent_y must match latents, got {tuple(y.shape)} vs {tuple(latents.shape)}"
            )
        _require_shape("clip_fea", clip_fea, 2, suffix=(257, 1280))
        expected_audio_frames = 1 + (latents.shape[1] - 1) * VAE_TEMPORAL_STRIDE
        _require_shape("audio", audio, 4)
        expected_audio = (expected_audio_frames, 5, 12, 768)
        if tuple(audio.shape) != expected_audio:
            raise ValueError(
                f"audio must have shape {expected_audio}, got {tuple(audio.shape)}"
            )
        if len(lmk203_data) != expected_audio_frames:
            raise ValueError(
                f"landmark frame count must be {expected_audio_frames}, got {len(lmk203_data)}"
            )

        msk = self._build_mask(
            latent_frames=y.shape[1],
            H=y.shape[2],
            W=y.shape[3],
        )
        y = torch.cat([msk, y], dim=0)

        min_padding_len, ref_target_masks = self.get_target_mask(
            lmk203_data,
            y.shape[2] * 8,
            y.shape[3] * 8,
        )

        return {
            "latents": latents,
            "context": self.prompt,
            "clip_fea": clip_fea,
            "y": y,
            "audio": audio,
            "min_padding_len": min_padding_len,
            "ref_target_masks": ref_target_masks,
        }

class InfiniteTalkTrain(pl.LightningModule):
    def __init__(
        self,
        model_config,
        wan_dit_path,
        resume_ckpt,
        infinitetalk_pred_model_path,
        learning_rate=1e-5,
        stage=1,
        motion_sub_loss_weight=0,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    ):
        """
        Lightning 封装：
        - 构建 WanModel
        - 配置 FlowMatch 调度器
        - 加载模型初始化权重
        """
        super().__init__()
        self.model = WanModel(**model_config)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        initial_weight_files = []
        for value in (wan_dit_path, infinitetalk_pred_model_path):
            if isinstance(value, str):
                initial_weight_files.extend(_split_checkpoint_paths(value))
            else:
                initial_weight_files.extend(value)
        self.load_model(initial_weight_files, resume_ckpt)

        self.stage = stage
        self.motion_sub_loss_weight = motion_sub_loss_weight
        self.learning_rate = learning_rate
        self.sp_size = 1

    def load_model(self, weight_files, resume_ckpt=None):
        """
        合并权重并加载到 self.model：
        - 忽略 patch_embedding 的预训练权重（保持模型当前初始化）
        - strict=False，以便允许 missing/unexpected keys
        - 1.3B release uses full-parameter training; loaded tensors stay trainable
        """
        merged_state_dict = {}
        for weight_file in weight_files:
            if not os.path.isfile(weight_file):
                raise FileNotFoundError(f"checkpoint does not exist: {weight_file}")
            sd = load_file(weight_file)
            merged_state_dict.update(sd)

        # 忽略 patch_embedding 权重（安全 pop，避免 KeyError）
        merged_state_dict.pop('patch_embedding.bias', None)
        merged_state_dict.pop('patch_embedding.weight', None)

        load_info = self.model.load_state_dict(merged_state_dict, strict=False)

        # 统计信息
        unexpected = set(getattr(load_info, "unexpected_keys", []))
        missing = set(getattr(load_info, "missing_keys", []))
        loaded_keys = set(merged_state_dict.keys()) - unexpected

        print(f"[load] loaded: {len(loaded_keys)}, unexpected: {len(unexpected)}, missing: {len(missing)}")
        if resume_ckpt:
            weight_files = _split_checkpoint_paths(resume_ckpt)
            merged_state_dict = {}
            for weight_file in weight_files:
                if not os.path.isfile(weight_file):
                    raise FileNotFoundError(f"resume checkpoint does not exist: {weight_file}")
                sd = load_file(weight_file)
                sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
                merged_state_dict.update(sd)
            self.model.load_state_dict(merged_state_dict)

        trainable_cnt = sum(p.requires_grad for p in self.model.parameters())
        print(f"[train] trainable parameter tensors: {trainable_cnt}")

    def training_step(self, batch, batch_idx):
        """
        单步训练：
        - 以 prob_crop 概率仅对后 (T - keep_heads) 帧加噪（前 keep_heads 帧保留原样用于上下文稳定）
        - 前向预测并与对应目标对齐计算加权 MSE
        """
        latents = batch['latents']
        context = batch['context']
        clip_fea = batch['clip_fea']
        y = batch['y']
        audio = batch['audio']
        min_padding_len = batch['min_padding_len']
        ref_target_masks = batch['ref_target_masks']

        B, C, T, H, W = validate_training_batch(batch)
        device, dtype = latents.device, latents.dtype

        r = random.random()
        if r < 0.1:
            keep_heads = 0
        elif r < 0.3:  # 即 r < 0.3
            keep_heads = 1
        else:
            keep_heads = 3

        m = random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.15, 0.15, 0.3, 0.3], k=1)[0]
        m = min(m, int(min_padding_len.item()))
        hs = slice(m, -m) if m > 0 else slice(None)
        ws = slice(m, -m) if m > 0 else slice(None)

        x_core = latents[:, :, keep_heads:, hs, ws].clone()

        noise = torch.randn_like(x_core)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=dtype, device=device)

        x_noisy_core = self.scheduler.add_noise(x_core, noise, timestep)
        training_target = self.scheduler.training_target(x_core, noise, timestep)

        if keep_heads > 0:
            latents_head = latents[:, :, :keep_heads, :, :].clone()
            latents_core = latents[:, :, keep_heads:, :, :].clone()
            latents_core[:, :, :, hs, ws] = x_noisy_core
            latent_noisy = torch.cat([latents_head, latents_core], dim=2)
        else:
            latents_core = latents.clone()
            latents_core[:, :, :, hs, ws] = x_noisy_core
            latent_noisy = latents_core

        patch_t, patch_h, patch_w = self.model.patch_size
        max_seq_len = (T // patch_t) * (H // patch_h) * (W // patch_w)
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        arg_c = {
            'context': context,
            'clip_fea': clip_fea,
            'seq_len': max_seq_len,
            'y': y,
            'audio': audio,
            'ref_target_masks': ref_target_masks.squeeze(0),
        }

        preds_noise = self.model(
            latent_noisy,
            timestep,
            **arg_c,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        preds_core = preds_noise[:, :, keep_heads:, hs, ws]
        if preds_core.shape != training_target.shape:
            raise RuntimeError(
                "prediction and target shapes differ: "
                f"{tuple(preds_core.shape)} vs {tuple(training_target.shape)}"
            )

        # FlowMatch 加权损失
        loss_weights = self.scheduler.training_weight(timestep)
        loss_flow = torch.nn.functional.mse_loss(
            preds_core.float(),
            training_target.float(),
            reduction='mean',
        )

        if self.motion_sub_loss_weight > 0:
            pre_sub_noise = preds_core[:, :, 1:, :, :].float() - preds_core[:, :, :-1, :, :].float()
            gt_sub_noise = training_target[:, :, 1:, :, :].float() - training_target[:, :, :-1, :, :].float()
            sub_loss = torch.nn.functional.mse_loss(pre_sub_noise, gt_sub_noise, reduction='mean')
            loss_flow = (
                loss_flow * (1 - self.motion_sub_loss_weight)
                + sub_loss * self.motion_sub_loss_weight
            )
        else:
            sub_loss = 0
        loss = loss_flow * loss_weights


        self.log_dict(
            {
                "train_loss": loss,
                "loss_flow": loss_flow * loss_weights,
                "random_choice_r": float(r),
                "random_choice_m": float(m),
            },
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        """
        仅优化未冻结参数
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        print("#" * 50)
        print(f"可训练参数数量: {len(trainable_params)}")
        print("#" * 50)

        if len(trainable_params) == 0:
            raise RuntimeError("没有可训练参数，请检查冻结逻辑或权重加载。")
        print(f"learning_rate: {self.learning_rate}")
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        return optimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train InfiniteTalk V2V 1.3B")
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--save_steps', default=3000, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--output_path', default='./checkpoints', type=str)
    parser.add_argument('--experiment_name', default='infinitetalk_train', type=str)
    parser.add_argument('--motion_sub_loss_weight', default=0, type=float)
    parser.add_argument(
        '--stage', default=2, type=int, choices=[1, 2],
        help="Experiment stage label; both stages currently use the same Flow Matching objective.",
    )
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--share_dir', default='./data/share', type=str)
    parser.add_argument('--wan_dit_path', default='./models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors', type=str)
    parser.add_argument('--infinitetalk_pred_model_path', default='./models/Chanjing-Avatar-V2V-1.3B/training_init/audio_proj.safetensors', type=str)
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Optional safetensors shards separated by commas')
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument(
        '--disable_gradient_checkpointing', action='store_true',
        help='Disable activation checkpointing (uses more GPU memory).',
    )
    parser.add_argument(
        '--use_gradient_checkpointing_offload', action='store_true',
        help='Offload saved checkpoint activations to CPU.',
    )
    parser.add_argument(
        '--validate_only', action='store_true',
        help='Validate one dataset sample and checkpoint tensor contracts without training.',
    )

    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    args = parser.parse_args()
    if not 0 <= args.motion_sub_loss_weight <= 1:
        parser.error("--motion_sub_loss_weight must be between 0 and 1")
    if args.disable_gradient_checkpointing and args.use_gradient_checkpointing_offload:
        parser.error(
            "--disable_gradient_checkpointing conflicts with "
            "--use_gradient_checkpointing_offload"
        )
    return args


def _checkpoint_shapes(paths, strip_model_prefix=False):
    shapes = {}
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                key = raw_key[6:] if strip_model_prefix and raw_key.startswith("model.") else raw_key
                if key in shapes:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                shapes[key] = tuple(handle.get_slice(raw_key).get_shape())
    return shapes


def _compare_checkpoint_shapes(actual, expected, require_complete, label):
    unexpected = sorted(set(actual) - set(expected))
    bad_shapes = {
        key: (shape, expected[key])
        for key, shape in actual.items()
        if key in expected and shape != expected[key]
    }
    missing = sorted(set(expected) - set(actual)) if require_complete else []
    if unexpected or bad_shapes or missing:
        parts = []
        if unexpected:
            parts.append(f"unexpected={unexpected[:10]}")
        if missing:
            parts.append(f"missing={missing[:10]}")
        if bad_shapes:
            parts.append(f"shape_mismatches={list(bad_shapes.items())[:10]}")
        raise ValueError(f"{label} checkpoint contract failed: " + "; ".join(parts))


def validate_training_setup(args):
    dataset = InfiniteTalkDataset(args.data_dir, args.share_dir)
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else torch.tensor([value])
        for key, value in sample.items()
    }
    _, _, latent_frames, height, width = validate_training_batch(batch)

    from accelerate import init_empty_weights

    with init_empty_weights():
        model = WanModel(**dict(t2v_1_3B_train))
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    initial_paths = _split_checkpoint_paths(args.wan_dit_path)
    initial_paths += _split_checkpoint_paths(args.infinitetalk_pred_model_path)
    initial = _checkpoint_shapes(initial_paths)
    initial.pop('patch_embedding.bias', None)
    initial.pop('patch_embedding.weight', None)
    _compare_checkpoint_shapes(initial, expected, False, "initial")

    if args.resume_ckpt:
        resumed = _checkpoint_shapes(
            _split_checkpoint_paths(args.resume_ckpt),
            strip_model_prefix=True,
        )
        _compare_checkpoint_shapes(resumed, expected, True, "resume")

    print(
        "Training validation passed: "
        f"samples={len(dataset)}, latent=(16,{latent_frames},{height},{width}), "
        f"initial_tensors={len(initial)}, model_tensors={len(expected)}, "
        f"resume={'yes' if args.resume_ckpt else 'no'}"
    )


def train(args):
    if args.validate_only:
        validate_training_setup(args)
        return

    model_config = {**t2v_1_3B_train}
    pl.seed_everything(args.seed, workers=True)
    dataset = InfiniteTalkDataset(
        data_dir=args.data_dir,
        share_dir=args.share_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = InfiniteTalkTrain(model_config,
        args.wan_dit_path,
        args.resume_ckpt,
        args.infinitetalk_pred_model_path,
        learning_rate=args.learning_rate,
        stage=args.stage,
        motion_sub_loss_weight=args.motion_sub_loss_weight,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )

    logger = TensorBoardLogger(
        save_dir=args.output_path,
        name="tensorboard",
        version=args.experiment_name,
    )



    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,  # 每1步记录一次日志
        callbacks=[
        pl.pytorch.callbacks.ModelCheckpoint(
            save_top_k=-1,
            every_n_train_steps=args.save_steps
        )
    ],
        logger=logger,
    )
    # ckpt_path = args.resume_ckpt if (args.resume_ckpt and os.path.isfile(args.resume_ckpt)) else None
    # if ckpt_path:
    #     trainer.fit(model, dataloader, ckpt_path=ckpt_path)
    # else:
    #     trainer.fit(model, dataloader)
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    args = parse_args()
    train(args)
