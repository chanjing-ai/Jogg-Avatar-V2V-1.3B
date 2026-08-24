import os
import math
import torch
import random
import argparse
import lightning as pl
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from lightning.pytorch.loggers import TensorBoardLogger
from safetensors.torch import load_file
from wan.modules.multitalk_model import WanModel
from wan.schedulers.flow_match import FlowMatchScheduler
from wan.configs.wan_t2v_1_3B_train import t2v_1_3B_train

from torch.utils.data import Dataset, DataLoader

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
        self.prompt = torch.load(os.path.join(share_dir, "context.pt"), map_location="cpu")

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
                    audio_path = os.path.join(root, file.replace("latents_", "audio_"))
                    latent_y_path = os.path.join(root, file.replace("latents_", "latent_y_"))
                    ref_context_path = os.path.join(root, file.replace("latents_", "ref_context_"))
                    lmk203_path = os.path.join(root, file.replace("latents_", "lmk203_").replace(".pth", ".npy"))
                    if all(os.path.exists(p) for p in (audio_path, latent_y_path, ref_context_path, lmk203_path)):
                        self.paths.append(latent_path)

        if not self.paths:
            raise ValueError(f"在 {data_dir} 中没有找到符合条件的样本文件")

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _build_mask(T: int = 81, H: int = 60, W: int = 60) -> torch.Tensor:
        """
        生成形状为 [4, T//4, H, W] 的 mask（与原逻辑一致）：
          - 先构造 [1, T, H, W]，仅第 1 帧为 1，其余为 0
          - 将第 1 帧重复 4 次放在时间维前端，再接后续 T-1 帧
          - 重整为 [1, T//4, 4, H, W] 后转置 -> [1, 4, T//4, H, W]
          - 最终去掉 batch 维 -> [4, T//4, H, W]
        """
        msk = torch.ones(1, T, H, W, dtype=torch.float32)
        msk[:, 1:] = 0
        m_front = torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1)  # [1, 4, H, W]
        msk = torch.cat([m_front, msk[:, 1:]], dim=1)  # 时间长度 4+(T-1)
        t4 = msk.shape[1] // 4
        msk = msk.view(1, t4, 4, H, W).transpose(1, 2).to(torch.float32).squeeze(0)  # [4, t4, H, W]
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

        msk = self._build_mask(T=81, H=y.shape[2], W=y.shape[3])
        y = torch.cat([msk, y], dim=0)

        min_padding_len, ref_target_masks = self.get_target_mask(lmk203_data, y.shape[2]*8, y.shape[3]*8)

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
        - 加载并冻结已加载参数
        """
        super().__init__()
        self.model = WanModel(**model_config)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.load_model([wan_dit_path, infinitetalk_pred_model_path], resume_ckpt)

        self.stage = stage
        self.motion_sub_loss_weight = motion_sub_loss_weight
        self.learning_rate = learning_rate
        self.vae_stride = (4, 8, 8)
        self.sp_size = 1

    def load_model(self, weight_files, resume_ckpt=None):
        """
        合并权重并加载到 self.model：
        - 忽略 patch_embedding 的预训练权重（保持模型当前初始化）
        - strict=False，以便允许 missing/unexpected keys
        - 冻结成功加载的参数
        """
        merged_state_dict = {}
        for weight_file in weight_files:
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
            weight_files = resume_ckpt.split(",")
            merged_state_dict = {}
            for weight_file in weight_files:
                sd = load_file(weight_file)
                sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
                merged_state_dict.update(sd)
            self.model.load_state_dict(merged_state_dict)

        # 冻结已加载参数
        frozen_cnt = 0
        # for name, param in self.model.named_parameters():
        #     if name in loaded_keys:
        #         param.requires_grad = False
        #         frozen_cnt += 1

        trainable_cnt = sum(p.requires_grad for p in self.model.parameters())
        print(f"[freeze] frozen_params: {frozen_cnt}, trainable_params: {trainable_cnt}")

    def training_step(self, batch, batch_idx):
        """
        单步训练：
        - 以 prob_crop 概率仅对后 (T - keep_heads) 帧加噪（前 keep_heads 帧保留原样用于上下文稳定）
        - 前向预测并与对应目标对齐计算加权 MSE
        """
        frames = 81

        latents = batch['latents']
        context = batch['context']
        clip_fea = batch['clip_fea']
        y = batch['y']
        audio = batch['audio']
        min_padding_len = batch['min_padding_len']
        ref_target_masks = batch['ref_target_masks']

        device, dtype = latents.device, latents.dtype
        B, C, T, H, W = latents.shape

        r = random.random()
        if r < 0.1:
            keep_heads = 0
        elif r < 0.3:  # 即 r < 0.3
            keep_heads = 1
        else:
            keep_heads = 3

        m = random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.15, 0.15, 0.3, 0.3], k=1)[0]
        m = min(m, min_padding_len[0])
        hs = slice(m, -m) if m > 0 else slice(None)
        ws = slice(m, -m) if m > 0 else slice(None)

        x_core = latents[:, :, keep_heads:, hs, ws].clone() # [B, 16, 18/20, H, W]

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

        max_seq_len = ((frames - 1) // self.vae_stride[0] + 1) * H * W // (
            self.model.patch_size[1] * self.model.patch_size[2]
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        arg_c = {
            'context': context,
            'clip_fea': clip_fea,
            'seq_len': max_seq_len,
            'y': y,
            'audio': audio,
            'ref_target_masks': ref_target_masks.squeeze(0),
        }

        preds_noise = self.model(latent_noisy, timestep, **arg_c)

        preds_core = preds_noise[:, :, keep_heads:, hs, ws]
        assert preds_core.shape == training_target.shape, "预测与监督目标形状不一致"

        # FlowMatch 加权损失
        loss_weights = self.scheduler.training_weight(timestep)
        loss_flow = torch.nn.functional.mse_loss(preds_core.float(), training_target.float(), reduction='mean')

        if self.motion_sub_loss_weight > 0:
            pre_sub_noise = preds_core[:, :, 1:, :, :].float() - preds_core[:, :, :-1, :, :].float()
            gt_sub_noise = training_target[:, :, 1:, :, :].float() - training_target[:, :, :-1, :, :].float()
            sub_loss = torch.nn.functional.mse_loss(pre_sub_noise, gt_sub_noise, reduction='mean')
            loss_flow = loss_flow * (1-self.motion_sub_loss_weight) + sub_loss * self.motion_sub_loss_weight
        else:
            sub_loss = 0
        loss = loss_flow * loss_weights


        self.log_dict({"train_loss": loss, "loss_flow": loss_flow*loss_weights, "random_choice_r": float(r), "random_choice_m": float(m)}, prog_bar=True)

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
    parser.add_argument('--stage', default=2, type=int, help="Stage of training.")
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--share_dir', default='./data/share', type=str)
    parser.add_argument('--wan_dit_path', default='./models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors', type=str)
    parser.add_argument('--infinitetalk_pred_model_path', default='./models/Jogg-Avatar-V2V-1.3B/training_init/audio_proj.safetensors', type=str)
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Optional safetensors shards separated by commas')
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    return parser.parse_args()

def train(args):
    model_config = {**t2v_1_3B_train}
    # pl.seed_everything(42, workers=True)
    # 使用 Dummy 数据集和 DataLoader
    dataset = InfiniteTalkDataset(data_dir=args.data_dir, share_dir=args.share_dir)  # 可以按需调大/调小
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
        motion_sub_loss_weight=args.motion_sub_loss_weight)

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
