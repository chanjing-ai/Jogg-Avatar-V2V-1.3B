import os
import sys
import cv2
import torch
import random
import argparse
import numpy as np
import multiprocessing as mp

from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision.transforms import v2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan.modules.vae import WanVAE
from wan.modules.clip import CLIPModel
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln


class VideoAudioDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.frame_interval = args.frame_interval
        self.min_num_frames = args.num_frames
        self.num_frames = args.num_frames
        self.device = args.device

        self.frame_process = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.vae = WanVAE(
            vae_pth=args.vae_path,
            device=self.device)

        self.clip = CLIPModel(
            dtype=torch.float16,
            device=self.device,
            checkpoint_path=args.clip_checkpoint,
            tokenizer_path=args.clip_tokenizer)

        self.all_dataset = []

        self.wav2vec_processor, self.wav2vec = self.custom_init(self.device, args.audio_encoder_path)

    def custom_init(self, device, wav2vec):
        audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
        return wav2vec_feature_extractor, audio_encoder

    def get_device_id(self):
        if "cuda:" in self.device:
            return self.device.split(":")[1]
        return 0

    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def get_audio_embedding(self, speech_array, fps, sr=16000):
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * fps

        audio_feature = np.squeeze(
            self.wav2vec_processor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(self.device)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.wav2vec(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()
        return audio_emb

    def segment_bbox_robust(
        self,
        landmarks_seg: np.ndarray,
        q_low: float = 0.03,
        q_high: float = 0.97,
    ) -> Tuple[float, float, float, float]:
        xs = landmarks_seg[:, :, 0].reshape(-1)
        ys = landmarks_seg[:, :, 1].reshape(-1)

        xmin = np.quantile(xs, q_low)
        xmax = np.quantile(xs, q_high)
        ymin = np.quantile(ys, q_low)
        ymax = np.quantile(ys, q_high)

        return float(xmin), float(ymin), float(xmax), float(ymax)

    def temporal_smooth(self, landmarks, win=5):
        kernel = np.ones(win) / win
        out = landmarks.copy()
        for i in range(landmarks.shape[1]):
            for d in range(2):
                out[:, i, d] = np.convolve(
                    landmarks[:, i, d], kernel, mode="same"
                )
        return out

    def design_box2_for_segment(
        self,
        landmarks_seg,
        width: int,
        height: int,
        margin: int = 16,
    ) -> Tuple[float, float, float, float]:
        landmarks = self.temporal_smooth(landmarks_seg, win=5)
        pxmin, pymin, pxmax, pymax = self.segment_bbox_robust(landmarks)

        w = pxmax - pxmin
        h = pymax - pymin
        side_base = max(w, h)
        side = side_base * 2.0
        side += 2 * margin

        cx = (pxmin + pxmax) / 2.0
        cy = (pymin + pymax) / 2.0

        x_min = cx - side / 2.0
        x_max = cx + side / 2.0
        y_min = cy - side / 2.0
        y_max = cy + side / 2.0

        if side <= width:
            x_min = max(0.0, min(x_min, float(width) - side))
            x_max = x_min + side
        else:
            x_min = 0.0
            x_max = float(width)

        if side <= height:
            y_min = max(0.0, min(y_min, float(height) - side))
            y_max = y_min + side
        else:
            y_min = 0.0
            y_max = float(height)

        x_min = max(0.0, min(x_min, float(width)))
        x_max = max(0.0, min(x_max, float(width)))
        y_min = max(0.0, min(y_min, float(height)))
        y_max = max(0.0, min(y_max, float(height)))

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def design_box2_for_clips_landmarks(
        self,
        clips,
        landmarks,
        ref_clip_idxs,
        width: int,
        height: int,
    ):
        box2_list = []
        ref_box2_map = {}
        for i in range(len(clips)):
            if i == 0 and len(clips) == 1:
                start_idx = clips[i][0]
                end_idx = clips[i][1] + 1
            elif i == 0:
                start_idx = clips[i][0]
                end_idx = clips[i][1] + 25 + 1
            elif i == len(clips) - 1:
                start_idx = clips[i][0] - 25
                start_idx = max(0, start_idx)
                end_idx = clips[i][1] + 1
            else:
                start_idx = clips[i][0] - 25
                start_idx = max(0, start_idx)
                end_idx = clips[i][1] + 25 + 1
            end_idx = min(end_idx, len(landmarks))
            if ref_clip_idxs[i] < start_idx or ref_clip_idxs[i] >= end_idx:
                return None, None

            box2 = self.design_box2_for_segment(landmarks[start_idx:end_idx], width, height)
            box2_list.append(box2)
            ref_box2_map[ref_clip_idxs[i]] = box2

        return box2_list, ref_box2_map

    def split_frames_into_clips(self, total_frames: int, num_frames: int, frame_interval: int):
        last_tag = False
        if total_frames < num_frames:
            return [], [], last_tag

        stride = num_frames
        clips = []
        start = 0
        while start + num_frames <= total_frames:
            end = start + num_frames - 1
            clips.append((start, end))
            start += stride

        remaining = total_frames - (clips[-1][1] + 1) if clips else total_frames
        if remaining > frame_interval:
            tail = (max(0, total_frames - num_frames), total_frames - 1)
            if not clips or clips[-1] != tail:
                clips.append(tail)
                last_tag = True

        def sample_k(lo: int, hi: int, k: int):
            lo = max(0, lo)
            hi = min(total_frames - 1, hi)
            if lo > hi:
                p = max(0, min(total_frames - 1, lo))
                return [p] * k
            pool = list(range(lo, hi + 1))
            if len(pool) >= k:
                return random.sample(pool, k)
            if len(pool) == 1:
                return [pool[0]] * k
            return [random.choice(pool) for _ in range(k)]

        clip_randoms = []
        n = len(clips)

        for i, (s, e) in enumerate(clips):
            is_first = i == 0
            is_last = i == n - 1

            if n == 1:
                picks = sample_k(s, e, 2)
            elif is_first:
                picks = sample_k(e + 1, e + 25, 2)
            elif is_last:
                picks = sample_k(s - 25, s - 1, 2)
            else:
                pre = sample_k(s - 25, s - 1, 1)[0]
                post = sample_k(e + 1, e + 25, 1)[0]
                picks = [pre, post]

            clip_randoms.append(tuple(picks))

        return clips, clip_randoms, last_tag

    def process_latent_y(self, frame):
        h, w = frame.shape[:2]
        candidates = [(512, 512), (448, 576), (576, 448)]
        orig_ratio = w / h
        target_w, target_h = min(
            candidates,
            key=lambda wh: abs((wh[0] / wh[1]) - orig_ratio)
        )
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        frame = canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.frame_process(Image.fromarray(frame))
        frame = frame.to(dtype=torch.float32, device=self.device)
        frame = frame.unsqueeze(0)
        frame = rearrange(frame, "T C H W -> C T H W")
        frame = frame.unsqueeze(0)
        zero_padding = torch.zeros(1, 3, 80, target_h, target_w, dtype=torch.float32, device=self.device)
        ref_padding = torch.cat([frame, zero_padding], dim=2)
        with torch.no_grad():
            y_latent = self.vae.encode(ref_padding)[0]
            ref_context = self.clip.visual(frame).squeeze(0)
        return y_latent.cpu(), ref_context.cpu()

    def process_latent_x(self, frames, landmarks, box2):
        x_min, y_min, x_max, y_max = box2
        w = x_max - x_min
        h = y_max - y_min
        candidates = [(512, 512), (448, 576), (576, 448)]
        orig_ratio = w / h
        target_w, target_h = min(
            candidates,
            key=lambda wh: abs((wh[0] / wh[1]) - orig_ratio)
        )
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        resize_frames = []
        resize_landmarks = []
        for frame, lmk in zip(frames, landmarks):
            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            frame = canvas
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.frame_process(Image.fromarray(frame))
            resize_frames.append(frame)

            lmk_c = lmk.copy()
            lmk_c[..., 0] = lmk_c[..., 0] - x_min
            lmk_c[..., 1] = lmk_c[..., 1] - y_min
            lmk_c[..., 0] = lmk_c[..., 0] * scale
            lmk_c[..., 1] = lmk_c[..., 1] * scale
            lmk_c[..., 0] = lmk_c[..., 0] + x_offset
            lmk_c[..., 1] = lmk_c[..., 1] + y_offset

            resize_landmarks.append(lmk_c)

        resize_frames = torch.stack(resize_frames, dim=0)
        resize_frames = rearrange(resize_frames, "T C H W -> C T H W")
        resize_frames = resize_frames.to(dtype=torch.float32, device=self.device)
        resize_frames = resize_frames.unsqueeze(0)
        with torch.no_grad():
            latents = self.vae.encode(resize_frames)[0]
        return latents.cpu(), resize_landmarks

    def process_audio(self, audio_path, clips, total_video_frames, fps, output_dir, sample_rate=16000):
        audio, sr = librosa.load(audio_path, sr=sample_rate)

        pad_frames = int(self.num_frames)
        samples_per_frame = sample_rate / float(fps)

        audio_len = len(audio)
        indices = (torch.arange(2 * 2 + 1) - 2) * 1

        save_index = 0
        for (s, e) in clips:
            left_pad = min(pad_frames, s)
            right_pad = min(pad_frames, (total_video_frames - 1) - e)

            ext_s = s - left_pad
            ext_e = e + right_pad + 1

            start_sample = int(round(ext_s * samples_per_frame))
            end_sample = int(round(ext_e * samples_per_frame))

            start_sample = max(0, min(start_sample, audio_len))
            end_sample = max(start_sample, min(end_sample, audio_len))

            seg_audio = audio[start_sample:end_sample]
            if seg_audio.size == 0:
                return None

            seg_audio = self.loudness_norm(seg_audio, sr=sample_rate)

            audio_emb = self.get_audio_embedding(seg_audio, fps=fps, sr=sample_rate)
            if audio_emb is None:
                return None

            ext_frames_eff = int(len(seg_audio) / samples_per_frame)
            ext_e_eff = ext_s + ext_frames_eff - 1
            right_pad_eff = max(0, ext_e_eff - e)
            left_pad_eff = left_pad

            T = audio_emb.shape[0]
            trim_left = min(left_pad_eff, T)
            trim_right = min(right_pad_eff, max(0, T - trim_left))
            trim_right = T - trim_right
            if trim_right - trim_left <= self.num_frames - 3:
                return None
            trim_right = trim_left + self.num_frames
            center_indices = torch.arange(
                trim_left,
                trim_right,
                1,
            ).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0]-1)
            audio_clip = audio_emb[center_indices][None, ...].squeeze(0)

            tensor_filename = f"audio_{save_index+1:03d}.pth"
            tensor_path = os.path.join(output_dir, tensor_filename)
            torch.save(audio_clip.cpu(), tensor_path)
            save_index += 1

    def run_getitem(self, video_path, audio_path, lmk_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert fps == 25, "fps must be 25"
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        landmarks = np.load(lmk_path)
        if len(landmarks) < self.num_frames:
            return False

        clips, clip_randoms, last_tag = self.split_frames_into_clips(len(landmarks), self.num_frames, self.frame_interval)

        ref_clip_idxs = []
        for clip_idx, clip_rand_idx in zip(clips, clip_randoms):
            rand_idx = random.randint(clip_idx[0], clip_idx[1])
            rand_idx_list = [clip_rand_idx[0], clip_rand_idx[1], rand_idx, clip_idx[0]]
            ref_clip_idx = random.choices(rand_idx_list, weights=[0.3, 0.3, 0.3, 0.1], k=1)[0]
            ref_clip_idxs.append(ref_clip_idx)

        box2_list, ref_box2_map = self.design_box2_for_clips_landmarks(clips, landmarks, ref_clip_idxs, width, height)

        if box2_list is None or ref_box2_map is None:
            return False

        ref_frmaes = {}
        for ref_clip_idx in ref_clip_idxs:
            ref_frmaes[ref_clip_idx] = None

        frame_idx = 0
        clip_index = 0

        l_frames = []
        m_frames = []
        if last_tag:
            last_start_idx = clips[-1][0]
            last_end_idx = clips[-2][1]
            last_box2 = box2_list[-1]
            last_box2_width = last_box2[2] - last_box2[0]
            last_box2_height = last_box2[3] - last_box2[1]
            if last_box2_width < 240 or last_box2_height < 240:
                print("Error: last_box2_width < 240 or last_box2_height < 240")
                return False
            last_x_min, last_y_min, last_x_max, last_y_max = last_box2

        start_idx, end_idx = clips[0][0], clips[0][1] + 1
        box2 = box2_list[0]
        box2_width = box2[2] - box2[0]
        box2_height = box2[3] - box2[1]
        if box2_width < 240 or box2_height < 240:
            print("Error: box2_width < 240 or box2_height < 240")
            return False

        x_min, y_min, x_max, y_max = box2
        save_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if last_tag:
                if frame_idx >= last_start_idx and frame_idx <= last_end_idx:
                    l_frames.append(frame[last_y_min:last_y_max, last_x_min:last_x_max])

            if frame_idx >= end_idx:
                m_frames = []
                clip_index += 1
                if clip_index >= len(clips):
                    break
                start_idx, end_idx = clips[clip_index][0], clips[clip_index][1] + 1
                box2 = box2_list[clip_index]
                box2_width = box2[2] - box2[0]
                box2_height = box2[3] - box2[1]
                if box2_width < 240 or box2_height < 240:
                    print("Error: box2_width < 240 or box2_height < 240")
                    return False
                x_min, y_min, x_max, y_max = box2

            m_frames.append(frame[y_min:y_max, x_min:x_max])
            if frame_idx in ref_clip_idxs:
                ref_box2 = ref_box2_map[frame_idx]
                ref_x_min, ref_y_min, ref_x_max, ref_y_max = ref_box2
                ref_frmaes[frame_idx] = frame[ref_y_min:ref_y_max, ref_x_min:ref_x_max]

            if len(m_frames) == self.num_frames:
                frames = m_frames
            elif len(m_frames) + len(l_frames) == self.num_frames and clip_index == len(clips) - 1:
                frames = l_frames + m_frames
            else:
                frames = []
            if len(frames) == self.num_frames:
                latents, lmk203_list = self.process_latent_x(frames, landmarks[start_idx:end_idx], box2)
                tensor_filename = f"latents_{save_index+1:03d}.pth"
                tensor_path = os.path.join(output_dir, tensor_filename)
                torch.save(latents, tensor_path)

                lmk203_list_path = os.path.join(output_dir, f"lmk203_{save_index+1:03d}.npy")
                lmk203 = {
                    "lmk203_list": lmk203_list
                }
                np.save(lmk203_list_path, lmk203)
                del frames
                m_frames = []
                save_index += 1

            frame_idx += 1
        cap.release()

        save_index = 0
        for frame_idx in ref_clip_idxs:
            if ref_frmaes[frame_idx] is None:
                return False
            y_latent, ref_context = self.process_latent_y(ref_frmaes[frame_idx])
            tensor_filename = f"latent_y_{save_index+1:03d}.pth"
            tensor_path = os.path.join(output_dir, tensor_filename)
            latent_y = {
                "y_latent": y_latent,
            }
            torch.save(latent_y, tensor_path)

            ref_context = {
                "ref_context": ref_context,
            }
            tensor_filename = f"ref_context_{save_index+1:03d}.pth"
            tensor_path = os.path.join(output_dir, tensor_filename)
            torch.save(ref_context, tensor_path)

            save_index += 1

        self.process_audio(audio_path, clips, len(landmarks), fps, output_dir)

        return True


def process_videos_on_gpu(gpu_id, video_paths, process_idx, args):
    torch.cuda.set_device(gpu_id)
    DEVICE = f"cuda:{gpu_id}"

    torch.cuda.empty_cache()
    print(f"[进程{process_idx}] GPU {gpu_id} (rank {gpu_id}) 可用: {torch.cuda.get_device_name(gpu_id)}")

    args.device = DEVICE

    dataset = VideoAudioDataset(args)
    data_root = Path(args.data_dir).expanduser().resolve()
    save_root = Path(args.save_dir).expanduser().resolve()
    aux_root = Path(args.aux_dir).expanduser().resolve() if args.aux_dir else data_root

    for video_path in tqdm(video_paths, desc=f"[GPU {gpu_id}][进程{process_idx}]"):
        try:
            video_path = str(video_path)
            vp = Path(video_path).resolve()
            try:
                rel = vp.relative_to(data_root)
            except ValueError:
                rel = Path(vp.name)
            audio_path = (aux_root / rel).with_suffix(".wav")
            lmk_path = (aux_root / rel).with_suffix(".npy")

            if not os.path.exists(audio_path):
                print(f"[GPU {gpu_id}][进程{process_idx}] 音频文件不存在: {audio_path}")
                continue

            if not os.path.exists(lmk_path):
                print(f"[GPU {gpu_id}][进程{process_idx}] landmark 文件不存在: {lmk_path}")
                continue

            output_dir = str(save_root / rel.with_suffix(""))

            if os.path.exists(output_dir):
                print(f"[GPU {gpu_id}][进程{process_idx}] 输出目录已存在，跳过: {output_dir}")
                continue

            os.makedirs(output_dir, exist_ok=True)
            success = dataset.run_getitem(str(video_path), str(audio_path), str(lmk_path), output_dir)
            if success:
                print(f"[GPU {gpu_id}][进程{process_idx}] 处理完成: {video_path}")
            else:
                print(f"[GPU {gpu_id}][进程{process_idx}] 处理失败: {video_path}")

        except Exception as e:
            print(f"[GPU {gpu_id}][进程{process_idx}] 处理 {video_path} 时出错: {e}")


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="多显卡并行视频预处理脚本")
    parser.add_argument("--frame_interval", type=int, default=21)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--landmark_model_path", type=str, default="./models/landmark.onnx")

    parser.add_argument("--audio_encoder_path", type=str, default="./models/chinese-wav2vec2-base")
    parser.add_argument("--vae_path", type=str, default="./models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--clip_checkpoint", type=str, default="./models/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    parser.add_argument("--clip_tokenizer", type=str, default="./models/Wan2.1-T2V-1.3B/xlm-roberta-large")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./dataset")
    parser.add_argument(
        "--aux_dir",
        type=str,
        default="",
        help="若设置，则从该镜像目录读取与视频相对路径一致的 wav/npy 辅助文件",
    )
    parser.add_argument("--num_gpus", type=int, default=6, help="使用的GPU数量")
    parser.add_argument("--procs_per_gpu", type=int, default=1, help="每个GPU上的进程数")
    parser.add_argument(
        "--first_gpu_id",
        type=int,
        default=0,
        help="第一个用于预处理的 GPU 编号（多进程会依次使用 first_gpu_id, first_gpu_id+1, ...）",
    )
    parser.add_argument(
        "--exclude_substr",
        type=str,
        default="",
        help="若设置，则路径中包含该子串的视频会被跳过（留空表示不过滤）",
    )

    args = parser.parse_args()

    data_root = Path(args.data_dir).expanduser().resolve()
    save_root = Path(args.save_dir).expanduser().resolve()
    video_files = list(data_root.rglob("*.mp4"))

    random.shuffle(video_files)
    new_video_files = []
    for video_file in video_files:
        video_file = Path(video_file).resolve()
        try:
            rel = video_file.relative_to(data_root)
        except ValueError:
            rel = Path(video_file.name)
        output_dir = save_root / rel.with_suffix("")
        if args.exclude_substr and args.exclude_substr in str(video_file):
            continue
        if not output_dir.exists():
            new_video_files.append(video_file)

    total_procs = args.num_gpus * args.procs_per_gpu
    video_chunks = [[] for _ in range(total_procs)]
    for idx, video_path in enumerate(new_video_files):
        video_chunks[idx % total_procs].append(video_path)

    print(f"使用 {args.num_gpus} 个GPU，每个GPU {args.procs_per_gpu} 个进程，总共 {total_procs} 个进程")
    for i, chunk in enumerate(video_chunks):
        print(f"进程 {i}: {len(chunk)} 个视频文件")

    processes = []
    for proc_idx in range(total_procs):
        gpu_id = args.first_gpu_id + (proc_idx // args.procs_per_gpu)

        p = mp.Process(
            target=process_videos_on_gpu,
            args=(gpu_id, video_chunks[proc_idx], proc_idx, args)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("所有视频处理完成！")


if __name__ == "__main__":
    main()
