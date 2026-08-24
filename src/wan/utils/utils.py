# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp
import numpy as np
import cv2
import imageio
import torch
import torchvision
from PIL import Image
import librosa
import soundfile as sf
import subprocess
import torchvision.transforms as v2
from einops import rearrange
from decord import VideoReader, cpu
import gc

__all__ = ['cache_video', 'cache_image', 'str2bool']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name



def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')

def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue

def convert_video_to_h264(input_video_path, output_video_path):
    subprocess.run(
        ['ffmpeg', '-i', input_video_path, '-c:v', 'libx264', '-c:a', 'copy', output_video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def is_video(path):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
    return os.path.splitext(path)[1].lower() in video_exts


def extract_specific_frames(video_path, frame_id, bbox=None):
    """bbox: optional dict with x_min, y_min, x_max, y_max (crop in original video coords)."""
    if is_video(video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        if frame_id < vr._num_frame:
            frame = vr[frame_id].asnumpy()  # RGB
        else:
            frame = vr[-1].asnumpy()
        del vr
        gc.collect()
        if bbox is not None:
            x_min = bbox["x_min"]
            y_min = bbox["y_min"]
            x_max = bbox["x_max"]
            y_max = bbox["y_max"]
            frame = frame[y_min:y_max, x_min:x_max]
        frame = Image.fromarray(frame)
    else:
        frame = Image.open(video_path).convert("RGB")
        if bbox is not None:
            x_min = bbox["x_min"]
            y_min = bbox["y_min"]
            x_max = bbox["x_max"]
            y_max = bbox["y_max"]
            frame = frame.crop((x_min, y_min, x_max, y_max))
    return frame


def resize_fit_letterbox(image, target_size):
    """
    与老版「先 crop 再驱动」一致：等比缩放后居中贴到 target 画布（letterbox），不拉伸。
    image: PIL Image 或 numpy (H,W,3)，返回 PIL Image。
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    orig_h, orig_w = image.height, image.width
    target_h, target_w = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new(image.mode, (target_w, target_h), (0, 0, 0))
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    return canvas


def extract_fragment_video(video_path, start_frame_id, end_frame_id, target_size, bbox=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    frame_id = 0
    target_h, target_w = target_size
    frame_process = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    x_min = bbox["x_min"] if bbox else 0
    y_min = bbox["y_min"] if bbox else 0
    x_max = bbox["x_max"] if bbox else None
    y_max = bbox["y_max"] if bbox else None
    while True:
        ret, frame = cap.read()
        if not ret and frame_id >= end_frame_id:
            break
        elif not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
               raise RuntimeError(f"Failed to open video: {video_path}")
            ret, frame = cap.read()
        if frame_id < start_frame_id:
            frame_id += 1
            continue
        if frame_id >= end_frame_id:
            break
        if frame_id >= start_frame_id and frame_id < end_frame_id:
            if bbox is not None:
                frame = frame[y_min:y_max, x_min:x_max]
                # 与老版「先 crop 再驱动」一致：等比缩放后居中贴到 target 画布，不拉伸
                crop_h, crop_w = frame.shape[0], frame.shape[1]
                scale = min(target_w / crop_w, target_h / crop_h)
                new_w = int(round(crop_w * scale))
                new_h = int(round(crop_h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
                x_off = (target_w - new_w) // 2
                y_off = (target_h - new_h) // 2
                canvas[y_off : y_off + new_h, x_off : x_off + new_w] = frame
                frame = canvas
            else:
                frame = cv2.resize(frame, (target_w, target_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame_process(frame)
            frames.append(frame)
        frame_id += 1
    cap.release()
    frames = torch.stack(frames, dim=0)
    frames = rearrange(frames, "T C H W -> C T H W")
    return frames

def get_video_codec(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    codec = result.stdout.decode().strip()
    return codec


def composite_face_back_to_video(
    original_video_path,
    generated_tensor,
    bbox_dict,
    output_path,
    audio_paths=None,
    fps=25,
    edge_feather_px=24,
):
    """
    将生成的人脸视频 (C T H W, value range -1~1) 按 bbox 贴回原视频并保存。
    bbox_dict: {x_min, y_min, x_max, y_max, width, height}（原视频坐标系）
    生成帧若为 letterbox（等比缩放+黑边），会先裁出内容区再缩放到 bbox，避免贴回后出现黑线/黑框。
    贴回策略与 backway 一致：内缩贴回区域 + 中心矩形 mask 经 dilate 后 stackBlur(81,81) 软边，避免边缘痕迹。
    edge_feather_px: 未使用（保留兼容），软边由 stackBlur 固定 81 控制。
    """
    from tqdm import tqdm
    x_min = bbox_dict["x_min"]
    y_min = bbox_dict["y_min"]
    x_max = bbox_dict["x_max"]
    y_max = bbox_dict["y_max"]
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    # 与 backway 一致：贴回时内缩，不贴满 bbox 边缘，减轻接缝
    ref_size = 512
    tx_offset = 0  # Reverted to 0 to match yijun branch behavior where frame_h//target_h was 0
    ty_offset = 0  # Reverted to 0 to match yijun branch behavior
    shrunk_w = crop_w - 2 * ty_offset
    shrunk_h = crop_h - 2 * tx_offset
    if shrunk_w <= 0 or shrunk_h <= 0:
        tx_offset = ty_offset = 0
        shrunk_w, shrunk_h = crop_w, crop_h

    # generated_tensor: (C T H W), -1~1
    gen = generated_tensor.permute(1, 2, 3, 0).cpu().numpy()
    gen = ((gen + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    T_gen, H_gen, W_gen, _ = gen.shape
    # 与 extract_fragment_video 的 letterbox 一致：生成帧中内容区为等比缩放后居中区域，裁出该区再贴回
    scale = min(W_gen / crop_w, H_gen / crop_h)
    new_w = int(round(crop_w * scale))
    new_h = int(round(crop_h * scale))
    x_off = (W_gen - new_w) // 2
    y_off = (H_gen - new_h) // 2

    # 边缘羽化 mask：与 backway 一致——中心矩形 255，dilate 后 stackBlur(81,81)，得到软边
    # mask 作用在「内缩后的贴回区域」上，形状 (shrunk_h, shrunk_w)
    n_x = int(crop_w / 512 * 12)  # Matched exactly to yijun backway.py calculation logic
    n_y = int(crop_h / 512 * 12)  # Matched exactly to yijun backway.py calculation logic
    mask = np.zeros((shrunk_h, shrunk_w), dtype=np.uint8)
    mask[n_y : shrunk_h - n_y, n_x : shrunk_w - n_x] = 255
    nate = crop_w / ref_size
    k = max(3, int(3 * nate))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.dilate(mask, kernel, iterations=3)
    # stackBlur 与 backway 一致，大核软边无痕迹
    if hasattr(cv2, "stackBlur"):
        mask = (cv2.stackBlur(mask, (81, 81)) / 255.0).astype(np.float32)
    else:
        mask = (cv2.GaussianBlur(mask.astype(np.float32), (81, 81), 0) / 255.0).astype(np.float32)
    mask = np.clip(mask, 0, 1)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {original_video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (orig_w, orig_h))
    # 贴回区域：内缩后的 roi（与 backway 一致）
    roi_shrunk = (
        slice(y_min + tx_offset, y_max - tx_offset),
        slice(x_min + ty_offset, x_max - ty_offset),
    )
    for t in tqdm(range(T_gen), desc="Composite"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = gen[t]
        face_content = face[y_off : y_off + new_h, x_off : x_off + new_w]
        face_resized = cv2.resize(face_content, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        # 只取内缩区域的人脸内容贴回
        face_inner = face_resized[
            tx_offset : crop_h - tx_offset,
            ty_offset : crop_w - ty_offset,
        ]
        orig_roi = frame[roi_shrunk].astype(np.float32)
        face_float = face_inner.astype(np.float32)
        blended = (face_float * mask + orig_roi * (1.0 - mask)).clip(0, 255).astype(np.uint8)
        frame[roi_shrunk] = blended
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    cap.release()
    writer.release()
    if audio_paths and len(audio_paths) > 0:
        duration = T_gen / fps
        tmp_no_audio = output_path + ".tmp_no_audio.mp4"
        os.rename(output_path, tmp_no_audio)
        crop_audio = output_path + ".crop_audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_paths[0], "-t", str(duration), crop_audio],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_no_audio, "-i", crop_audio,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", output_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        )
        for f in [tmp_no_audio, crop_audio]:
            try:
                os.remove(f)
            except Exception:
                pass
    return output_path


def split_wav_librosa(wav_path, segments, save_dir):
    y, sr = librosa.load(wav_path, sr=None)
    filename = wav_path.split('/')[-1].split('.')[0]
    save_list = []
    for idx, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        out_path = os.path.join(save_dir, filename + str(start) + '_' + str(end) + '.wav')
        sf.write(out_path, segment, sr)
        print(f"Saved {out_path}: {start}s to {end}s")
        save_list.append(out_path)
    return save_list
