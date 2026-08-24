#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理原视频：只跑一遍，只保存坐标类中间结果（.npy + _bbox.json），不保存 crop 视频。
生成阶段将根据 bbox 在线裁剪、生成后再贴回原视频。
"""
import repo_paths

repo_paths.ensure_project_src_path()

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from landmark_pipeline import (
    create_landmark_runner,
    create_scrfd_runner,
    process_video,
    landmark_process_videos,
)


def temporal_smooth(landmarks, win=5):
    """landmarks: (T, N, 2)"""
    kernel = np.ones(win) / win
    out = landmarks.copy()
    for i in range(landmarks.shape[1]):
        for d in range(2):
            out[:, i, d] = np.convolve(
                landmarks[:, i, d], kernel, mode="same"
            )
    return out


def segment_bbox_robust(landmarks_seg, q_low=0.03, q_high=0.97):
    """(T, N, 2) -> (xmin, ymin, xmax, ymax)"""
    xs = landmarks_seg[:, :, 0].reshape(-1)
    ys = landmarks_seg[:, :, 1].reshape(-1)
    xmin = np.quantile(xs, q_low)
    xmax = np.quantile(xs, q_high)
    ymin = np.quantile(ys, q_low)
    ymax = np.quantile(ys, q_high)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def design_bbox_from_landmarks(landmarks_seg, width, height, margin=16):
    """
    由片段级关键点 (T, 203, 2) 计算原图坐标系下的人脸 bbox。
    与 backway/test 中 design_box2_for_segment 逻辑一致，返回 (x_min, y_min, x_max, y_max)。
    注意：需要先提取人脸关键点 (114-137, 202)，与 src/wan/multitalk.py 中 get_target_mask 的逻辑一致。
    """
    # 先提取人脸关键点 (114-137, 202)，共 25 个点
    face_indices = list(range(114, 138)) + [202]
    face_landmarks = np.array([
        [landmarks_seg[i][j] for j in face_indices]
        for i in range(len(landmarks_seg))
    ], dtype=np.float64)
    pxmin, pymin, pxmax, pymax = segment_bbox_robust(face_landmarks)
    w = pxmax - pxmin
    h = pymax - pymin
    side_base = max(w, h)
    side = side_base * 2.0 + 2 * margin
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
        x_min, x_max = 0.0, float(width)
    if side <= height:
        y_min = max(0.0, min(y_min, float(height) - side))
        y_max = y_min + side
    else:
        y_min, y_max = 0.0, float(height)
    x_min = max(0.0, min(x_min, float(width)))
    x_max = max(0.0, min(x_max, float(width)))
    y_min = max(0.0, min(y_min, float(height)))
    y_max = max(0.0, min(y_max, float(height)))
    return int(x_min), int(y_min), int(x_max), int(y_max)


def process_one_video_no_crop(video_path, lmk_runner, scrfd_runner):
    """
    对单个原视频：提关键点 -> 算 bbox -> 转成 crop 空间关键点，只写 .npy 和 _bbox.json。
    返回 (success, landmarks_crop, bbox_dict) 或 (False, None, None)。
    """
    import cv2
    landmarks_list = process_video(video_path, lmk_runner, scrfd_runner)
    if landmarks_list is None:
        return False, None, None
    landmarks = np.array(landmarks_list)
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    smoothed = temporal_smooth(landmarks, win=5)
    x_min, y_min, x_max, y_max = design_bbox_from_landmarks(smoothed, width, height)
    # 关键点转到 crop 空间（与 crop 区域左上角对齐）
    landmarks_crop = landmarks.copy()
    landmarks_crop[:, :, 0] -= x_min
    landmarks_crop[:, :, 1] -= y_min
    bbox_dict = {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": width,
        "height": height,
    }
    return True, landmarks_crop, bbox_dict


def process_videos_on_gpu_no_crop(gpu_id, video_chunk, proc_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    lmk_runner = create_landmark_runner(device_id=0)
    scrfd_runner = create_scrfd_runner()
    success_count = 0
    error_count = 0
    for video_path in tqdm(video_chunk, desc=f"进程{proc_idx}(GPU{gpu_id})"):
        try:
            base = video_path.with_suffix("")
            npy_path = base.with_suffix(".npy")
            bbox_path = base.parent / (base.name + "_bbox.json")
            if npy_path.exists() and bbox_path.exists():
                print(f"进程{proc_idx}: 跳过已存在 {video_path}")
                continue
            ok, landmarks_crop, bbox_dict = process_one_video_no_crop(
                video_path, lmk_runner, scrfd_runner
            )
            if ok:
                np.save(npy_path, landmarks_crop)
                with open(bbox_path, "w", encoding="utf-8") as f:
                    json.dump(bbox_dict, f, indent=2)
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            print(f"进程{proc_idx}: 处理 {video_path} 出错: {e}")
    print(f"进程 {proc_idx} (GPU{gpu_id}) 完成，成功: {success_count}, 失败: {error_count}")


def preprocess_videos_no_crop(data_dir, num_gpus=1, procs_per_gpu=1):
    video_files = (
        list(Path(data_dir).rglob("*.mp4"))
        + list(Path(data_dir).rglob("*.avi"))
        + list(Path(data_dir).rglob("*.mov"))
        + list(Path(data_dir).rglob("*.mkv"))
    )
    if not video_files:
        print(f"未在 {data_dir} 下找到视频文件")
        return
    n_procs = num_gpus * procs_per_gpu
    chunks = [[] for _ in range(n_procs)]
    for i, p in enumerate(video_files):
        chunks[i % n_procs].append(p)
    import multiprocessing as mp
    processes = []
    for proc_idx in range(n_procs):
        gpu_id = proc_idx // procs_per_gpu
        p = mp.Process(
            target=process_videos_on_gpu_no_crop,
            args=(gpu_id, chunks[proc_idx], proc_idx),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("预处理完成（仅保存 .npy 与 _bbox.json，未保存 crop 视频）")


def main():
    parser = argparse.ArgumentParser(description="原视频预处理：只保存关键点与 bbox，不保存 crop 视频")
    parser.add_argument("--data_dir", type=str, default="", help="原视频所在目录")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--procs_per_gpu", type=int, default=1)
    args = parser.parse_args()
    if args.data_dir:
        preprocess_videos_no_crop(args.data_dir, args.num_gpus, args.procs_per_gpu)


if __name__ == "__main__":
    main()
